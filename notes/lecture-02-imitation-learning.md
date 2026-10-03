# Lecture 02 — Imitation Learning

- 课程版本：Spring 2026（2025 有录像）
- 日期：2026-04-03
- 官方材料：[02_cs224r_imitation_2026.pdf](../slides/spring2026/02_cs224r_imitation_2026.pdf) ·
  [2025 版](../slides/spring2025/02_cs224r_imitation_2025.pdf) ·
  [Lecture 2 视频](https://www.youtube.com/watch?v=WxRDyObrm_M) ·
  作业：HW1（Flappy Bird，action chunking）
- 状态：初稿

## 学习目标

- 写出 BC 的目标函数并解释其失败模式：compounding error 与 multimodal averaging；
- 会用生成式策略处理多模态：MoG、离散自回归、diffusion、flow matching；
- 理解 action chunking 与 receding horizon control，知道 `T=20/K=10` 的意义；
- 推出 flow matching 的插值路径、目标速度与 Euler 采样；
- 描述 DAgger 的聚合数据集迭代与为什么它能缓解分布偏移。

## 先修知识

- L1 的 MDP 与轨迹；极大似然；高斯分布与重参数化；基础 ODE/积分离散化；
- 反向传播与 MLP。

## 核心概念

### 1. 行为克隆（BC）

给定专家数据 `D = {(s_i, a_i)}`（可来自轨迹），学策略 `π_θ(a | s)`：

```text
确定性策略（回归）：  L_MSE = (1/N) Σ_i ||π_θ(s_i) − a_i||²
随机策略（MLE）：     L_NLL = −(1/N) Σ_i log π_θ(a_i | s_i)
```

BC 假设演示的数据分布覆盖测试时遇到的状态分布。这个假设在闭环控制中不成立。

### 2. 复合误差（compounding error）

设每一步策略犯错的概率约为 `ε`。BC 训练时只见过专家分布的状态，测试时一旦偏离，
策略进入训练分布之外的状态，错误被进一步放大。教材中的经典结果（Ross & Bagnell）：

- 朴素 BC 的期望错误随时间近似**二次增长** `O(εT²)`；
- 加入专家纠正/在线聚合后可以降到 `O(εT)`。

直觉：误差进入状态，通过动力学再次影响下一步输入，形成反馈回路。

### 3. 多模态动作

当演示来自不同策略/不同风格（左绕、右绕都能成功），条件分布 `p(a | s)` 是多峰的。
MSE 回归收敛到**条件均值**，可能落在两个峰之间的「撞墙」动作上。

解决思路：把策略变成**生成模型**，对 `p(a | s)` 建模而不是回归单点：

| 方式 | 说明 | 代价 |
| --- | --- | --- |
| 混合高斯（MoG） | 输出 K 个高斯分量 | 模态数需预设，K 大难训 |
| 离散化 + 自回归 | 每维离散化后逐维建模 | 维度灾难、推理慢 |
| Diffusion policy | 迭代去噪生成动作 | 采样步数多，训练较复杂 |
| Flow matching | 回归速度场，ODE 采样 | 实现简单，常与 diffusion 相当或更好 |

### 4. Action chunking

现代机器人策略通常一次预测 `T` 个未来动作 `a_{t:t+T}`，只执行前 `K < T` 个
（receding horizon control），然后重新查询策略：

```text
预测 T=20 个动作 → 执行 K=10 个 → 用最新观测重新预测
```

好处：减少每步推理成本、降低短视抖动、缓解单步误差累积；执行段仍是开环，但重查保持反馈。
代表工作：ACT（Zhao et al., ALOHA）、Diffusion Policy（Chi et al.）。

### 5. Diffusion 直觉

前向过程给数据逐步加噪：`q(a_τ | a_{τ-1}) = N(√(1−β_τ) a_{τ-1}, β_τ I)`；
训练网络预测噪声（或 score），推理时从纯噪声逐步去噪。

- 训练目标（DDPM 简化版）：`E[||ε − ε_θ(a_τ, τ)||²]`；
- 采样需要几十到上百步，动作维低时通常可接受。

### 6. Flow matching

设干净动作块 `a_1`，噪声 `a_0 ~ N(0, I)`，定义线性插值路径（rectified flow 特例）：

```text
a_τ = τ · a_1 + (1 − τ) · a_0,     τ ~ U(0,1)
```

目标速度是常向量：

```text
v* = d a_τ / dτ = a_1 − a_0
```

网络 `v_θ(s, a_τ, τ)` 的损失：

```text
L_FM = E_{(s,a_1)∈D, a_0~N(0,I), τ~U(0,1)} || v_θ(s, a_τ, τ) − (a_1 − a_0) ||²
```

采样：从 `a_0 ~ N(0, I)` 出发，用 Euler 积分 `n` 步：

```text
a_{τ + 1/n} = a_τ + (1/n) · v_θ(s, a_τ, τ)
```

最后把 `a_1` 截断到动作范围（Flappy Bird 为 `[0,1]`）。flow matching 与 diffusion
都学「噪声 → 数据」的变换，但 FM 直接回归速度、无需噪声调度，训练更稳定。

### 7. DAgger

DAgger（Dataset Aggregation）交互式修正分布偏移：

1. 用当前策略 `π_θ` rollout，得到它访问的状态序列 `s_1…s_T`；
2. 在这些状态上查询**专家** `π*` 标注动作，形成新数据
   `D_i = {(s, π*(s)) | s ~ D_{π_θ}}`；
3. 聚合 `D ← D ∪ D_i`，在聚合数据上重新训练（通常监督学习即可）；
4. 重复多轮。

关键点：数据分布是「学习者诱导的」，标签来自专家，因此错误状态获得直接监督。
`D_Agger` 满足「专家在访问分布上的最优性」，训练分布逐轮靠近测试分布。

- 代价：需要专家可查询（人、确定性规则、特权控制器）；
- 变体：SMILe（用纠正分布）、HG-DAgger（人在环纠正）、ThriftyDAgger。

### 8. 演示从哪里来

- 人类遥操作/VR、动捕；机器人演示数据（ALOHA）；
- 人类视频（存在 embodiment gap：外观、关节结构、动作空间不同）；
- 专家控制器/规划器合成数据；
- 仿真 + sim-to-real（L16）。

## 关键推导：BC 误差为何是二次的

简化的有限 MDP 分析（Ross & Bagnell 2010）：记策略在分布内单步犯错概率为 `ε`。
BC 的测试分布与训练分布不一致，第 `t` 步累计犯错概率上界约为 `εt`；每条轨迹长度 `T`，
总错误 `Σ_t Pr(第 t 步出错) ≈ ε · T(T+1)/2 = O(εT²)`。
DAgger 通过不断把测试分布纳入训练集，使第 `t` 步出错概率不随 `t` 增长，总量回到 `O(εT)`。

Flappy Bird 的 HW1 实践里，hard mode 的多模态间隙使 MSE 回归显著退化，正是上面两个问题的叠加。

## 实现映射（HW1）

| 概念 | 文件 | TODO | 验收 |
| --- | --- | --- | --- |
| BC 策略 | `starter-code/networks.py` | `BCPolicy.__init__/forward`：3 层 MLP + Sigmoid | 输出 shape `(B, 20)` |
| BC 损失 | `starter-code/losses.py` | `bc_loss`：MSE | loss 随训练下降 |
| Flow schedule | `starter-code/networks.py` | `FlowMatchingSchedule.interpolate/sample` | `a_τ, v` shape 正确 |
| FM 损失 | `starter-code/losses.py` | `flow_matching_loss` | 速度回归 loss |
| DAgger | `starter-code/dagger.py` | `DeterministicExpert.act`、`rollout_episode`、`rollout_and_relabel` | 5 轮学习曲线上升 |

复现命令：

```bash
python main.py --method bc_reg --env easy
python main.py --method bc_reg --env hard
python main.py --method bc_flow --env hard
python main.py --method dagger --env hard --n-iter 5
python main.py --plot
```

## 易错点与反例

- **flow 插值方向搞反**：`τ=0` 是噪声、`τ=1` 是数据；速度目标是 `a_1 − a_0`，不是反过来；
- **Euler 步长**：`a ← a + (1/n)·v`，注意 `1/n` 而不是 `n`；`n` 太小时采样噪声大；
- **输出未 clamp**：Flappy Bird 动作范围 `[0,1]`，不 clamp 会导致 PD 控制器饱和异常；
- **DAgger 忘记聚合旧数据**：只训最新一轮会导致灾难性遗忘；
- **DeterministicExpert 必须与随机专家一致**：hard mode 下专家有随机性，需给出确定化规则，
  否则回归目标仍然多峰；
- **评估口径**：HW1 用 50 个 episode 的存活长度均值 ± 标准差，不是训练 loss。

## 实验或练习

1. 在 hard mode 上对比 MSE 回归与 flow matching 的存活长度，解释差异来源；
2. 画 DAgger 每轮的评估曲线，并叠加 BC 基线水平线；
3. 把 action chunk 的执行的 `K` 从 10 改成 1/20，观察平滑性与反应速度的权衡。

## 高频问题

- 问：BC 为什么不能保证好策略？答：分布偏移导致复合误差；多模态导致均值坍缩。
- 问：DAgger 与 BC 的本质区别？答：数据分布由学习者诱导并聚合，标签仍来自专家。
- 问：flow matching 相比 diffusion 的优势？答：直接回归速度场，无噪声调度，训练/实现更简单。
- 问：action chunking 的代价？答：开环执行段内无法即时纠错，`K` 越大越不灵活。

## 本讲小结

模仿学习把 RL 退化成监督学习，代价是分布偏移与多模态。解法分别是生成式策略（建模多峰）
与 DAgger（在线修正分布）。HW1 恰好覆盖这两条主线。从 L3 开始，我们不再依赖专家，
而是用奖励信号直接优化策略。

## 延伸阅读

- Ross, Gordon, Bagnell, *A Reduction of Imitation Learning and Structured Prediction to
  No-Regret Online Learning* (DAgger), 2011.
- Chi et al., *Diffusion Policy*, 2024；Zhao et al., *ALOHA*, 2023.
- Lipman et al., *Flow Matching for Generative Modeling*, 2023.
- 官方 slides 与 HW1 handout：`assignments/spring2026/homework1-imitation/`。
