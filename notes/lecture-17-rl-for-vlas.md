# Lecture 17 — RL for Robot Foundation Models / VLAs

- 课程版本：Spring 2026
- 日期：2026-05-27
- 官方材料：[17_cs224r_rl_vlas_2026.pdf](../slides/spring2026/17_cs224r_rl_vlas_2026.pdf) ·
  参考：OpenVLA、π0.5、GR00T、diffusion steering、advantage-conditioned BC
- 状态：初稿

## 学习目标

- 定义 VLA（Vision-Language-Action）模型，说明它为何是机器人基础模型的主流形态；
- 解释「IL 预训练 + RL 后训练」的动机与三类难点；
- 写出离线 RL 微调 VLA 的配方：监督值函数 + advantage-conditioned 行为克隆；
- 理解在线 RL 的轻量方案：残差策略、diffusion steering、冻结 VLA 只学编辑器；
- 总结 VLA RL 的开放问题（TD 学习、探索、真实世界安全、评测）。

## 先修知识

- L2 模仿学习/动作生成；L5 PPO；L7 离线 RL；L16 sim2real。

## 核心概念

### 1. VLA 是什么

VLA 把视觉 + 语言指令映射为动作序列，通常由预训练 VLM + 动作头（自回归 token、
diffusion/flow head）构成。训练以模仿学习为主（机器人演示 + 人类视频），
代表：OpenVLA、π0/π0.5、GR00T N1/N2、RT-2。

### 2. 为什么 VLA 需要 RL

- IL 在多样化数据上常**平台在 ~80% 成功率**：长尾失败、分布漂移、任务约束难以演示；
- RL 能用奖励继续提升，尤其擅长修正「差一点成功」的边界行为；
- 类比 LLM：预训练/SFT 打底，RL 后训练提升任务表现。

### 3. 为什么 VLA 上做 RL 难

1. **模型大**：全参数 RL 微调显存/算力昂贵；
2. **采样贵**：真实机器人 rollout 慢且不可并行；
3. **奖励稀疏**：成功/失败信号，偏置在长时域；
4. **安全**：探索可能损坏硬件/环境；
5. **分布**：diffusion/flow 策略的似然难算，PPO 类方法不直接适用。

### 4. 离线 RL 微调配方（推荐起点）

以 π0.5 之后的实践为例：

```text
1. 用成功/失败数据训练值函数 V(s)（Monte Carlo 回报：成功 1、失败 0）
2. 计算优势 A(s,a) ≈ V(s) − 或 return − V
3. 用 advantage-conditioned 监督学习更新 VLA：
   在 prompt/条件里加入「好/坏」token（advantage conditioning），只对好样本模仿
4. 迭代：新策略收集数据 → 更新 V → 再训练
```

优点：不需要对 VLA 做 policy gradient；复用监督训练管线；
值函数是监督式的（MC），稳定。风险：MC 值高方差、离线数据覆盖有限。

### 5. 在线 RL 的轻量化路线

- **PPO 直接微调**：OpenVLA / π0.5 上的实验显示可行但昂贵、调参敏感；
- **Diffusion steering**：冻结 VLA，学一个小策略输出噪声/编辑向量，使采样动作更好；
- **残差策略（residual policy）**：`a = a_VLA + a_res(s)`，只训练小网络；
- **动作编辑器**：学高斯策略对 VLA 动作做局部修正（scale/shift/selection）。

共同主题：**不要动大模型**，把 RL 放在小策略/小价值网络上，迭代式收集数据。

### 6. 迭代式离线 RL 循环

```text
π_0 (SFT VLA)
  → rollout 收集数据（含失败）
  → 训练 V(s)（MC）
  → advantage 加权/条件微调 π
  → 新数据 ... 重复
```

这是「iterated offline RL」范式：每轮都用当前策略数据，避免纯离线分布问题，
同时控制真机风险（少量多次、成功经验优先）。

### 7. 训练阶段观察

- 初期：MC 值函数帮助策略快速爬升；
- 后期：需要更好的信用分配（TD、过程奖励）以突破平台；
- 不同任务的最优点不同，配方需要按任务调整。

## 关键推导：为什么 advantage conditioning 能替代 policy gradient

离线数据下策略梯度需要 `∇log π_θ(a|s)`（对 diffusion/flow 不方便）与行为分布修正。
若把优势当作**条件**：`π_θ(a | s, A>0)`，则只需对「有正优势的样本」做条件 BC：

```text
L = −E_{(s,a)~D, A(s,a)>0}[ log π_θ(a | s, success) ]  （示意）
```

推理时用 `success` 条件生成动作。它与过滤 BC 的区别在于使用连续/分级条件，
保留多样性并可在推理时控制行为。

## 实现映射

| 概念 | 材料 | 说明 |
| --- | --- | --- |
| 动作生成策略 | Default Project 的 diffusion/flow 采样 | 同理适用于 VLA head |
| 离线 RL | HW3 AWAC/IQL | 值函数 + 加权回归思路 |
| 优势加权 | `awac_agent.py`/`iql_agent.py` | VLA 版只是把策略换成大模型 |
| 迭代式循环 | 期末项目 | 可复现小规模版本 |

## 易错点与反例

- **直接全参数 PPO**：显存/时间不可行，先用小策略/冻结主干；
- **MC 值方差**：稀疏奖励下 `V` 估计噪声大，需大量数据与平滑；
- **只用成功数据**：会忽略「接近失败」的宝贵信号；
- **忽略安全**：探索必须加物理约束/人工干预；
- **评测只看训练任务**：VLA 的价值在于泛化，需 held-out 指令/场景。

## 实验或练习

1. 用一个小 diffusion/flow 策略（如 Flappy Bird 的 flow 策略）实现 advantage conditioning
   的玩具版本：把回报高低作为条件 token 训练，观察能否只生成高回报动作；
2. 阅读 π0.5 或 OpenVLA+RL 论文，整理「数据 → 值函数 → 条件训练」的流程图；
3. 讨论：残差策略与 diffusion steering 在什么任务上更优？

## 高频问题

- 问：VLA 为什么卡在 80%？答：IL 拟合演示分布，长尾/边界情况缺数据。
- 问：VLA RL 最难的成本项？答：真机采样与奖励，其次是模型大小。
- 问：离线 RL 微调 VLA 的关键？答：监督式 V 学习 + 优势条件的行为克隆。
- 问：在线 RL 的轻量方案？答：冻结 VLA，学残差/噪声编辑器/价值函数。
- 问：与经典 RL 的关系？答：同一套值/优势概念，只是函数近似换成了大模型。

## 本讲小结

VLA 时代把 RL 后训练推向「大数据 + 大模型 + 小 RL 策略」的范式。离线迭代、优势条件训练、
轻量在线编辑是三条主线。最后一讲回到全局：如何做研究、还有哪些开放问题。

## 延伸阅读

- Kim et al., *OpenVLA*, 2024；Physical Intelligence, *π0.5*, 2025.
- NVIDIA, *GR00T N1*, 2025；相关 RL 微调论文见 slides。
- 官方 slides：`slides/spring2026/17_cs224r_rl_vlas_2026.pdf`。
