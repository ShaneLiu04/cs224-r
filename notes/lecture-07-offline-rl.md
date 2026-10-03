# Lecture 07 — Offline Reinforcement Learning

- 课程版本：Spring 2026（2025 有录像）
- 日期：2026-04-22
- 官方材料：[07_cs224r_offline_rl_2026.pdf](../slides/spring2026/07_cs224r_offline_rl_2026.pdf) ·
  [2025 版](../slides/spring2025/07_cs224r_offline_rl_2025.pdf) ·
  [Lecture 7 视频](https://www.youtube.com/watch?v=lRDaXnPIzks) ·
  论文：AWR、IQL、CQL、AWAC；作业：HW3
- 状态：初稿

## 学习目标

- 定义离线 RL 问题，解释为什么不能直接用 off-policy actor-critic；
- 说清 OOD 动作上的 Q 值误差如何被 bootstrap 放大；
- 写出 AWR/AWAC 的 advantage 加权回归目标，理解温度 `λ` 的作用；
- 推导 expectile 回归，写出 IQL 的 `V`/`Q`/策略三件套；
- 比较 IQL 与 AWAC/CQL，理解 stitching 现象以及在 PointMass 上如何检验。

## 先修知识

- L4–L6：actor-critic、Q-learning、Bellman 方程、BC 与 AWR 的关系。

## 核心概念

### 1. 问题设定

给定固定数据集 `D = {(s, a, r, s')}`（由未知行为策略 `µ` 收集），
不与环境交互，学习最大化期望回报的策略。典型来源：日志数据、人类演示、旧策略回放、
其他任务数据。

### 2. 为什么不能直接用 off-policy 方法

以 SAC 为例，critic 目标包含 `a' ~ π_θ(·|s')`：当 `π_θ` 偏离 `µ` 时，`a'` 可能是数据中
从未出现过的动作（OOD），网络对它的 Q 估计没有监督信号。Bootstrapping 让错误沿轨迹
传播：`Q` 高估 OOD 动作 → 策略偏向 OOD 动作 → 下轮又查询 OOD 动作 → 误差放大（extrapolation error）。

模仿学习只学数据内动作，所以没有 OOD 问题，但**无法超越数据**（不能 stitching，
也不理解奖励）。离线 RL 要同时利用奖励并约束在数据支撑集附近。

### 3. 简单基线：加权模仿

**BC**：只匹配动作，无视奖励。

**Filtered BC**：只用高回报轨迹。数据利用率低、难处理次优轨迹拼接。

**AWR（Advantage-Weighted Regression）**：用指数的优势权重：

```text
L_π(θ) = −E_{(s,a)~D}[ log π_θ(a|s) · exp( (1/λ) · A^{π_k}(s,a) ) ]
```

- 优势越高（比当前策略平均好）的动作权重越大；
- `λ → ∞` 退化为 BC；`λ → 0` 只模仿最优动作；
- 优点：完全不需要 OOD 查询，天然离线；
- 缺点：需要估计 `A`。若用单步/MC 回报，优势精度有限；若用 TD，则又回到 OOD 问题。

### 4. 用 TD 估计优势，避免 OOD：IQL

IQL（Implicit Q-Learning）的关键技巧是**expectile 回归**：不学条件均值，而学条件分布的
上分位数。

Expectile 定义：

```text
L_2^ζ(µ) = |ζ − 1{µ ≤ 0}| · µ²
argmin_m E[ L_2^ζ(X − m) ]
```

- `ζ = 0.5` 时退化为 MSE（均值）；
- `ζ > 0.5` 时对 `X > m` 的样本加权更多，解为上 expectile（大数据侧）；
- 直觉：让 `V_ψ(s)` 逼近 `max_{a' ~ 数据} Q(s,a')` 的近似上界，而无需查询 OOD 动作。

三件套：

```text
L_V(ψ) = E_{(s,a)~D}[ L_2^ζ( Q_θ̄(s,a) − V_ψ(s) ) ]          # 拟合 Q 的上 expectile
L_Q(θ) = E_{(s,a,s')~D}[ ( r(s,a) + γ V_ψ(s') − Q_θ(s,a) )² ]  # 只对数据动作回归，用 V 做目标
L_π(θ) = −E_{(s,a)~D}[ log π_θ(a|s) · exp( β (Q_θ̄(s,a) − V_ψ(s)) ) ]  # AWR 抽取策略
```

- `Q̄` 是 target Q 网络；
- critic 只在数据动作 `a ~ D` 上训练，**不采样 `a'`**，因此没有 OOD 动作；
- `V` 只是「数据内优势的 amortized 估计」，不做 `max` 的显式优化；
- `ζ` 是核心超参：`ζ=0.9` 常用，太低接近均值，太高会被数据中的极端值主导。

IQL 的训练目标可以理解为「算策略改进而不真的在 OOD 动作上 query」。

### 5. CQL（Conservative Q-Learning）

另一条路线：显式压低 OOD 动作的 Q 值，让 Q 成为真实值的下界：

```text
L_CQL = α ( E_{s~D, a~µ}[Q(s,a)] − E_{(s,a)~D}[Q(s,a)] ) + L_TD
```

- 第一项压 OOD 分布（如当前策略 `µ=π_θ`）的 Q；
- 第二项抬数据动作的 Q；
- `α` 控制保守程度。CQL 简单但需要采样动作，计算量较大。

AWAC（HW3 使用）介于 AWR 与 SAC 之间：critic 用 Bellman 更新（但目标动作来自当前策略，
带来 OOD 风险），策略用 advantage 加权回归。实践中常加 BC 正则或使用数据动作近似。

### 6. 离线 RL 的评估与 stitching

- **评估**：只用数据内回报会高估；主流做法是在相同环境做少量在线评估，
  报告 `Eval_AverageReturn`；
- **stitching**：数据中可能只有「A→B 的最优片段」与「B→C 的最优片段」，
  离线 RL 应能拼出 A→C 的更好策略。IQL 由于 `Q` 的目标只基于数据内转移并传播，
  在 PointMass 这类任务上通常表现出 stitching；Filtered BC 做不到。

## 关键推导：IQL 为什么能避开 OOD

如果直接学 `max`：`Q(s,a) = r + γ max_{a'} Q(s',a')`，就需要评估 `argmax` 动作，
它可能不在数据中。IQL 用 `V_ψ(s')` 代替 `max_{a'} Q(s',a')`：

```text
Q target = r + γ V_ψ(s')
```

而 `V_ψ(s)` 通过 expectile 回归从 `Q(s,a)`（a 来自数据）隐式逼近上界。
因此整条更新链只涉及数据动作。代价是 expectile 只是 max 的近似：`ζ` 越大越接近 max，
方差也越大。

## 实现映射（HW3）

| 概念 | 文件 | TODO | 验收 |
| --- | --- | --- | --- |
| AWAC critic | `assignments/spring2026/homework3-offline-rl/starter-code/cs224r/critics/awac_critic.py` | clipped double-Q 目标 | 3 seeds 平均回报 |
| AWAC agent | `.../agents/awac_agent.py` | 加权 NLL 策略更新 | antmaze-umaze/medium-diverse |
| IQL critic | `.../critics/iql_critic.py` | expectile loss、V 更新、Q 更新 | `ζ=0.2/0.9` 对比 |
| IQL agent | `.../agents/iql_agent.py` | AWR 策略抽取 | 同上 |
| Stitching | `offline_datasets/pointmass_stitching_dataset.npz` | 对比 Filtered BC | 回报是否超过数据最优 −46 |

运行命令（官方要求 Modal）：

```bash
modal run --detach modal_train_para.py --algo awac \
  --env-name antmaze-umaze-v0 --exp-name awac_antmaze_umaze --use-wandb

modal run --detach modal_train_para.py --algo iql \
  --env-name antmaze-umaze-v0 --iql-expectile 0.9 \
  --exp-name iql_zeta_0.9_umaze --use-wandb
```

## 易错点与反例

- **critic 对策略采样动作做 Bellman 目标（经典 off-policy）**：离线场景会 OOD 爆掉；
- **expectile 实现方向写反**：`ζ>0.5` 应给 `µ>0`（预测偏低）更大权重，最终学上分位数；
- **`V` 忘了 stop-gradient**：`V` 的目标必须 detach；
- **策略更新用了错误的 A**：IQL 的策略权重是 `exp(β(Q−V))`，其中 `Q` 用 target 网络；
- **评估只看训练 loss**：必须看环境评估回报；
- **将 AWAC 的 `a'` 来自当前策略**：这正是它与 IQL 的差别，也是它在困难任务上更脆弱的原因。

## 实验或练习

1. 在 PointMass stitching 数据上跑 IQL 与 Filtered BC，比较最大回报与轨迹图，判断是否出现
   「拼接出数据中不存在的路径」；
2. 扫 IQL expectile `{0.2, 0.5, 0.9}`，画回报曲线，解释趋势；
3. AWAC 与 IQL 在 antmaze-medium 上的差距，从 OOD 动作角度给出 3 句话解释。

## 高频问题

- 问：离线 RL 的核心困难？答：OOD 动作上的价值误差被 bootstrap 放大。
- 问：AWR 与 BC 的区别？答：按优势指数加权，好动作多模仿。
- 问：IQL 的 expectile 在近似什么？答：数据内动作的 `Q` 上分位数，用来代替 `max`。
- 问：为什么 IQL 不需要采样 `a'`？答：用 `V(s')` 做 Bellman 目标，训练只用数据动作。
- 问：CQL 与 IQL 的差别？答：CQL 显式惩罚 OOD 动作价值；IQL 结构性避免查询 OOD。
- 问：什么是 stitching？答：把数据中不同轨迹的优质片段拼成更优策略。

## 本讲小结

离线 RL = 「利用奖励」+「不离开数据支撑」。AWR 用加权模仿，IQL 用 expectile 把 `max` 隐式化，
CQL 用惩罚压 OOD。HW3 的 AWAC/IQL 对比与 PointMass stitching 是理解这些设计的最佳实验。
下一讲追问一个更根本的问题：奖励本身从哪来？

## 延伸阅读

- Kostrikov et al., *Offline RL with Implicit Q-Learning* (IQL), 2021.
- Nair et al., *AWAC*, 2020；Peng et al., *AWR*, 2019.
- Kumar et al., *Conservative Q-Learning* (CQL), 2020.
- Fu et al., *D4RL*, 2020；官方 HW3 handout。
