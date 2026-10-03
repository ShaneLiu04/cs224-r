# Lecture 04 — Actor-Critic Methods

- 课程版本：Spring 2026（2025 有录像）
- 日期：2026-04-10
- 官方材料：[04_cs224r_actor_critic_2026.pdf](../slides/spring2026/04_cs224r_actor_critic_2026.pdf) ·
  [2025 版](../slides/spring2025/04_cs224r_actor_critic_2025.pdf) ·
  [Lecture 4 视频](https://www.youtube.com/watch?v=oejFZShW9hU) ·
  论文：Sutton et al. (1999)、PPO (2017)、GAE (2016)，作业：HW2 P2（PPO）
- 状态：初稿

## 摘要

Actor-critic 用学出来的值函数替代策略梯度里的 Monte Carlo 回报，从而在偏差与方差之间取得
可调平衡。本讲先厘清 `V/Q/A` 三者的关系，再比较 Monte Carlo、TD 与 n-step 估计，然后推导
GAE——把「用几步 bootstrap」变成一个由 `λ` 控制的连续旋钮。最后组装 batch actor-critic 循环，
并说明多轮更新与 replay 如何把它推向 off-policy（L5 的 PPO/SAC）。

## 一页速览卡（面试前 5 分钟）

**必须记住的五件事**

1. `A^π(s,a)=Q^π(s,a)−V^π(s)`；策略梯度的理想权重是优势，不是原始回报。
2. MC 无偏但高方差；TD 有偏低方差可在线；n-step 折中。
3. TD 残差 `δ_t=r_t+γ(1−d_t)V(s_{t+1})−V(s_t)`；终止状态不 bootstrap。
4. GAE：`Â_t=δ_t+γλ(1−d_t)Â_{t+1}`；`λ=0` 是 TD，`λ=1` 是 MC 优势。
5. 值目标是常数（stop-gradient）；`returns = advantages + values`。

**必须能默写的公式**

```text
V^π(s)=E[Σγ^t r | s],  Q^π(s,a)=E[·| s,a],  A^π=Q^π−V^π
δ_t = r_t + γ(1−d_t)V_φ(s_{t+1}) − V_φ(s_t)
Â_t = δ_t + γλ(1−d_t)Â_{t+1},        Â_T = 0
L_V(φ) = E[(V_φ(s_t) − (Â_t + V_φ(s_t))_sg)²],   L_π(θ)=−E[logπ_θ(a_t|s_t)·Â_t]
```

**一句话区分**

- MC vs TD：整条轨迹回报 vs 一步 bootstrap；无偏/高方差 vs 有偏/低方差。
- `λ` 两端：`λ=0` 纯 TD（偏差大），`λ=1` 纯 MC（方差大）。
- 同一批数据更新轮数：1 轮 = on-policy 近似；多轮 = 需要比率修正（PPO 的 clip）。

## 学习目标

- 区分 `V^π`、`Q^π`、`A^π` 的定义与相互关系；
- 比较 Monte Carlo 与 bootstrap 估计值函数的偏差/方差；
- 推导 n-step 回报与 TD 残差，写出 GAE 递推；
- 组装一个 batch actor-critic 训练循环，知道每一步的 shape 与 stop-gradient；
- 理解「多轮梯度更新 + 重放」如何把 on-policy 方法推向 off-policy（L5 铺垫）。

## 先修知识

- L3 策略梯度；条件期望与递推；Bellman 期望方程。

## 核心概念

### 1. 三个基本对象

```text
V^π(s)  = E_{τ~π}[ Σ_{t=0}^∞ γ^t r_t | s_0 = s ]
Q^π(s,a)= E_{τ~π}[ Σ_{t=0}^∞ γ^t r_t | s_0 = s, a_0 = a ]
A^π(s,a)= Q^π(s,a) − V^π(s)          （注意 π 固定）
```

策略梯度里的「回报减 baseline」正好是 `A^π` 的估计：`R_t − V(s_t)`。

### 2. 估计 V：MC vs bootstrap

| 方法 | 目标 | 偏差 | 方差 | 数据 |
| --- | --- | --- | --- | --- |
| Monte Carlo | `G_t = Σ_{t'≥t} γ^{t'-t} r_{t'}` | 无偏 | 高（整条轨迹） | 需完整 episode |
| TD(0) | `r_t + γV_φ(s_{t+1})` | 有偏（依赖 φ） | 低 | 可在线、可复用 |
| n-step | `Σ_{k=0}^{n-1} γ^k r_{t+k} + γ^n V_φ(s_{t+n})` | 中等 | 中等 | 折中 |

TD 残差：

```text
δ_t = r_t + γ(1−d_t) V_φ(s_{t+1}) − V_φ(s_t)
L_V(φ) = E[ (V_φ(s_t) − target)² ],   target 视为常数（stop-gradient）
```

`d_t ∈ {0,1}` 是终止标志：终止时 `s_{t+1}` 不应 bootstrap。

### 3. GAE（Generalized Advantage Estimation）

对多步 TD 残差做指数加权：

```text
Â_t = δ_t + γλ(1−d_t) Â_{t+1},   Â_T = 0
```

- `λ=0`：`Â_t = δ_t`，TD 优势，低方差高偏差；
- `λ=1`：`Â_t = Σ_{k≥0} γ^k δ_{t+k} = R_t − V(s_t)`，MC 优势，高方差低偏差；
- 实践中 `λ ∈ [0.9, 0.99]`。

值函数回归目标：`R̂_t = Â_t + V_φ(s_t)`（stop-gradient）。

### 4. Batch actor-critic 算法

```text
for iteration:
    1. 用 π_θ 收集 N 条轨迹（或固定步数）
    2. 计算 V_φ(s_t)（no_grad）
    3. 计算 Â_t / R̂_t（GAE）
    4. 更新 critic：最小化 (V_φ(s_t) − R̂_t)²
    5. 更新 actor：最大化 Σ log π_θ(a_t|s_t) · Â_t   （或 PPO loss）
```

策略损失（最简单版）：

```text
L_π(θ) = −E[ Σ_t log π_θ(a_t|s_t) · Â_t ]
```

### 5. 折扣因子

`γ` 既是收敛因子也是「有效时域」：`H_eff ≈ 1/(1−γ)`。`γ=0.99 → 100` 步、
`γ=0.999 → 1000` 步。`γ` 越小越短视、方差越低；太大则方差高、值函数难拟合。

### 6. 向 off-policy 演进

- **Version 1：多轮梯度更新**。同一批数据上更新多次，数据比当前策略「旧」一点，
  用重要性比率修正（PPO，L5）；
- **Version 2：replay buffer**。保存历史数据、随机采样；需要修正价值估计与策略更新
  （SAC，L5）。

## 关键推导：GAE

由 `Â_t = δ_t + γλÂ_{t+1}` 展开：

```text
Â_t = δ_t + γλδ_{t+1} + (γλ)²δ_{t+2} + … + (γλ)^{T-t}δ_T
```

当 `λ=1` 且 `d=0` 时：

```text
Â_t = Σ_{k≥0} γ^k (r_{t+k} + γV(s_{t+k+1}) − V(s_{t+k}))
    = Σ_{k≥0} γ^k r_{t+k} − V(s_t)      （望远镜求和）
    = G_t − V(s_t)
```

这正是「MC 优势」。`λ` 在 TD 与 MC 之间做平滑插值。实现时**逆序**循环即可：

```python
adv = 0
for t in reversed(range(T)):
    delta = rew[t] + gamma * (1 - done[t]) * next_value[t] - value[t]
    adv = delta + gamma * lam * (1 - done[t]) * adv
    advantages[t] = adv
returns = advantages + value
```

## 深入推导

### 推导 1：n-step 回报的偏差-方差

`n` 步目标 `G_t^{(n)}=Σ_{k=0}^{n-1}γ^k r_{t+k}+γ^n V(s_{t+n})`。

- `n=1`：目标依赖 `V_φ`（偏差来自函数逼近），但只用到一步随机奖励（方差小）；
- `n→∞`：目标趋向 MC 回报（无偏），但包含全部随机奖励（方差大）；
- 实践中 `n` 的选择等价于 GAE 的 `λ`（两者是同一 bias-variance 曲线上的不同参数化）。

### 推导 2：GAE 的展开与 telescoping

递推 `Â_t=δ_t+γλÂ_{t+1}` 展开：

```text
Â_t = δ_t + γλδ_{t+1} + (γλ)²δ_{t+2} + … 
```

当 `λ=1`、无终止时：

```text
Â_t = Σ_{k≥0} γ^k [r_{t+k} + γV(s_{t+k+1}) − V(s_{t+k})]
    = Σ_{k≥0} γ^k r_{t+k} − V(s_t) + lim_{K→∞}γ^K V(s_{t+K})
    = R_t − V(s_t)                        （γ<1 时末项趋近 0）
```

即「MC 优势 = 回报 − baseline」，这正是 `λ` 在两端插值的原因。

### 推导 3：为什么 critic 目标必须 stop-gradient

若目标 `R̂_t=Â_t+V_φ(s_t)` 不 detach，则 `L_V` 的梯度里同时含「预测变小」和「目标变小」
两股力量，网络可能通过让 `V(s_t)` 下降来降低 loss，而不是更准确地预测回报。
`detach` 把目标固定成回归问题，类似监督学习。

### 推导 4：终止与截断

`done` 处不能用 `V(s_{T+1})`：episode 结束后未来回报为 0。若使用时间限制截断（time limit），
标准做法是同时传 `done` 与 `truncated`，只对真正终止停止 bootstrap，对截断保留 bootstrap。

## 实现映射（HW2 P2）

| 概念 | 文件 | TODO | 验收 |
| --- | --- | --- | --- |
| GAE | `assignments/spring2026/homework2-online-rl/starter-code/on_policy.py` | `compute_gae` | 与官方公式一致，`returns = adv + values` |
| PPO 更新 | 同文件 `update` | (a) `no_grad` 下算 GAE；(b) clipped surrogate | 训练后 `eval/episode_success ≥ 25%` |
| 截断正态策略 | 同文件（给定） | `TruncatedNormal` 采样与 `log_prob` | 动作在 `[−1,1]` |
| 参考策略 KL | 同文件（给定） | `π_ref` 反向 KL 正则 | 防止漂移过远 |

PPO-Clip：

```text
ρ_t = exp( log π_θ(a_t|s_t) − log π_old(a_t|s_t) )
L_CLIP = −E[ min( ρ_t Â_t, clip(ρ_t, 1−ε, 1+ε) Â_t ) ]
```

## 易错点与反例

- **忘记 stop-gradient**：advantage/return 必须视为常数，否则 critic 目标会「自己追自己」；
- **终止 bootstrap**：`done=True` 时不能加 `γV(s')`；
- **GAE 逆序边界**：`Â_T` 初始化为 0，用 `(1−done)` 截断；
- **比率用概率相除**：应使用 `log` 差再 `exp`，数值稳定；
- **同一批数据更新太多轮**：策略漂移，clip 失效；PPO 通常 3–10 epochs、minibatch 打乱；
- **advantage 未归一化**：不同任务尺度差异大，标准化能稳定训练。

## 实验或练习

1. 实现 `compute_gae` 后，用 `λ=0 / 0.95 / 1` 训练比较方差与最终成功率；
2. 关闭 `π_ref` 的 KL 正则，观察 PPO 是否崩掉；
3. 思考：为什么 actor-critic 是「有偏但低方差」的 policy gradient？偏差来自哪里？

## 高频问题

- 问：`V` 与 `Q` 的关系？答：`V(s)=E_{a~π}[Q(s,a)]`；`A=Q−V`。
- 问：GAE 的 `λ` 控制什么？答：TD 与 MC 的插值，偏差-方差折中。
- 问：critic 的回归目标为什么不反传到 `V(s_t)` 里？答：目标必须固定，否则是移动靶。
- 问：为什么要多轮更新同一批数据？答：样本效率；代价是分布偏移，用 clip/KL 限制。

## 面试追问链（由浅入深）

1. Actor-critic 相比 REINFORCE 改了什么？→ 用 `V_φ` 当 baseline/引导，换偏差降方差。
2. 为什么 critic 也要学而不是直接算？→ 状态空间大，只能函数逼近；这也是偏差来源。
3. `λ` 增大时曲线怎么变？→ 更接近 MC：偏差下降、方差上升，需要更多样本/更小步长。
4. 为什么优势要归一化？→ 不同任务回报尺度差异大；标准化让策略梯度步长可比。
5. 多轮更新同一批数据为什么需要 clip？→ 数据来自 `π_old`，比率偏离 1 时估计失效。
6. 值函数不正交会怎样？→ 优势估计有偏 → 策略被错误方向拉动；这是 PPO 常见失败模式。

## 自测题（附要点）

1. 分别写出 `V/Q/A` 的定义并证明 `V(s)=E_{a~π}[Q(s,a)]`。
   - 要点：对首步动作取期望即得。
2. 推导 `λ=1` 时 GAE 等于 MC 优势。
   - 要点：望远镜求和，末项在 γ<1 下消失。
3. 给定 `[r_0,r_1,r_2]`、`V(s)`、`γ,λ`，手算 `δ_t` 与 `Â_t`（逆序）。
   - 要点：从 T−1 往 0，`Â=δ+γλ(1−d)Â_next`。
4. 说明 `returns=advantages+values` 在代码中的作用。
   - 要点：一次算出 critic 回归目标与 actor 权重。

## 关联笔记

- 前置：[L3 策略梯度](lecture-03-policy-gradients.md)
- 后续：[L5 PPO/SAC](lecture-05-off-policy-actor-critic.md)、
  [L6 Q-learning](lecture-06-q-learning.md)

## 本讲小结

Actor-critic = policy gradient + 学出来的 baseline。GAE 把「用几步 bootstrap」变成可调旋钮，
PPO 把「数据能用几次」变成可调旋钮。下一讲深入 off-policy：replay buffer、SAC 与 PPO 的工程细节。

## 延伸阅读

- Sutton et al., *Policy Gradient Methods for RL with Function Approximation*, 1999.
- Schulman et al., *High-Dimensional Continuous Control Using GAE*, 2016.
- Schulman et al., *Proximal Policy Optimization Algorithms*, 2017.
