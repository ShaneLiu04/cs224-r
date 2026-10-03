# Lecture 03 — Policy Gradients

- 课程版本：Spring 2026（2025 有录像）
- 日期：2026-04-08
- 官方材料：[03_cs224r_policy_gradients_2026.pdf](../slides/spring2026/03_cs224r_policy_gradients_2026.pdf) ·
  [2025 版](../slides/spring2025/03_cs224r_policy_gradients_2025.pdf) ·
  [Lecture 3 视频](https://www.youtube.com/watch?v=KCAOXd4IO9o) ·
  经典论文：Williams (1992) REINFORCE
- 状态：初稿

## 摘要

本讲从 `p_θ(τ)` 出发完整推导策略梯度：利用 `∇p=p∇log p` 与「环境项与 θ 无关」两个事实，
把不可直接求导的期望回报变成可采样估计 `E[Σ∇logπ(a_t|s_t)·R(τ)]`。随后用 causality 去掉
未来无关奖励（reward-to-go）、用 baseline 无偏地降低方差，最后给出重要性采样形式的
off-policy 版本。策略梯度把 RL 变成「以回报为权重的最优似然」，是所有 actor-critic 方法的原点。

## 一页速览卡（面试前 5 分钟）

**必须记住的五件事**

1. 推导三步：`∇∫pR = ∫p∇log p·R = E[∇log p(τ)·R(τ)]`。
2. `∇log p(τ)=Σ_t∇logπ_θ(a_t|s_t)`：初始分布与转移概率都不含 θ。
3. Causality：`t` 之前的奖励与 `a_t` 独立，只乘 reward-to-go `R_t`，无偏且降方差。
4. Baseline：任意 `b(s_t)` 满足 `E_{a~π}[∇logπ·b(s)]=0`，减去它不改变期望。
5. Off-policy：用 `π_θ/π_old` 修正动作分布，实践中只做单步比率（PPO 的起点）。

**必须能默写的公式**

```text
∇J(θ) = E_{τ~p_θ}[ Σ_t ∇log π_θ(a_t|s_t) · R(τ) ]
      = E_{τ~p_θ}[ Σ_t ∇log π_θ(a_t|s_t) · (R_t − b(s_t)) ]
R_t   = Σ_{t'=t}^{T} γ^{t'-t} r_{t'}
E_{a~π_θ}[ ∇log π_θ(a|s) · b(s) ] = b(s) ∇Σ_a π_θ(a|s) = 0
```

**一句话区分**

- PG vs BC：样本按回报加权 vs 权重恒为 1。
- `R(τ)` vs `R_t`：整条回报 vs 从 t 起的折扣回报；期望相同、方差后者更小。
- On-policy vs off-policy PG：数据必须由当前策略采集 vs 用重要性比率复用旧数据。

## 学习目标

- 从 `p_θ(τ)` 出发完整推导 `∇J(θ)`，解释环境项为何消失；
- 用 causality 去掉未来无关奖励，写出 reward-to-go 形式；
- 说明 baseline 的无偏性，理解方差来源与常见缓解手段；
- 写出可实现的 REINFORCE 伪代码与 PyTorch 损失；
- 推导重要性采样形式的 off-policy policy gradient，理解其适用边界。

## 先修知识

- L1–L2；对数求导技巧 `∇p = p∇log p`；重要性采样恒等式。

## 核心概念

### 1. 在线 RL 循环

```text
初始化策略 π_θ（随机 / 模仿学习 / 启发式）
repeat:
    1. 用 π_θ 与环境交互，收集轨迹
    2. 估计 ∇J(θ)
    3. 梯度上升更新 θ
```

在线 RL 与模仿学习的区别：没有专家标签，只有奖励；数据来自策略自身。

### 2. 目标与轨迹概率

```text
J(θ) = E_{τ ~ p_θ(τ)} [ R(τ) ],   R(τ) = Σ_t r(s_t, a_t)

p_θ(τ) = ρ0(s_1) Π_t π_θ(a_t | s_t) P(s_{t+1} | s_t, a_t)
```

### 3. 策略梯度定理（推导主线）

对期望求导：

```text
∇J(θ) = ∇ ∫ p_θ(τ) R(τ) dτ
      = ∫ ∇p_θ(τ) R(τ) dτ
      = ∫ p_θ(τ) ∇log p_θ(τ) R(τ) dτ
      = E_{τ~p_θ}[ ∇log p_θ(τ) R(τ) ]
```

而

```text
∇log p_θ(τ) = ∇log ρ0(s_1) + Σ_t ∇log π_θ(a_t|s_t)
              + Σ_t ∇log P(s_{t+1}|s_t,a_t)
            = Σ_t ∇log π_θ(a_t|s_t)          （前两项与 θ 无关）
```

所以

```text
∇J(θ) = E_{τ~p_θ}[ (Σ_t ∇log π_θ(a_t|s_t)) · R(τ) ]
```

「environment 项消失」是所有 policy gradient 方法的支点：不需要知道动力学就可以无偏估计梯度。

### 4. Causality / reward-to-go

`t` 时刻的动作只能影响 `t` 及之后的奖励，之前的奖励与 `a_t` 条件独立：

```text
E_{a_t~π} [ ∇log π_θ(a_t|s_t) · r_{t'} ] = 0,   当 t' < t
```

因此可以只乘 **reward-to-go**：

```text
∇J(θ) = E[ Σ_t ∇log π_θ(a_t|s_t) · Σ_{t'=t}^{T} r_{t'} ]
```

实现上就是用 `R_t`（从 t 开始的折扣回报）替换整条轨迹回报 `R(τ)`，方差显著下降。

### 5. Baseline

对任意只依赖状态的函数 `b(s_t)`：

```text
E_{a_t~π_θ} [ ∇log π_θ(a_t|s_t) · b(s_t) ]
 = b(s_t) ∇_θ Σ_a π_θ(a|s_t) = b(s_t) ∇_θ 1 = 0
```

所以减去 baseline 不改变期望（无偏），但能降低方差。当 `b(s_t) ≈ V^π(s_t)` 时，
`R_t − b` 近似优势 `A^π(s_t, a_t)`。常用 `b`：回报均值、训练好的 `V_φ(s)`。

### 6. 梯度的直觉

`∇J` 是「用回报加权的最优似然」：

- `R(τ) > 0`：提高这条轨迹中所有动作的对数概率；
- `R(τ) < 0`：降低；
- 权重偏离基线时方向相反，因此 baseline 只改方向分配不改期望。

这与模仿学习的 MLE 只差一个权重：BC 把所有权重设为 1。

### 7. 方差来源与缓解

1. **轨迹随机性**：单条轨迹回报波动大 → 多轨迹平均、并行环境；
2. **长时域**：`R_t` 方差随 `T` 增长 → reward-to-go、`γ<1`、baseline；
3. **动作分布**：`∇log π` 尺度随动作维/参数化变化 → 自适应学习率、advantage 归一化；
4. **信用分配**：所有动作共享同一回报 → `γ`、GAE（L4）、n-step。

### 8. Off-policy 版本与重要性采样

若数据来自旧策略 `π_old`（如行为策略 `β`）：

```text
J(θ) = E_{τ~p_old}[ (Π_t π_θ(a_t|s_t) / π_old(a_t|s_t)) R(τ) ]
```

逐动作比率的乘积会随 `T` 爆炸，常见做法是只对单步动作做重要性采样：

```text
∇J(θ) ≈ E_{s~d^{π_old}, a~π_old}[ (π_θ(a|s) / π_old(a|s)) ∇log π_θ(a|s) · A^{π_old}(s,a) ]
```

这是 v-trace、PPO 等方法的起点。注意两点偏差来源：

- 只对动作比率采样 → 忽略状态分布差异（近似）；
- 只做一步 → 忽略后续轨迹分布（近似）。

## 关键推导：单步重要性权重下的 off-policy 目标

从 on-policy 目标 `J = E_{s~d^π, a~π}[Q^π(s,a)]` 出发，用行为策略 `β` 的采样改写：

```text
J = E_{s~d^π, a~β}[ (π(a|s)/β(a|s)) Q^π(s,a) ]
```

对 `θ` 求导并用 `∇Q^π` 可通过 `Q^π` 当常数近似（确定性等价），得到
`E_{a~β}[ (π/β) ∇log π · Q ]`，再减去 baseline 换成 advantage。实际算法（PPO）会用
旧策略比 `ρ = π_θ/π_old` 截断以控制方差。

## 深入推导

### 推导 1：从期望到可采样估计（逐步）

```text
∇J(θ) = ∇ ∫ p_θ(τ) R(τ) dτ
      = ∫ ∇p_θ(τ) R(τ) dτ                       （交换积分与梯度，R 不含 θ）
      = ∫ p_θ(τ) · [∇log p_θ(τ)] · R(τ) dτ      （∇p = p∇log p）
      = E_{τ~p_θ}[ ∇log p_θ(τ) R(τ) ]            （写成期望 ⇒ 可 Monte Carlo）
```

再展开 `log p_θ(τ)`：

```text
log p_θ(τ) = log ρ0(s1) + Σ_t log π_θ(a_t|s_t) + Σ_t log P(s_{t+1}|s_t,a_t)
```

`ρ0`、`P` 与 θ 无关，逐项求导后只剩 `Σ_t∇logπ_θ(a_t|s_t)`。

### 推导 2：Causality（条件期望证明）

对 `t' < t`，`r_{t'}` 在给定 `τ_{1:t}` 下与 `a_t` 无关，因此

```text
E_{a_t~π}[ ∇log π(a_t|s_t) · r_{t'} | s_t ] = r_{t'} · E[∇log π] = r_{t'} · ∇Σ_a π = 0
```

把整条回报换成 `R_t` 后，被去掉项的期望为 0，所以估计无偏。

### 推导 3：Variance reduction 的直观

设 `g_t = ∇logπ(a_t|s_t)`。`Var(Σ_t g_t·R) = Σ_t Var(g_t R) + 2Σ_{i<j}Cov(·)`。
用 `R_t` 后每个 `g_t` 只乘与自身相关的回报；减去 `b(s_t)` 再消掉 `g_t` 的均值分量。
最优 baseline 近似为 `V^π(s_t)`（这正是 actor-critic 的动机，L4）。

### 推导 4：Off-policy 单步比率的两个近似

从 on-policy 目标 `E_{s~d^π,a~π}[Q^π]` 改写为行为策略采样：

```text
E_{s~d^π, a~β}[ (π_θ(a|s)/β(a|s)) Q^π(s,a) ]
```

对 θ 求导时：(1) 把 `d^π` 换成 `d^{π_old}`（忽略状态分布随参数的漂移）；
(2) 把 `Q^π` 当常数、只对显式的 `π_θ/β` 与 `∇logπ_θ` 求导。两者都是近似，
PPO 用 clip 控制由此产生的偏差（L5）。

## 实现映射

| 概念 | 对应代码 | 说明 |
| --- | --- | --- |
| 轨迹收集 | HW2 `starter-code/train_on_policy.py`（PPO 的 rollout 部分） | 记录 `obs, act, rew, done` |
| reward-to-go | 自己实现练习 | 逆序累加 `R_t = r_t + γR_{t+1}` |
| REINFORCE loss | 自己实现练习 | `-(logp * R_t).mean()` |
| GAE 替代 | HW2 `on_policy.py::compute_gae` | L4 主题 |

小练习（不提交）：在 Flappy Bird 上用 REINFORCE + reward-to-go 训一个策略，
观察 baseline 对学习曲线方差的影响，作为 HW1 之后的热身。

## 易错点与反例

- **对 `p_θ(τ)` 求导时忘记 `∇log` 换元**：直接对 `R(τ)` 求导会得到 0（R 不含 θ）；
- **把 `R(τ)` 当成可微目标**：期望的梯度估计是 `∇logπ · R`，不是对 R 反传；
- **reward-to-go 方向写反**：`R_t` 是从 `t` 往后的折扣和，逆序计算；
- **baseline 依赖动作**：`b(s, a)` 一般会有偏，只有 `b(s)` 才保证无偏；
- **多个 episode 丢了 `done` 边界**：跨 episode 累加回报是经典 bug；
- **off-policy 比率连乘**：`Π_t π_θ/π_old` 方差爆炸，实践用 PPO 的截断或每步比率。

## 实验或练习

1. 固定网络与随机种子，只改「乘 `R(τ)` / 乘 `R_t` / 减 baseline」三档，比较梯度方差与收敛速度；
2. 用有限差分验证：对一维小参数策略，数值梯度 vs `E[∇logπ·R]` 估计，检验无偏性；
3. 思考题：为什么 `R_t` 的方差比 `R(τ)` 小，但期望不变？用条件期望证明。

## 高频问题

- 问：推导 policy gradient 的关键一步？答：`∇p = p∇log p`，再用 `∇log p(τ) = Σ∇logπ`。
- 问：baseline 为什么无偏？答：`E_a[∇logπ(a|s)] = ∇Σπ = 0`。
- 问：reward-to-go 为什么成立？答：过去奖励与当前动作独立，条件期望为零。
- 问：on-policy 的样本效率瓶颈？答：更新一次后数据分布失效，必须重新采样。
- 问：off-policy PG 的两个近似？答：状态分布不匹配、单步比率忽略后续轨迹。

## 面试追问链（由浅入深）

1. 为什么不直接对 `J(θ)` 反传？→ 期望对分布求导，采样不可微；`∇log` 把梯度变成
   「回报加权对数概率」的期望，可无偏估计。
2. 为什么环境项消失？→ `ρ0` 与 `P(s'|s,a)` 不依赖策略参数；这是 model-free 的数学基础。
3. 只用 `R_t` 会不会有偏？→ 不会，被丢弃项对 `a_t` 的条件期望为 0。
4. Baseline 会不会改变最优解？→ 不改变期望梯度，只重新分配方向；`b≈V` 时方差最小。
5. 为什么 PG 样本效率低？→ 更新一次后 `π_old` 失效，必须重新采样；纠正是 off-policy 的比率。
6. 方差还能怎么降？→ 多轨迹/并行、advantage 归一化、`γ`、GAE、控制动作分布尺度（L4）。

## 自测题（附要点）

1. 完整写出 `∇J` 推导的四行，并标注每步用到的恒等式。
   - 要点：积分交换、`∇p=p∇log p`、期望定义、`log` 展开。
2. 证明 baseline 无偏，并给出最优 baseline 的近似形式。
   - 要点：`E[∇logπ]=0`；最优 `b(s)≈V^π(s)`。
3. 写出 off-policy 单步比率估计，并指出两个偏差来源。
   - 要点：状态分布不匹配、把 Q 视为常数。
4. 数值直觉：动作维度从 1 增到 100，`∇logπ` 的方差大致如何变化？为什么？
   - 要点：随维度增大；似然项随维度累加，需要 advantage 归一化/自适应步长。

## 关联笔记

- 前置：[L2 模仿学习](lecture-02-imitation-learning.md)
- 后续：[L4 Actor-Critic 与 GAE](lecture-04-actor-critic.md)、
  [L5 PPO 与 off-policy 修正](lecture-05-off-policy-actor-critic.md)

## 本讲小结

策略梯度把 RL 变成「带权重的极大似然」，只需要可微策略与回报样本。它无偏但高方差，
L4 将引入值函数作为 baseline（actor-critic）与 GAE 来系统性地降低方差。

## 延伸阅读

- Williams, *Simple Statistical Gradient-Following Algorithms for Connectionist RL*, 1992.
- Sutton et al., *Policy Gradient Methods for RL with Function Approximation*, 1999.
- Schulman et al., *High-Dimensional Continuous Control Using GAE*, 2016（L4 预备）。
