# Lecture 03 — Policy Gradients

- 课程版本：Spring 2026（2025 有录像）
- 日期：2026-04-08
- 官方材料：[03_cs224r_policy_gradients_2026.pdf](../slides/spring2026/03_cs224r_policy_gradients_2026.pdf) ·
  [2025 版](../slides/spring2025/03_cs224r_policy_gradients_2025.pdf) ·
  [Lecture 3 视频](https://www.youtube.com/watch?v=KCAOXd4IO9o) ·
  经典论文：Williams (1992) REINFORCE
- 状态：初稿

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

## 本讲小结

策略梯度把 RL 变成「带权重的极大似然」，只需要可微策略与回报样本。它无偏但高方差，
L4 将引入值函数作为 baseline（actor-critic）与 GAE 来系统性地降低方差。

## 延伸阅读

- Williams, *Simple Statistical Gradient-Following Algorithms for Connectionist RL*, 1992.
- Sutton et al., *Policy Gradient Methods for RL with Function Approximation*, 1999.
- Schulman et al., *High-Dimensional Continuous Control Using GAE*, 2016（L4 预备）。
