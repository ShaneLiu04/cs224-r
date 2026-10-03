# Lecture 05 — Off-Policy Actor-Critic Methods

- 课程版本：Spring 2026（2025 有录像）
- 日期：2026-04-15
- 官方材料：[05_cs224r_offpolicy_actor_critic_2026.pdf](../slides/spring2026/05_cs224r_offpolicy_actor_critic_2026.pdf) ·
  [2025 版](../slides/spring2025/05_cs224r_offpolicy_actor_critic_2025.pdf) ·
  [Lecture 5 视频](https://www.youtube.com/watch?v=cRGKc-nAWho) ·
  作业：HW2 P2/P3
- 状态：初稿

## 学习目标

- 说出 PPO 的三个工程要点（clip、多轮复用、GAE/KL），并能写伪代码；
- 理解 replay buffer 带来的 off-policy 问题：价值估计、策略更新、探索；
- 写出 SAC 的最大熵目标、soft Q 更新与重参数化策略梯度；
- 比较 PPO 与 SAC：稳定性、样本效率、适用场景；
- 说明 PPO 为什么成为 LLM RLHF 的默认算法。

## 先修知识

- L3–L4：策略梯度、优势、GAE、actor-critic；
- 重要性采样；高斯策略与重参数化技巧。

## 核心概念

### 1. PPO：少样本多次更新的 actor-critic

同一批 rollout 上做多轮 minibatch 更新，用概率比限制更新幅度：

```text
ρ_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)
L_CLIP = −E_t[ min( ρ_t Â_t, clip(ρ_t, 1−ε, 1+ε) Â_t ) ]
```

为什么 `min`：`Â>0` 时限制 `ρ` 不超过 `1+ε`；`Â<0` 时限制不低于 `1−ε`。
好处：只有「变好」的方向被保留，避免为了增加目标而极端改变策略。

PPO 的完整组成：

1. 并行环境收集 rollout（`γ`、`λ`、`T` 步）；
2. 用 GAE 算优势，critic 回归 `R̂_t`；
3. 多 epoch minibatch 更新 actor/critic；
4. 可选：value clipping、advantage 归一化、熵正则、学习率调度；
5. 在 LLM 场景可加对参考策略的（反向）KL 惩罚。

### 2. Replay buffer 与 off-policy 的三个修正

Replay buffer 缓存 `(s, a, r, s', d)`，采样时数据可能来自任意旧策略。三处不匹配：

1. **价值估计**：`E_{a'~π}[Q(s',a')]` 不能用旧数据动作近似 → 用目标网络 + bootstrap；
2. **策略更新**：数据分布 `β ≠ π_θ` → 用重要性权重或确定性策略梯度（DPG）；
3. **探索**：策略可能过于确定 → 最大熵或显式噪声。

### 3. SAC：最大熵 actor-critic

目标在回报上加策略熵：

```text
J(π) = Σ_t E[ r(s_t, a_t) + α H(π(·|s_t)) ]
```

直觉：在能拿同样回报的策略中，优先选「尽量随机」的，以保持探索与鲁棒性；
`α` 可以自动调节（目标熵约束）。

三件套：

- **Soft Q**：`Q(s,a)` 的值包含未来熵项，目标用 clipped double-Q：
  `y = r + γ( min_i Q̄_i(s', a') − α log π(a'|s') )`；
- **重参数化策略**：`a = tanh(μ_θ(s) + σ_θ(s) ⊙ ξ), ξ~N(0,I)`，可直接对 Q 反传；
- **目标网络 + 慢更新**：`φ̄ ← τφ + (1−τ)φ̄`。

策略损失：

```text
L_π = E_{s~D, ξ}[ α log π_θ(a_θ(s,ξ)|s) − min_i Q_i(s, a_θ(s,ξ)) ]
```

Q 损失：

```text
L_Q = E_{(s,a,r,s')~D}[ (Q_i(s,a) − y)² ],  y = r + γ( min_j Q̄_j(s',a') − α log π(a'|s') )
```

### 4. PPO vs SAC

| 维度 | PPO | SAC |
| --- | --- | --- |
| 数据 | 新采的 rollout，多次更新后丢弃 | replay buffer，持续复用 |
| 样本效率 | 中低 | 高 |
| 稳定性 | 高（clip + 小步长） | 中（对 Q 的偏差敏感） |
| 并行 | 适合大规模并行仿真 | 单机多环境即可 |
| 代表应用 | 人形机器人、RLHF | 精密操作、快速真机学习 |

经验法则（Chelsea 的建议）：需要稳定、能大规模并行仿真、奖励可塑时用 PPO；
真机、样本贵、需要更高数据效率时用 SAC 系。

### 5. PPO 与语言模型

RLHF 中：策略 = 语言模型，动作 = 生成 token，奖励 = reward model 打分 − KL 惩罚。
PPO 的 clip 与「同一批 prompt 多轮更新」恰好契合现代训练管线（vLLM 采样 + 训练分离）。
缺点是 critic 与 policy 同样大、显存开销高——GRPO/RLOO 用组内相对奖励绕开 critic（L10）。

## 关键推导：SAC 策略梯度的重参数化

对 `L_π = E[α log π(a|s) − Q(s,a)]`，直接用 `a = tanh(μ + σξ)` 对 `θ` 求导：

```text
∇_θ L_π = E_ξ[ ∇_a(α log π(a|s) − Q(s,a)) · ∇_θ a_θ(s,ξ) ]
```

（实现中 `log π` 的 `tanh` 雅可比项由分布类的 `rsample` 处理。）
相比 likelihood-ratio 版本，重参数化梯度方差更低，是连续控制的标准做法。

### 6. 其他算法一瞥

- **DDPG**：确定性策略 + Q 梯度，易过估计、脆；
- **TD3**：clipped double-Q + 目标动作平滑 + 延迟策略更新；
- **Distributional RL**（C51、QR-DQN）：建模回报分布而非期望，改善稳定性。

## 实现映射

| 概念 | HW2 文件 | 说明 |
| --- | --- | --- |
| PPO clip / GAE | `homework2-online-rl/starter-code/on_policy.py` | P2：成功率 ≥25% |
| Off-policy AC | `homework2-online-rl/starter-code/off_policy.py` | P3：critic ensemble + BC 预训练 + UTD |
| Bellman 目标 | `update_critic` | `y = r + γ min(Q̄_i(s',a'), Q̄_j(s',a'))`，min 取自两个**随机采样**的 target critic |
| 策略更新 | `update_actor` | `L_π = −(1/N) Σ_i Q_{φ_i}(s, a')`，`a'~π_θ` |
| 目标网络 | `utils.py::soft_update_params` | `Q̄ = (1−ρ)Q̄ + ρQ` |

注意 HW2 的 off-policy AC 与真正 SAC 的差别：没有熵项、目标动作从当前策略采样而非 target
策略、BC 正则用于稳定。它是 PPO 之外的「重放 + 多步价值」路线，便于对比。

## 易错点与反例

- **`ρ` 算错基准**：必须除以采样时的 `π_old`，而不是上一 minibatch 的策略；每个 rollout
  收集后先算 `old_log_prob` 并冻结；
- **clip 后忘记取 `min`**：只 clip 不取 min 会改变目标方向；
- **target network 忘了 no_grad**：Bellman target 必须 detach；
- **double-Q 的两种采样**：min 的对象是目标网络而非在线网络；
- **SAC 的 `log π` 数值问题**：tanh 变换要对数密度修正；
- **UTD 倍率**：critic 更新次数增加要配 target 更新频率与正则，否则过估计爆掉（HW2 的
  `num_critics=10, utd=5` 正是这个消融）。

## 实验或练习

1. HW2 P3 对比 `2 critics / UTD=1` 与 `10 critics / UTD=5` 的成功率曲线，解释 UTD 的影响；
2. 把 PPO 的 `clip_eps` 从 0.2 改到 0.5/0.05，观察稳定性；
3. 思考：为什么 PPO 用 GAE 而 SAC 通常用单步 TD？各自的偏差-方差权衡是什么？

## 高频问题

- 问：PPO 与 TRPO 的关系？答：TRPO 解 KL 约束的优化问题，PPO 用 clip 近似，工程更简单。
- 问：PPO 中 `ρ_t` 与 off-policy 的关系？答：多轮复用时数据来自 `π_old`，比率修正分布。
- 问：SAC 的熵项为什么有助于探索？答：最大化熵 ⇒ 在低价值区域也保持多样性。
- 问：为什么 `min` 两个 Q 能缓解过估计？答：`min` 有向下偏差，抵消 `max` 的向上偏差。
- 问：RLHF 为什么偏爱 PPO 而不是 SAC？答：生成数据昂贵、on-policy 稳定、并行采样管线成熟。

## 本讲小结

Off-policy actor-critic 用重放与多次更新换样本效率，代价是分布偏移与过估计。PPO 选择保守
（clip），SAC 选择熵正则 + 双 Q。下一讲走另一条路：完全去掉 actor，只学 Q。

## 延伸阅读

- Schulman et al., *PPO*, 2017；Haarnoja et al., *Soft Actor-Critic*, 2018.
- Fujimoto et al., *TD3*, 2018；Lillicrap et al., *DDPG*, 2015.
- 官方 slides 与 HW2 handout：`assignments/spring2026/homework2-online-rl/`。
