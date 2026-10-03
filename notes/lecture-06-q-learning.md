# Lecture 06 — Q-Learning

- 课程版本：Spring 2026（2025 有录像）
- 日期：2026-04-17
- 官方材料：[06_cs224r_qlearning_2026.pdf](../slides/spring2026/06_cs224r_qlearning_2026.pdf) ·
  [2025 版](../slides/spring2025/06_cs224r_qlearning_2025.pdf) ·
  [Lecture 6 视频](https://www.youtube.com/watch?v=-7kv6jf0isQ) ·
  [Q-learning section](../slides/spring2025/section_q_learning_tutorial.pdf) ·
  作业：HW2 P1（gridworld）
- 状态：初稿

## 摘要

Q-learning 完全去掉 actor：学习最优动作价值 `Q*`，策略隐式地是每步 `argmax`。Bellman 最优
算子是 `γ`-压缩映射，保证表格情形下收敛。深度版本需要 replay buffer 与 target network 两个
工程支柱；`max` 带来的过估计偏差用 Double Q-learning 修正。本讲最后汇总在线 RL 算法地图，
给出「选哪个算法」的判断框架。

## 一页速览卡（面试前 5 分钟）

**必须记住的五件事**

1. `Q*(s,a)=r+γE_{s'}[max_{a'}Q*(s',a')]`；策略 `π*(s)=argmax_a Q*(s,a)`。
2. 表格更新 `Q←Q+α[r+γmaxQ'−Q]`；深度版用 target network 计算目标、replay 采样。
3. `E[max Q̂] ≥ max E[Q̂]`：过估计随动作数与噪声增大，bootstrap 沿轨迹累积。
4. Double Q：一个网络选动作、另一个评估；`y=r+γQ̄(s',argmaxQ_φ(s',a'))`。
5. 收敛条件：所有状态-动作无限次访问 + 步长 `Σα=∞, Σα²<∞`（表格）/ 函数逼近需额外条件。

**必须能默写的公式**

```text
Q(s,a) ← Q(s,a) + α[ r + γ max_{a'} Q(s',a') − Q(s,a) ]
L_Q(φ) = E_{(s,a,r,s')~D}[ ( Q_φ(s,a) − (r + γ max_{a'} Q̄(s',a'))_sg )² ]
Double: y = r + γ Q̄(s', argmax_{a'} Q_φ(s',a'))
n-step: y^{(n)} = Σ_{k=0}^{n-1} γ^k r_{t+k} + γ^n max_{a'} Q̄(s_{t+n}, a')
```

**一句话区分**

- Q-learning vs SARSA：目标用 `max`（off-policy）vs 实际下一动作（on-policy）。
- Tabular vs DQN：精确表 vs 函数逼近 + replay + target net。
- 过估计 vs 高方差：系统性向上偏差 vs 采样波动；前者用 Double Q，后者用平均/减小学习率。

## 学习目标

- 写出 `Q*` 的 Bellman 最优方程，并从 actor-critic 推导出 Q-learning；
- 解释 Q-learning 为什么 off-policy、隐式的策略是什么；
- 说明 target network 与 replay buffer 分别解决什么问题；
- 手推 `max` 导致过估计的原因，写出 Double Q-learning 的目标；
- 完成 HW2 P1 的三个奖励场景并解释轨迹选择。

## 先修知识

- L4–L5 的 `Q^π`、Bellman 方程、bootstrap；`ε`-greedy 探索。

## 核心概念

### 1. 从 actor-critic 到 critic-only

`Q^π(s,a)` 的定义是对「之后继续按 π 行动」的期望回报。若把策略也一起优化，
最优 Q 函数满足 Bellman 最优方程：

```text
Q*(s,a) = r(s,a) + γ E_{s'}[ max_{a'} Q*(s',a') ]
```

由此得到**最优策略**（无需单独的 actor）：

```text
π*(s) = argmax_a Q*(s,a)
```

### 2. Q-learning 更新

表格版（HW2 P1）：

```text
Q(s_t,a_t) ← Q(s_t,a_t) + α [ r_t + γ max_{a'} Q(s_t+1, a') − Q(s_t,a_t) ]
```

深度版：用网络 `Q_φ` 回归目标

```text
L_Q(φ) = E_{(s,a,r,s')~D}[ ( Q_φ(s,a) − [ r + γ max_{a'} Q̄_φ(s',a') ] )² ]
```

- 目标里的 `max` 用 **target network** `Q̄` 计算；
- 数据从 **replay buffer** `D` 采样；
- 行为策略只需覆盖可能的动作，更新目标是纯 off-policy 的。

### 3. 数据收集与探索

Q-learning 对行为策略没有要求，但**探索**仍关键：

- `ε`-greedy：以 `ε` 概率均匀随机动作，否则 `argmax Q`；
- Boltzmann/softmax：按 `exp(Q/τ)` 采样，温度 `τ` 控制随机性；
- 乐观初始化、计数奖励等（L14）。

在 HW2 的 gridworld 中，`ε` 从 1 逐步衰减到小值，保证覆盖后收敛到贪心。

### 4. 训练稳定性组件

| 组件 | 作用 |
| --- | --- |
| Target network | 固定回归目标，避免「追自己」的移动靶 |
| Replay buffer | 打破样本相关性，重复利用数据 |
| 硬更新 vs Polyak | 每 C 步复制 vs 每步 `φ̄←τφ+(1−τ)φ̄` |
| Huber/MSE | 控制异常 TD 误差的梯度 |
| Double Q | 缓解过估计 |

### 5. Overestimation bias

`max_a Q̂(s',a')` 对带噪估计取最大会产生向上偏差：

```text
E[ max_a Q̂(s',a') ] ≥ max_a E[ Q̂(s',a') ]
```

偏差随动作数与噪声方差增大。Bootstrapping 会把偏差沿轨迹累积，导致 Q 值虚高、
策略被错误动作吸引。

### 6. Double Q-learning

用两个网络解耦「选动作」与「评估动作」：

```text
y = r + γ Q_{φ2}(s', argmax_{a'} Q_{φ1}(s',a'))
```

理论目标是无偏的（交换两个网络角色可再得一组样本）。实践：Double DQN 用在线网络选、
目标网络评估：

```text
y = r + γ Q̄(s', argmax_{a'} Q_φ(s',a'))
```

### 7. N-step 回报

把 1 步 bootstrap 换成 n 步：

```text
y_t^{(n)} = Σ_{k=0}^{n-1} γ^k r_{t+k} + γ^n max_{a'} Q̄(s_{t+n}, a')
```

偏差更小、方差更大；DQN 系常用 `n=3~5`，或在 replay 中存 n-step 转移。

### 8. 实践案例

- DQN 在 Atari 上超过人类平均水平（Mnih et al., 2015）；Double DQN、Dueling、PER、
  Rainbow 逐步改进；
- QT-Opt 把 Q-learning 用到真机抓取（Kalashnikov et al., 2018）；
- Q-learning 与 BC 混合（BCQ、CQL）是离线 RL 的基础（L7）。

### 9. 在线 RL 算法总览

| 算法 | 策略更新 | 数据 | 偏差/方差 | 典型用途 |
| --- | --- | --- | --- | --- |
| Vanilla PG | 似然比 × 回报 | on-policy | 无偏低方差高 | 教学、简单任务 |
| Actor-critic (PPO) | 似然比 × 优势（clip） | 近 on-policy | 有偏低方差 | 机器人、RLHF |
| SAC | 重参数化 × Q + 熵 | off-policy | 有偏低方差 | 样本敏感任务 |
| Q-learning | 隐式 `argmax` | off-policy | bootstrap 偏差 | 离散动作、Atari |

**选择建议**：动作连续且需要稳定并行 → PPO；真机/样本贵 → SAC；动作离散、数据可重放 →
Q-learning/DQN；需要在线微调且对奖励敏感 → 先跑 PPO 基线。

## 关键推导：Q-learning 为何收敛到 `Q*`

把更新写成算子 `T`：

```text
(TQ)(s,a) = r(s,a) + γ E_{s'}[ max_{a'} Q(s',a') ]
```

`T` 是 `γ`-压缩（`||TQ1 − TQ2||_∞ ≤ γ||Q1 − Q2||_∞`），由 Banach 不动点定理，
迭代 `Q_{k+1}=TQ_k` 收敛到唯一不动点 `Q*`。随机采样版本在步长满足
`Σα_t = ∞, Σα_t² < ∞` 且所有 `(s,a)` 无限次访问时几乎必然收敛。

## 深入推导

### 推导 1：Bellman 最优算子是 γ-压缩

定义算子 `(TQ)(s,a)=r(s,a)+γE_{s'}[max_{a'}Q(s',a')]`。对任意 `Q1,Q2`：

```text
|TQ1(s,a) − TQ2(s,a)| ≤ γ E_{s'}[ |max_{a'}Q1(s',a') − max_{a'}Q2(s',a')| ]
                      ≤ γ E_{s'}[ max_{a'} |Q1(s',a') − Q2(s',a')| ]
                      ≤ γ ||Q1 − Q2||_∞
```

（第二行用 `|max f − max g| ≤ max|f−g|`。）取上确界得 `||TQ1−TQ2||_∞ ≤ γ||Q1−Q2||_∞`。
由 Banach 不动点定理，`T` 有唯一不动点 `Q*`，迭代收敛；误差每步至少乘 `γ`。

### 推导 2：过估计偏差的来源

设 `Q̂(s',a')=Q*(s',a')+ε_a`，`ε_a` 零均值（如独立噪声，方差 `σ²`）。则

```text
E[ max_a Q̂(s',a') ] ≥ max_a E[ Q̂(s',a') ] = max_a Q*(s',a')
```

不等式来自 `max` 的凸性（Jensen）。当 `A` 个动作、噪声高斯的经典近似为
`E[max] ≈ μ + σ√(2 ln A)`，动作越多、噪声越大，偏差越大。Bootstrapping 把该偏差反馈进
目标，随时间放大。

### 推导 3：Double Q 为什么无偏（理想情形）

若两个估计的噪声独立：`Q_A=Q*+ε_A`，`Q_B=Q*+ε_B`，且 `argmax` 在真值处唯一，
则选出的动作与 `ε_B` 独立，于是

```text
E[ Q_B(s', argmax Q_A) ] = E[ Q*(s',a*) + ε_B ] = Q*(s',a*)
```

实践中用「在线网络选、目标网络评」近似独立性，偏差显著减小但不严格为零。

### 推导 4：为什么 replay 与 target net 是必需的

- replay：训练样本时序相关，直接在线更新会遗忘/震荡；随机采样近似 i.i.d.；
- target net：若目标 `r+γmaxQ_φ` 随 φ 每步变化，回归目标是移动靶，容易发散；
  Polyak/硬更新把它变成准静态监督问题。

## 实现映射（HW2 P1）

| 概念 | 文件 | TODO | 验收 |
| --- | --- | --- | --- |
| `ε`-greedy | `homework2-online-rl/starter-code/gridworld_q_learning.py` | `choose_action` | 平局任意选择 |
| Q 更新 | 同文件 | `train_q_learning` | 每个 episode 收敛 |
| 奖励设计 | 三种场景 | `r = r_step + R1·1[Goal1] + R2·1[Goal2]` | 报告轨迹与总回报 |

三个场景的预期差异（`5×4` 网格，Goal2 距离 4 步、Goal1 距离 7 步）：

| 场景 | `r_step` | `R1`（远） | `R2`（近） | 预期行为 |
| --- | --- | --- | --- | --- |
| 1 | −1 | 10 | 5 | 比较总回报：近目标 vs 远目标（需手算） |
| 2 | −2 | 10 | 5 | 每步代价更高，更偏向近目标 |
| 3 | +1 | 1 | 1 | 每步正收益，可能永不终止或直奔目标（取决于设置） |

*具体轨迹以自己运行结果为准；关键是用折扣回报手算验证选择。*

## 易错点与反例

- **忘记 target network**：目标随参数每步变化，训练震荡；
- **`done` 时仍 bootstrap**：终止状态 Q 被错误抬高；
- **`ε` 衰减太快**：覆盖不足，Q 收敛到局部；
- **奖励尺度**：`r_step` 正负决定「尽快结束还是拖时间」，不是无关紧要的常数；
- **过估计误判**：Q 值大不代表策略好，要看实际评估回报；
- **离散动作的 tie-breaking**：`argmax` 平局要随机化，否则探索有偏。

## 实验或练习

1. 实现 `choose_action` 与 `train_q_learning`，跑三个场景并记录最优路径；
2. 关闭 target network（直接用在线网络）对比稳定性；
3. 在 gridworld 上实现 Double Q-learning，观察 Q 值与真实折扣回报的差距缩小。

## 高频问题

- 问：Q-learning 隐式的策略是什么？答：每步 `argmax_a Q(s,a)`，因此是 off-policy。
- 问：为什么要 replay buffer？答：打破时序相关性、提升数据利用率、稳定训练。
- 问：过估计的来源？答：`E[max] ≥ max[E]`，噪声越大越严重。
- 问：Double Q 为什么不引入额外偏差？答：选择与评估使用独立噪声，期望上无偏。
- 问：Q-learning 与 SARSA 的差别？答：SARSA 用实际下一动作（on-policy），Q-learning 用 `max`（off-policy）。

## 面试追问链（由浅入深）

1. Q-learning 如何在不显式学策略的情况下改进策略？→ 目标是 `max_a Q`，等价于贪心策略；
   行为策略只需覆盖动作空间。
2. 为什么 on-policy 的 SARSA 在悬崖行走上更保守？→ SARSA 评估带探索的当前策略，
   会把 `ε` 的下跌风险算进 Q；Q-learning 假设之后总是最优行动。
3. 过估计一定有害吗？→ 不一定：适度的乐观可促进探索；但在函数逼近 + bootstrap 下会失控。
4. Double Q 的两个网络需要完全独立吗？→ 理想是独立噪声；实践中在线/目标网络共享弱相关，
   仍能显著缓解但非零偏差。
5. DQN 的经典工程技巧有哪些？→ replay、target net、Huber loss、梯度裁剪、
   奖励归一化、frame stack、ε 衰减。
6. 为什么连续动作不能直接用 `max`？→ 无法枚举；用 actor 近似 argmax（DDPG/SAC）或
   采样/优化（QT-Opt）。

## 自测题（附要点）

1. 写出 Q-learning 更新并说明它为什么 off-policy。
   - 要点：目标只用 `max`，与行为策略无关。
2. 用 `|max f − max g| ≤ max|f−g|` 证明 `T` 是 `γ`-压缩。
   - 要点：见推导 1。
3. 推导 `E[max Q̂] ≥ max E[Q̂]`，并说明偏差随什么增长。
   - 要点：Jensen + `σ√(2lnA)`；动作数与噪声。
4. 写出 Double Q 的目标并说明「选」与「评」分别用哪个网络。
   - 要点：在线选、目标评。
5. 在 HW2 gridworld 中，三种奖励设置为何导向不同目标？
   - 要点：折扣总回报 = 每步代价 × 步数 + 目标奖励；正负步价改变「快 vs 慢」偏好。

## 关联笔记

- 前置：[L5 Off-policy AC](lecture-05-off-policy-actor-critic.md)
- 后续：[L7 离线 RL](lecture-07-offline-rl.md)（固定数据下的 Q 学习问题）、
  [L11 模型类 RL](lecture-11-model-based-rl.md)

## 本讲小结

Q-learning 把策略隐式藏在 `argmax` 里，是离散动作与离线数据的天然选择。Target network、
replay、Double Q 是工程必备。下一讲进入离线场景：数据固定、不能交互，Q-learning 的过估计
会被放大成核心难题。

## 延伸阅读

- Watkins & Dayan, *Q-learning*, 1992；Mnih et al., *DQN*, 2015.
- van Hasselt et al., *Double Q-learning*, 2015；*Deep RL with Double Q-learning*, 2016.
- Kalashnikov et al., *QT-Opt*, 2018.
