# Lecture 14 — Exploration（2025 版专题）

- 课程版本：Spring 2025（2026 版未开设本讲，作为补充）
- 日期：2025-05-16（2026 对应周次无课）
- 官方材料：[14_cs224r_exploration_2025.pdf](../slides/spring2025/14_cs224r_exploration_2025.pdf) ·
  [Lecture 14 视频](https://www.youtube.com/watch?v=4tlSKdi8teU)
- 状态：初稿（依据 2025 slides 整理）

## 学习目标

- 区分 exploration 与 exploitation，理解 regret/样本复杂度的基本权衡；
- 掌握 `ε`-greedy、softmax、UCB/计数法、Thompson sampling 的基本形式；
- 理解内在奖励：预测误差（ICM）、随机网络蒸馏（RND）、熵正则、信息增益；
- 知道稀疏奖励下探索为何困难，以及常见的组合策略。

## 先修知识

- L3–L6 的策略/值方法；多臂老虎机基本概念；KL 与熵。

## 核心概念

### 1. 为什么需要探索

最优策略需要全局信息，但智能体只能通过行动获得信息：

- **利用（exploitation）**：选择当前估计最优的动作；
- **探索（exploration）**：尝试可能更差但能提供信息的动作。

遗憾（regret）目标：与事后最优策略的累计差距 `Regret(T) = T·r* − Σ_t r_t`。
在多臂老虎机中，UCB/Thompson 可达到对数级遗憾；在 MDP 中问题更难。

### 2. 简单探索策略

| 方法 | 形式 | 特点 |
| --- | --- | --- |
| `ε`-greedy | 以 `ε` 概率随机 | 简单；不利用不确定性 |
| Softmax/Boltzmann | `a ~ softmax(Q/τ)` | 按价值差距分配概率 |
| 乐观初始化 | `Q ← Q_max` | 鼓励访问未尝试动作 |
| 计数/UCB | `a = argmax Q + c√(log t / N(a))` | 理论保证，离散/表格友好 |

深度 RL 中 `ε` 衰减、熵正则（SAC/PPO）是最常用的实用探索。

### 3. 内在奖励（intrinsic motivation）

当外部奖励稀疏时，构造「新奇度」奖励：

- **计数/伪计数**：状态访问次数（或密度模型导出的伪计数）；
- **ICM**：用前向模型预测误差 `||φ(s') − f(φ(s),a)||²` 作为好奇奖励，
  同时可训练逆模型提取与动作相关的特征；
- **RND（Random Network Distillation）**：固定随机网络输出为目标，
  训练预测网络；预测误差大 = 新颖状态；
- **信息增益/熵**：鼓励减少对任务参数的不确定性（与 L13 后验采样呼应）。

风险：**噪声电视（noisy TV）问题**——随机性强的区域永远有预测误差，导致智能体沉迷。
缓解：只对可控/与环境交互相关信息给奖励，或用集成/贝叶斯不确定性。

### 4. 基于不确定性的探索

- **贝叶斯/集成 Q**：不确定性高（方差大）的动作获得探索加成（如 UCB 风格
  `a = argmax Q + κσ`）；
- **Thompson sampling**：从 Q 的后验采样一个假设再贪心（posterior sampling）；
- **RND/ICM** 是它的近似实现。

### 5. 最大熵 RL

SAC 的目标 `E[Σ r + αH(π)]` 把探索写进目标函数：

```text
J(π) = E[ Σ_t ( r_t + α H(π(·|s_t)) ) ]
```

好处：不依赖外部探索启发式；温度 `α` 自动调节熵水平。PPO 中常用熵系数 `c_ent`。

### 6. 稀疏奖励的组合拳

实践上通常组合使用：

1. 演示/BC 预热（HW2 的 BC 预训练、RLPD）；
2. 课程学习与目标重标（HER，L12）；
3. 内在奖励（RND/ICM）；
4. 后验采样/集成不确定性（PEARL、SAC+ensemble）；
5. 随机化环境参数（domain randomization 也是被动的探索）。

## 关键推导：UCB 的直觉

对每个动作维护均值 `μ̂(a)` 与访问次数 `N(a)`，选择

```text
a_t = argmax_a [ μ̂(a) + c √( log t / N(a) ) ]
```

- 第一项利用，第二项探索；
- `N` 小则探索加成大；`t` 大则被迫逐步尝试所有动作；
- 在独立同分布老虎机下可达 `O(log T)` regret。MDP 中状态相关版本需要
  count 或不确定性的函数近似。

## 实现映射

| 概念 | 材料 | 说明 |
| --- | --- | --- |
| `ε`-greedy | `assignments/spring2026/homework2-online-rl/starter-code/gridworld_q_learning.py` | HW2 P1 |
| 熵正则 | `homework2-online-rl/.../on_policy.py`（PPO 配置）、SAC 系 | `entropy_coef` |
| 内在奖励 | 无直接作业；可做期末项目 | RND/ICM 实现练习 |
| 后验采样 | HW4 meta-RL（2025/2023） | PEARL |

## 易错点与反例

- **`ε` 一直很大**：策略永远无法收敛到最优；
- **内在奖励压过外部奖励**：需要退火或归一化；
- **噪声电视问题**：预测误差奖励会被随机区域吸引；
- **只探索不加利用**：replay 数据分布过散，critic 学不好；
- **探索与安全**：真机探索需要约束（safe RL、shield）。

## 实验或练习

1. 在 Gridworld 上对比 `ε`-greedy、softmax、UCB 的收敛步数；
2. 在稀疏奖励导航中实现 RND，观察是否出现「卡在噪声区域」并设计缓解；
3. 把 SAC 的熵温度固定/自动调节做对比，记录策略熵曲线。

## 高频问题

- 问：探索-利用的数学框架？答：遗憾最小化；老虎机有对数遗憾算法。
- 问：RND 与 ICM 的差别？答：RND 用固定随机网络的预测误差；ICM 用前向动力学误差 + 逆模型。
- 问：SAC 的熵项与 `ε`-greedy 的关系？答：前者是目标函数内生的探索，后者是外挂噪声。
- 问：posterior sampling 为什么是好的探索？答：按当前信念随机选假设并最优行动，天然平衡信息与回报。
- 问：稀疏奖励的最佳实践？答：演示预热 + 课程/重标 + 内在奖励 + 不确定性，多管齐下。

## 本讲小结

探索是 RL 样本效率的根源性问题。实用探索 = 熵/`ε`（局部） + 内在奖励（全局新奇） +
不确定性/后验采样（信息导向） + 课程与演示（结构化）。这些工具会在元学习与 LLM RL 中再次出现。

## 延伸阅读

- Pathak et al., *ICM*, 2017；Burda et al., *RND*, 2019.
- Osband et al., *Bootstrapped DQN*, 2016；Russo et al., *Thompson Sampling*, 2018.
- Auer et al., *UCB*, 2002；Haarnoja et al., *SAC*, 2018.
- 官方 2025 slides：`slides/spring2025/14_cs224r_exploration_2025.pdf`。
