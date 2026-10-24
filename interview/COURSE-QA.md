# 课程问答索引

一句话答案。推导步骤在 [FORMULAS.md](FORMULAS.md)，连续追问在各讲笔记的「面试追问链」，
实验数字在 [PORTFOLIO.md](PORTFOLIO.md)。本页从课程 README 移出，避免和入口页重复。

符号：`s` 状态、`o` 观测、`a` 动作、`γ` 折扣、`ρ` 概率比、`A` 优势、`ζ` expectile、`β/λ` 温度。

## 公式

| 量 | 公式 | 说明 |
| --- | --- | --- |
| 回报 | `G = Σ_{t=0}^{T} γ^t r_t` | 折扣累积奖励 |
| 策略梯度 | `∇J(θ) = E_{τ~πθ}[Σ_t ∇log πθ(a_t\|s_t) · R_t]` | `R_t` 为 reward-to-go |
| causality 化简 | `E[Σ_t ∇log πθ(a_t\|s_t) · (Σ_{t'<t} r_{t'})] = 0` | 去掉与未来动作无关的项 |
| baseline | `∇J = E[Σ_t ∇log πθ(a_t\|s_t) · (R_t − b(s_t))]` | 任意只依赖状态的 `b` 不引入偏差 |
| 优势 | `A^π(s,a) = Q^π(s,a) − V^π(s)` | 相对基线的度量 |
| 贝尔曼最优 | `Q*(s,a) = r + γ max_{a'} Q*(s',a')` | Q-learning 的更新目标 |
| TD 残差 / GAE | `δ_t = r_t + γV(s_{t+1}) − V(s_t)`；`Â_t = δ_t + γλÂ_{t+1}` | 偏差-方差折中 |
| Q-learning 更新 | `Q(s,a) ← Q(s,a) + α[r + γ max_{a'} Q(s',a') − Q(s,a)]` | off-policy，max 带来过估计 |
| Double Q | `y = r + γ Q_{θ'}(s', argmax_{a'} Q_θ(s',a'))` | 解耦选择与评估 |
| AWR/AWAC | `L_π = −E_D[log π(a\|s) · exp(A(s,a)/λ)]` | advantage 加权模仿 |
| Expectile | `L_2^ζ(µ) = \|ζ − 1{µ≤0}\| µ²` | `ζ>0.5` 时学上分位数 |
| IQL | `L_V = E[L_2^ζ(Q_{\bar θ}(s,a) − V_ψ(s))]`；`L_Q = E[(r + γV_ψ(s') − Q_θ(s,a))²]` | 不查询 OOD 动作 |
| CQL 正则 | `α E_{a~µ}[Q(s,a)] − α E_{a~D}[Q(s,a)]` | 压低 OOD 动作价值 |
| DPO | `L = −E[log σ(β log(π_θ(y_w\|x)/π_ref) − β log(π_θ(y_l\|x)/π_ref))]` | 隐式奖励 `β log(π_θ/π_ref)` |
| GRPO/RLOO 优势 | `Â_i = r_i − mean_{j≠i} r_j`（leave-one-out） | 组内相对优势，无 critic |
| PPO clip | `L = −E[min(ρ_t Â_t, clip(ρ_t,1±ε) Â_t)]` | 多轮复用同一批数据 |
| Flow matching | `a_τ = τa_1 + (1−τ)a_0`；目标 `v = a_1 − a_0` | Euler 积分 `a ← a + v/n` |
| 动作块 | 预测 `T=20` 个动作，执行 `K=10` 后重查 | receding horizon |

术语对照见 [GLOSSARY](../notes/GLOSSARY.md)。

## 模仿学习与策略梯度（L1–L5）

| 问题 | 答题要点 | 笔记 |
| --- | --- | --- |
| BC 为什么会 compounding error？ | 测试时状态偏离训练分布，误差随步数超线性累积（`O(T²ε)` 直觉） | [L2](../notes/lecture-02-imitation-learning.md) |
| 为什么 BC 在 multimodal 数据上会「平均」？ | MSE 对多峰条件分布求均值，产生数据分布外的动作 | [L2](../notes/lecture-02-imitation-learning.md) |
| action chunking 为什么有效？ | 减少每步重查带来的误差累积和推理开销；receding horizon 保留反馈 | [L2](../notes/lecture-02-imitation-learning.md) |
| flow matching 与 diffusion 的关系？ | 都学从噪声到数据的变换；FM 直接回归速度场，采样用 ODE | [L2](../notes/lecture-02-imitation-learning.md) |
| DAgger 为什么能修正分布偏移？ | 用当前策略访问的状态加专家标注，聚合后覆盖出错状态 | [L2](../notes/lecture-02-imitation-learning.md) |
| REINFORCE 的梯度怎么推？ | `∇log P(τ)` 展开后环境项不含 θ | [L3](../notes/lecture-03-policy-gradients.md) |
| reward-to-go 与 causality？ | `t' < t` 的奖励与 `a_t` 独立，条件期望为 0 | [L3](../notes/lecture-03-policy-gradients.md) |
| baseline 为什么无偏？ | `E_{a~π}[∇log π(a\|s)·b(s)] = b(s)∇Σπ = 0` | [L3](../notes/lecture-03-policy-gradients.md) |
| MC vs TD 估计 V？ | MC 无偏高方差；TD 有偏低方差，可在线更新 | [L4](../notes/lecture-04-actor-critic.md) |
| GAE 在做什么？ | 指数加权多步 TD 残差；`λ=0` 为 TD，`λ=1` 接近 MC | [L4](../notes/lecture-04-actor-critic.md) |
| PPO 为什么 clip？ | 限制新旧策略概率比，旧数据上的更新步长有界 | [L5](../notes/lecture-05-off-policy-actor-critic.md) |
| SAC 的三个组件？ | 最大熵目标、clipped double-Q、重参数化与自动温度 | [L5](../notes/lecture-05-off-policy-actor-critic.md) |

## 值方法、离线 RL 与奖励（L6–L9）

| 问题 | 答题要点 | 笔记 |
| --- | --- | --- |
| Q-learning 为什么 off-policy？ | 目标是 `max_a Q`，不依赖采样策略的动作分布 | [L6](../notes/lecture-06-q-learning.md) |
| 为什么会 overestimation？ | `max` 作用在带噪估计上，`E[max Q̂] ≥ max E[Q̂]` | [L6](../notes/lecture-06-q-learning.md) |
| Double Q 改了什么？ | 一组网络选动作，另一组评估 | [L6](../notes/lecture-06-q-learning.md) |
| 离线 RL 的核心难点？ | 数据外动作的 Q 不可信，bootstrapping 会放大 | [L7](../notes/lecture-07-offline-rl.md) |
| AWR 与 BC 的区别？ | `exp(A/λ)` 加权，好动作多模仿 | [L7](../notes/lecture-07-offline-rl.md) |
| expectile 为什么有用？ | 非对称 L2 学上分位数，逼近数据里较高的动作价值 | [L7](../notes/lecture-07-offline-rl.md) |
| IQL 与 AWAC 的区别？ | IQL 用 `V(s')`，不在下一状态采样动作 | [L7](../notes/lecture-07-offline-rl.md) |
| CQL 在惩罚什么？ | 压低数据外动作的 Q，抬高数据动作的 Q | [L7](../notes/lecture-07-offline-rl.md) |
| goal classifier 被 exploit 怎么办？ | 分类器漏洞会被 RL 钻到；要负样本或约束 | [L8](../notes/lecture-08-reward-learning.md) |
| 偏好数据怎么变成奖励？ | Bradley-Terry 加交叉熵，学 `R(x,y)` | [L8](../notes/lecture-08-reward-learning.md) |
| DPO 为什么可以不训 reward model？ | 奖励写成 `β log(π_θ/π_ref)`，直接对策略做偏好损失 | [L9](../notes/lecture-09-rlhf-preference-optimization.md) |
| reward over-optimization？ | proxy 上升时真实偏好可以下降，用 KL 或早停 | [L9](../notes/lecture-09-rlhf-preference-optimization.md) |

## LLM RL、模型与元学习（L10–L14）

| 问题 | 答题要点 | 笔记 |
| --- | --- | --- |
| GRPO 与 PPO 的差别？ | 去掉 value network，用同 prompt 一组样本做优势 | [L10](../notes/lecture-10-rl-llm-reasoning.md) |
| RLOO 的 baseline？ | `Â_i = r_i − mean_{j≠i} r_j`，不需要 critic | [L10](../notes/lecture-10-rl-llm-reasoning.md) |
| test-time scaling 怎么比？ | 画准确率对采样数，不比单点 | [L10](../notes/lecture-10-rl-llm-reasoning.md) |
| MBRL 的风险？ | 样本效率高，模型误差会被策略利用 | [L11](../notes/lecture-11-model-based-rl.md) |
| MBPO 为什么短 rollout？ | 从真实状态出发，限制展开长度 | [L11](../notes/lecture-11-model-based-rl.md) |
| goal-conditioned 与 multi-task？ | GCRL 是目标作为条件变量的 multi-task | [L12](../notes/lecture-12-multi-task-goal-conditioned-rl.md) |
| HER 的核心操作？ | 用实际到达的状态重标记目标 | [L12](../notes/lecture-12-multi-task-goal-conditioned-rl.md) |
| RL² 与 MAML？ | RL² 把历史放进 RNN；MAML 学一个可快速适应的初始化 | [L13](../notes/lecture-13-meta-rl.md) |
| meta-RL 的探索难在哪？ | 要为识别任务收集信息，探索和执行耦合 | [L13](../notes/lecture-13-meta-rl.md) |
| intrinsic reward 有哪些？ | 计数、预测误差（ICM）、RND、熵 | [L14](../notes/lecture-14-exploration.md) |

## 层次、机器人与前沿（L15–L18）

| 问题 | 答题要点 | 笔记 |
| --- | --- | --- |
| 层次化的三个设计问题？ | 子目标怎么表示、各层怎么监督、何时切换 | [L15](../notes/lecture-15-hierarchical-rl-il.md) |
| 高层策略为什么用 DAgger？ | 错误子目标会累积，用纠正数据重新标注 | [L15](../notes/lecture-15-hierarchical-rl-il.md) |
| domain randomization？ | 随机化动力学和感知参数，学对这个分布鲁棒的策略 | [L16](../notes/lecture-16-sim-to-real.md) |
| 为什么 sim-to-real 仍常用 PPO？ | on-policy 稳定，并行仿真吞吐高 | [L16](../notes/lecture-16-sim-to-real.md) |
| VLA 加 RL 为什么难？ | 模型大、真实采样贵、奖励稀疏 | [L17](../notes/lecture-17-rl-for-vlas.md) |
| 离线微调 VLA 的一种配方？ | 监督学 V，再用 advantage-conditioned 行为克隆 | [L17](../notes/lecture-17-rl-for-vlas.md) |
| 在线微调的轻量方案？ | 冻结 VLA，只学动作残差 | [L17](../notes/lecture-17-rl-for-vlas.md) |
| 怎么开始做一个 RL 实验？ | 复现 baseline，一次只改一个变量，失败的设置也留下 | [L18](../notes/lecture-18-frontiers.md) |
