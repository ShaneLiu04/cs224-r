# 术语与符号表

保留英文术语；括号内为常见别名。每条尽量给出「一句话定义 + 常见陷阱」。

## 1. 符号表

| 符号 | 含义 | 出处 |
| --- | --- | --- |
| `s_t` / `o_t` | 状态 / 观测（POMDP 中 `o` 是 `s` 的部分信息） | [L1](lecture-01-intro-mdps.md) |
| `a_t` | 动作 | L1 |
| `τ` | 轨迹 `(s_1,a_1,…,s_T,a_T)` | L1 |
| `P(s'\|s,a)` | 转移概率（dynamics） | L1 |
| `r(s,a)` / `r_t` | 奖励 | L1 |
| `γ` | 折扣因子；有效时域 `≈ 1/(1−γ)` | L1 |
| `ρ_0` | 初始状态分布 | L1 |
| `d^π(s)` | 策略 π 下的状态访问分布 | [L3](lecture-03-policy-gradients.md) |
| `J(θ)` | 目标函数 `E_{τ~p_θ}[Σγ^{t-1}r_t]` | L1/L3 |
| `V^π(s)` | 状态价值 `E[Σγ^t r \| s]` | [L4](lecture-04-actor-critic.md) |
| `Q^π(s,a)` | 动作价值 | L4 |
| `A^π(s,a)` | 优势 `Q−V` | L4 |
| `δ_t` | TD 残差 `r+γV(s')−V(s)` | L4 |
| `λ` | GAE 平滑参数 / AWR 温度 | L4/[L7](lecture-07-offline-rl.md) |
| `Â_t` / `R̂_t` | GAE 优势估计 / 值回归目标 | L4 |
| `ρ_t` | 概率比 `π_θ/π_old` | [L5](lecture-05-off-policy-actor-critic.md) |
| `α` | 学习率 / SAC 熵温度 / CQL 保守系数（按上下文） | L5/L7 |
| `ε` | `ε`-greedy 概率 / PPO clip 范围 | [L6](lecture-06-q-learning.md)/L5 |
| `ζ` | expectile 参数，`ζ>0.5` 学上分位数 | L7 |
| `β` | KL/DPO 温度、温度参数 | [L9](lecture-09-rlhf-preference-optimization.md) |
| `K` | 组大小（GRPO/RLOO）、action chunk 执行长度（按上下文） | [L10](lecture-10-rl-llm-reasoning.md)/[L2](lecture-02-imitation-learning.md) |
| `z` | 任务/目标/技能隐变量 | [L12](lecture-12-multi-task-goal-conditioned-rl.md) |
| `s_g` | 目标状态（goal） | L12 |
| `k` | 模型 rollout 长度 | [L11](lecture-11-model-based-rl.md) |
| `e` | 域随机化参数 | [L16](lecture-16-sim-to-real.md) |

## 2. 公式速查（面试默写版）

| 公式 | 表达式 | 出处 |
| --- | --- | --- |
| 回报 | `G = Σ_t γ^t r_t` | [L1](lecture-01-intro-mdps.md) |
| 轨迹梯度恒等式 | `∇log p_θ(τ) = Σ_t ∇log π_θ(a_t\|s_t)` | L1/L3 |
| 策略梯度 | `∇J = E[Σ_t ∇logπ(a_t\|s_t)·R_t]` | [L3](lecture-03-policy-gradients.md) |
| baseline 无偏 | `E_{a~π}[∇logπ·b(s)] = 0` | L3 |
| TD 残差 | `δ_t = r_t + γ(1−d_t)V(s_{t+1}) − V(s_t)` | [L4](lecture-04-actor-critic.md) |
| GAE | `Â_t = δ_t + γλ(1−d_t)Â_{t+1}` | L4 |
| PPO clip | `−E[min(ρÂ, clip(ρ,1±ε)Â)]` | [L5](lecture-05-off-policy-actor-critic.md) |
| SAC Q 目标 | `y = r + γ(min Q̄ − α log π)` | L5 |
| Bellman 最优 | `Q* = r + γ max_{a'} Q*` | [L6](lecture-06-q-learning.md) |
| Double Q | `y = r + γ Q̄(s', argmax Q)` | L6 |
| AWR/AWAC | `−E[logπ·exp(A/λ)]` | [L7](lecture-07-offline-rl.md) |
| expectile | `L₂^ζ(µ)=\|ζ−1{µ≤0}\|µ²` | L7 |
| IQL-V/Q | `L₂^ζ(Q̄−V)`；`(r+γV(s')−Q)²` | L7 |
| CQL 正则 | `α(E_µ[Q] − E_D[Q])` | L7 |
| Bradley-Terry | `σ(R_w − R_l)` | [L8](lecture-08-reward-learning.md) |
| RLHF 最优策略 | `π* ∝ π_ref exp(R/β)` | [L9](lecture-09-rlhf-preference-optimization.md) |
| DPO | `−logσ(β[Δ_w−Δ_l])` | L9 |
| GRPO 优势 | `(r_i−mean)/std` | [L10](lecture-10-rl-llm-reasoning.md) |
| RLOO 优势 | `r_i − mean_{j≠i}r_j` | L10 |
| pass@k | `1−C(n−c,k)/C(n,k)` | L10 |
| MBPO | 短 rollout `k` 步 + SAC | [L11](lecture-11-model-based-rl.md) |
| HER | `g'=s_{t'}`，`r'=1` | [L12](lecture-12-multi-task-goal-conditioned-rl.md) |
| MAML | `θ'=θ−α∇L_T(θ)` | [L13](lecture-13-meta-rl.md) |
| PEARL ELBO | `E_q[logp(c\|z)]−KL(q\|\|p(z))` | L13 |
| UCB | `μ̂ + c√(log t/N)` | [L14](lecture-14-exploration.md) |
| options | `(I_ω, π_ω, β_ω)` | [L15](lecture-15-hierarchical-rl-il.md) |
| DR | `max_π E_{e~p(e)}[J]` | [L16](lecture-16-sim-to-real.md) |
| 优势条件 BC | `−E_{A>0}[logπ(a\|s,success)]` | [L17](lecture-17-rl-for-vlas.md) |

## 3. MDP 与训练基础

| 术语 | 含义 / 陷阱 |
| --- | --- |
| MDP / POMDP | 马尔可夫决策过程 / 部分可观测；后者需历史编码 |
| deterministic / stochastic policy | `a=μ(s)` / `a~π(a\|s)`；随机策略可探索且似然可导 |
| episode / rollout / trajectory | 一条完整交互 / 与环境交互收集 / 状态动作序列 |
| horizon `T` / 有效时域 | 轨迹长度 / `1/(1−γ)` |
| sparse / dense reward | 只在成功给分 / 每步给分 |
| reward shaping | 人工中间奖励；势能函数可保证策略不变 |
| return / reward-to-go | `R(τ)` / 从 `t` 起的折扣回报 `R_t` |
| on-policy / off-policy | 数据是否必须来自当前策略 |
| replay buffer | 缓存历史转移；off-policy 的样本源 |
| target network | 慢更新目标网络，避免移动靶 |
| importance sampling `ρ` | `π_θ/π_old`；连乘会爆炸 |
| bootstrap | 用自身估计构造目标；偏差与误差传播的来源 |
| discount `γ` | 折扣因子；也是信用分配范围旋钮 |
| UTD | update-to-data：critic 更新次数 / 环境步数 |

## 4. 值函数与 actor-critic

| 术语 | 含义 / 陷阱 |
| --- | --- |
| `V/Q/A` | 状态值 / 动作值 / 优势；`V=E_a[Q]`，`A=Q−V` |
| Bellman equation | `V(s)=E[r+γV(s')]`；最优形式含 `max` |
| TD error | `r+γV(s')−V(s)`；GAE 的基元 |
| MC / TD / n-step | 整条回报 / 一步 bootstrap / 折中 |
| GAE | 指数加权多步 TD 残差；`λ` 调偏差-方差 |
| actor / critic | 策略网络 / 值网络 |
| PPO / TRPO | clip 近似信任域 / 二阶 KL 约束 |
| SAC / DDPG / TD3 | 最大熵 AC / 确定性策略 / 双 Q + 平滑 + 延迟更新 |
| A2C/A3C | 同步/异步并行 actor-critic |
| entropy bonus | 熵正则，内生探索 |
| overestimation bias | `E[max Q̂] ≥ max E[Q̂]`，随动作数与噪声增大 |
| clipped double-Q | 取两个 Q 的最小，抵消向上的过估计 |

## 5. 模仿学习与生成式策略

| 术语 | 含义 / 陷阱 |
| --- | --- |
| BC | 行为克隆；分布内拟合 ≠ 闭环可用 |
| DAgger | 学习者访问状态 + 专家标注 + 聚合；需专家可查询 |
| compounding error | 闭环误差累积；朴素 BC 近似 `O(εT²)` |
| covariate shift | 测试状态分布 ≠ 训练分布 |
| action chunking | 一次预测多个动作；`T` 预测、`K` 执行 |
| receding horizon | 执行前 K 个后重查，保留反馈 |
| multimodal action | 同一状态多个合理动作；MSE 会均值坍缩 |
| diffusion policy | 迭代去噪生成动作 |
| flow matching | 回归速度场 + ODE 采样；`a_τ=τa₁+(1−τ)a₀` |
| MoG / 自回归离散化 | 另外两种多模态建模方式 |
| embodiment gap | 人类演示与机器人形态差异 |

## 6. 离线 RL

| 术语 | 含义 / 陷阱 |
| --- | --- |
| offline RL / batch RL | 固定数据集，无探索 |
| OOD action | 数据分布外动作；Q 值无监督 |
| extrapolation error | OOD 价值误差被 bootstrap 放大 |
| AWR / AWAC | 优势加权回归 / 加权 actor-critic |
| IQL | expectile 学 V + 只用数据动作更新 Q |
| CQL | 显式惩罚 OOD 动作价值 |
| expectile `ζ` | 非对称 L2；`ζ>0.5` 学上分位，近似 max |
| stitching | 拼接不同轨迹的优质片段 |
| filtered BC | 只用高回报数据做模仿 |
| D4RL / AntMaze / PointMass | 数据集基准与任务 |
| one-step RL / RLPD / Cal-QL | 离线-在线混合的常见基线 |

## 7. 奖励、偏好与 LLM RL

| 术语 | 含义 / 陷阱 |
| --- | --- |
| reward model `R_φ` | 从偏好学出的奖励模型 |
| Bradley-Terry | 成对偏好概率 `σ(R_w−R_l)`；奖励平移不变 |
| RLHF / RLAIF | 人类 / AI 反馈的 RL 对齐 |
| KL penalty | `−β log(π/π_ref)`，约束偏离参考模型 |
| reward hacking | 钻代理奖励空子；训练奖励涨、真实偏好跌 |
| over-optimization | 过度优化代理导致真实目标下降 |
| DPO | 隐式奖励 + 偏好分类损失，无需 RM/采样 |
| IPO | 间隔平方损失，防 DPO 过拟合偏好对 |
| RLVR | 可验证奖励（规则/程序判定） |
| GRPO | 组内归一化优势，无 critic |
| RLOO | leave-one-out 基线，无偏且无 critic |
| pass@k | k 次采样至少一次成功的概率；衡量探索覆盖 |
| test-time compute | 推理时算力（CoT、采样、验证、搜索） |
| CoT / self-consistency | 链式思考 / 多数投票 |
| verifier / process reward | 验证器 / 过程奖励模型 |

## 8. 多任务、元学习与探索

| 术语 | 含义 / 陷阱 |
| --- | --- |
| task distribution `p(T)` | 任务分布 |
| conditional MDP | 状态/奖励/策略条件于 `z` |
| multi-task RL / IL | 多任务强化/模仿学习 |
| GCRL / `s_g` | 目标条件 RL / 目标状态 |
| HER | 用实际到达状态重标目标；目标分布有偏 |
| meta-RL | 少量交互快速适应新任务的 RL |
| RL² / MAML / PEARL | 隐状态适应 / 学初始化 / 概率上下文 |
| posterior sampling | 从任务后验采样假设再最优行动 |
| intrinsic reward | 新奇度/信息增益奖励 |
| ICM / RND | 预测误差 / 随机网络蒸馏 |
| UCB / Thompson sampling | 乐观面对不确定性 / 后验采样 |
| noisy TV | 内在奖励被随机区域吸引的陷阱 |

## 9. 层次、机器人与 VLA

| 术语 | 含义 / 陷阱 |
| --- | --- |
| options / semi-MDP | 带终止条件的时序抽象 / 决策间隔可变 |
| skill discovery / DIAYN | 无监督技能发现 / 最大化 `I(s;z)` |
| hierarchical IL / RL | 子目标高层 + 执行低层 |
| high-level DAgger | 人工纠正子目标，覆盖错误传播 |
| VLM / VLA | 视觉语言模型 / 视觉-语言-动作模型 |
| IL plateau | 模仿学习成功率平台（约 80%） |
| advantage conditioning | 把优势作为条件做监督学习 |
| residual policy | 在基座策略上加残差的小策略 |
| diffusion steering | 冻结 VLA，学噪声编辑策略 |
| domain randomization | 随机化仿真参数学鲁棒策略 |
| system identification | 用真机数据拟合仿真参数 |
| residual dynamics | 仿真 + 学残差修正 |
| actuator net | 电机力矩模型 |
| RMA | 特权信息 → 历史蒸馏的适应方法 |
| asymmetric actor-critic | critic 用特权信息，actor 不用 |
| retargeting / tracking | 人类动作重定向 / 跟踪参考动作 |
| sim2real gap | 动力学与感知两类失配 |

## 10. 易混概念对照

| A | B | 一句话区分 |
| --- | --- | --- |
| on-policy | off-policy | 数据是否必须来自当前策略 |
| MC | TD | 整条回报无偏高方差 / 一步 bootstrap 有偏低方差 |
| PPO | TRPO | clip 近似 / 二阶 KL 约束 |
| PPO | SAC | 稳定、并行、on-policy 复用 / 高样本效率、replay |
| Q-learning | SARSA | 目标 `max`（off-policy） / 实际下一动作（on-policy） |
| AWR | AWAC | 纯加权模仿 / 加权 + Bellman critic |
| AWAC | IQL | 目标动作采样 `a'~π` / 用 `V(s')`，只碰数据动作 |
| IQL | CQL | 结构性避免 OOD / 显式惩罚 OOD |
| RLHF | DPO | 显式 RM + 在线 RL / 隐式奖励 + 离线分类 |
| DPO | IPO | log-sigmoid / 间隔平方 |
| GRPO | RLOO | 组均值方差归一化 / leave-one-out 均值 |
| multi-task | meta-RL | 测试任务已知 / 需从经验推断 |
| RL² | MAML | 隐状态适应 / 少量梯度步适应 |
| DR | system ID | 随机化求鲁棒 / 用真机数据缩不确定性 |
| teacher-student | asymmetric AC | 测试用 student / critic 专属特权信息 |
| HER | GCRL | 重标技术 / 问题设定（`z=s_g`） |
| VLA IL | VLA RL | 模仿演示 / 用奖励后训练突破平台 |

## 11. 常用缩写

| 缩写 | 全称 |
| --- | --- |
| MDP / POMDP | Markov Decision Process / Partially Observable MDP |
| IL / BC / DAgger | Imitation Learning / Behavior Cloning / Dataset Aggregation |
| PG / AC | Policy Gradient / Actor-Critic |
| PPO / TRPO / GAE | Proximal Policy Optimization / Trust Region Policy Optimization / Generalized Advantage Estimation |
| SAC / DDPG / TD3 | Soft Actor-Critic / Deep Deterministic PG / Twin Delayed DDPG |
| DQN / DDQN / PER | Deep Q-Network / Double DQN / Prioritized Experience Replay |
| RLHF / DPO / IPO | RL from Human Feedback / Direct Preference Optimization / Identity Preference Optimization |
| RLVR / GRPO / RLOO | RL with Verifiable Rewards / Group Relative Policy Optimization / REINFORCE Leave-One-Out |
| MBRL / MPC / MPPI / CEM | Model-Based RL / Model Predictive Control / Model Predictive Path Integral / Cross-Entropy Method |
| IQL / CQL / AWR / AWAC | Implicit Q-Learning / Conservative Q-Learning / Advantage-Weighted Regression / Advantage-Weighted Actor-Critic |
| HER / GCRL | Hindsight Experience Replay / Goal-Conditioned RL |
| VLA / VLM | Vision-Language-Action / Vision-Language Model |
| D4RL / MJPC | Datasets for Deep Data-Driven RL / MuJoCo MPC |
| RMA / DR | Rapid Motor Adaptation / Domain Randomization |
