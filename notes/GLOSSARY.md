# 术语与符号表

按主题分组，保留英文术语；括号内为常见别名。

## MDP 与轨迹

| 符号/术语 | 含义 |
| --- | --- |
| `s_t` / `o_t` | 状态 / 观测（POMDP 中 `o` 是 `s` 的部分信息） |
| `a_t` | 动作 |
| `τ` | 轨迹 trajectory `(s_1,a_1,…,s_T,a_T)` |
| `P(s' \| s,a)` | 转移概率 dynamics |
| `r(s,a)` / `r_t` | reward |
| `γ` | 折扣因子；有效时域 `≈ 1/(1−γ)` |
| `ρ_0` | 初始状态分布 |
| `d^π(s)` | 策略 π 下的状态访问分布 |
| MDP / POMDP | 马尔可夫决策过程 / 部分可观测 |
| deterministic / stochastic policy | `a=μ(s)` / `a ~ π(a\|s)` |
| episode / rollout | 一条轨迹 / 用策略与环境交互收集轨迹 |
| horizon `T` | 轨迹长度 |
| sparse / dense reward | 只在成功时给分 / 每步给分 |
| reward shaping | 人工设计中间奖励 |
| return / reward-to-go | `R(τ)` / 从 t 起的折扣回报 `R_t` |

## 值函数与算法

| 符号/术语 | 含义 |
| --- | --- |
| `V^π(s)` | 状态价值 `E[Σγ^t r \| s_0=s]` |
| `Q^π(s,a)` | 动作价值（state-action value） |
| `A^π(s,a)` | 优势 `Q−V` |
| Bellman equation | `V(s)=E[r+γV(s')]`；最优形式含 `max` |
| TD error `δ_t` | `r + γV(s') − V(s)` |
| MC / TD / n-step | 用整条回报 / 一步 bootstrap / 中间折中估计 value |
| bootstrap | 用自身估计构造目标 |
| `λ` (GAE / AWR 温度) | GAE 平滑参数 / advantage 加权温度 |
| GAE | Generalized Advantage Estimation |
| on-policy / off-policy | 数据是否必须来自当前策略 |
| replay buffer | 缓存历史转移的 buffer |
| target network | 慢更新的目标网络 |
| actor / critic | 策略网络 / 值网络 |
| importance sampling `ρ` | `π_θ(a\|s)/π_old(a\|s)` |
| PPO / TRPO | 截断 / 信赖域策略优化 |
| SAC / DDPG / TD3 | off-policy actor-critic 家族 |
| DQN / Double DQN | 深度 Q-learning 及其过估计修正 |
| overestimation bias | `E[max Q̂] ≥ max E[Q̂]` 导致的正偏差 |
| UTD | update-to-data ratio，critic 更新次数 / 环境步数 |

## 模仿学习与生成式策略

| 符号/术语 | 含义 |
| --- | --- |
| BC | Behavior Cloning，行为克隆 |
| DAgger | Dataset Aggregation，在线聚合专家标注 |
| compounding error | 闭环执行中误差累积 |
| covariate shift | 测试状态分布 ≠ 训练状态分布 |
| action chunking | 一次预测多个未来动作 |
| receding horizon control | 只执行前 K 个动作后重查 |
| multimodal action distribution | 同一状态下多个可选动作 |
| diffusion policy | 用扩散模型生成动作的策略 |
| flow matching / rectified flow | 学习噪声到数据速度场的生成方法 |
| Euler integration | 用固定步长解 ODE 的采样方法 |
| MoG | Mixture of Gaussians |
| expert / intervention | 专家策略 / 人工接管纠正 |

## 离线 RL、奖励与几何

| 符号/术语 | 含义 |
| --- | --- |
| offline RL / batch RL | 仅用固定数据集的 RL |
| OOD action | 数据分布外动作 |
| extrapolation error | OOD 上价值估计误差被 bootstrap 放大 |
| AWR / AWAC | Advantage-Weighted Regression / Actor-Critic |
| IQL | Implicit Q-Learning |
| CQL | Conservative Q-Learning |
| expectile `ζ` | 非对称 L2 分位数参数；`ζ>0.5` 学上分位 |
| stitching | 拼接数据中不同轨迹的优质片段 |
| D4RL / AntMaze / PointMass | 离线 RL 数据集与任务 |
| filtered BC | 只用高回报数据的 BC |
| reward model `R_φ` | 从偏好学出的奖励模型 |
| Bradley-Terry | 成对偏好概率模型 `σ(R_w−R_l)` |
| RLHF / RLAIF | 人类 / AI 反馈的 RL 对齐 |
| DPO / IPO | 直接偏好优化 / 身份偏好优化 |
| KL penalty | 约束策略偏离参考模型的惩罚项 |
| reward hacking / over-optimization | 钻代理奖励空子 / 训练奖励涨而真实偏好跌 |
| RLVR / verifiable reward | 可验证奖励（规则/程序判定） |
| GRPO / RLOO | 组内相对优势的无 critic 策略优化 |
| pass@k | k 次采样至少一次成功的概率 |
| test-time compute | 推理时算力（CoT、采样、搜索） |

## 多任务、层次与机器人

| 符号/术语 | 含义 |
| --- | --- |
| task distribution `p(T)` | 任务分布 |
| conditional MDP | 状态/奖励/策略条件于任务 `z` 的 MDP |
| multi-task RL / IL | 多任务强化/模仿学习 |
| GCRL | Goal-Conditioned RL，目标条件 RL |
| `s_g` | 目标状态 |
| HER | Hindsight Experience Replay，事后重标目标 |
| meta-RL | 元强化学习：少量交互快速适应新任务 |
| RL² / MAML / PEARL | 黑盒元 RL / 学初始化 / 概率上下文元 RL |
| posterior sampling | 从任务后验采样假设再最优行动 |
| intrinsic reward | 内在奖励（新奇度、信息增益） |
| ICM / RND | 好奇心模块 / 随机网络蒸馏 |
| UCB / Thompson sampling | 乐观面对不确定性 / 后验采样 |
| options / semi-MDP | 带终止条件的时序抽象 |
| skill discovery / DIAYN | 无监督技能发现 |
| hierarchical IL / RL | 层次化模仿/强化学习 |
| domain randomization | 随机化仿真参数以迁移真机 |
| sim2real / real2sim | 仿真到真实 / 真实数据校准仿真 |
| system identification | 系统辨识 |
| RMA / asymmetric actor-critic | 特权信息蒸馏 / 非对称 actor-critic |
| retargeting / tracking | 人类动作重定向 / 跟踪参考动作 |
| VLA | Vision-Language-Action 模型 |
| residual policy | 在基座策略上加残差的小策略 |
| advantage conditioning | 把优势作为条件做监督学习 |

## 常用缩写

| 缩写 | 全称 |
| --- | --- |
| MDP | Markov Decision Process |
| IL / BC / DAgger | Imitation Learning / Behavior Cloning / Dataset Aggregation |
| PG / AC | Policy Gradient / Actor-Critic |
| PPO / TRPO / GAE | Proximal Policy Optimization / Trust Region Policy Optimization / Generalized Advantage Estimation |
| RLHF / DPO / IPO | RL from Human Feedback / Direct Preference Optimization / Identity Preference Optimization |
| RLVR / GRPO / RLOO | RL with Verifiable Rewards / Group Relative Policy Optimization / REINFORCE Leave-One-Out |
| MBRL / MPC / MPPI / CEM | Model-Based RL / Model Predictive Control / Model Predictive Path Integral / Cross-Entropy Method |
| IQL / CQL / AWR / AWAC | Implicit Q-Learning / Conservative Q-Learning / Advantage-Weighted Regression / Advantage-Weighted Actor-Critic |
| HER / GCRL | Hindsight Experience Replay / Goal-Conditioned RL |
| VLA / VLM | Vision-Language-Action / Vision-Language Model |
| D4RL / MJPC | Datasets for Deep Data-Driven RL / MuJoCo MPC |
