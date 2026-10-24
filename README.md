# CS229-R

![面向面试的深度强化学习](assets/cs229r-cover.png)

面向面试的深度强化学习。笔记、公式和多种子实验只为一件事服务：当场推导，比较算法，把曲线上的数字讲清楚。

材料来自 Stanford CS224R（Chelsea Finn，2026 / 2025 / 2023）。作者 [ShaneLiu04](https://github.com/ShaneLiu04)。与 Stanford 及课程团队无隶属关系。`solutions/` 是个人复现，正在修读该课的同学不要把它当作作业提交。

## 三张图

留出集 500 题，每题 16 条。长 think SFT 的 pass@1 是 0.272，pass@16 是 0.662。长程 RL 把 pass@1 抬到 0.52–0.54，pass@16 停在 0.64–0.71。面试里把这件事讲成分布变尖：小 k 上升，大 k 没有拉开。

![留出集 pass@k](solutions/spring2026/default-project/artifacts/extension/figures/ext_fig1_passk_heldout.png)

同一份 Medium 迷宫。IQL（ζ=0.7）在 RND 数据上是 −50.9，在随机数据上是 −90.7。CQL（α=0.1，6 个种子）反过来：随机数据 −34.3±6.1，RND 数据 −67.6。数据没有绝对的好坏，要看算法怎么用它。starter 里的 `--use_rnd` 并不改变数据，采集器写死了最短路贪心。

![CQL 的两种数据](solutions/spring2025/homework3-offline-rl/artifacts/figures/hw3_fig8_cql_data_source.png)

bit-flip 的奖励只有 0 和 −1。无 hindsight 时 25-bit 成功率是 0。final、future、random 三种重标记都到 1.0。

![HER，25-bit](solutions/spring2025/homework4-goal-meta-rl/artifacts/figures/hw4_fig1_bitflip_success.png)

网格上走路是 0.1，读地图再坐公交是 0.7。DREAM 0.696±0.007，RL² 0.143±0.073。把每步探索代价设为 0，回报仍是 0.70，探索长度顶到回合上限。

## 开口的一句

往下的推导在笔记，90 秒的讲法在 [作品集](interview/PORTFOLIO.md)，可手推的步骤在 [公式卡](interview/FORMULAS.md)。

| 模块 | 开口 | 笔记 | 证据 |
| --- | --- | --- | --- |
| 模仿 | MSE 在多峰动作上取条件均值，落到数据没有的动作上。测试时状态偏离训练集，误差随步数复合。 | [L2](notes/lecture-02-imitation-learning.md) | [HW1](solutions/spring2026/homework1-imitation/README.md) |
| 策略梯度 | `t` 之前的奖励与 `a_t` 独立，那一项的梯度期望是 0。只依赖状态的 baseline 同样无偏。 | [L3](notes/lecture-03-policy-gradients.md) | [公式卡](interview/FORMULAS.md) |
| GAE | `λ=0` 是一步 TD，`λ=1` 回到 reward-to-go。中间换的是偏差和方差。 | [L4](notes/lecture-04-actor-critic.md) | [HW2](solutions/spring2026/homework2-online-rl/README.md) |
| PPO / SAC | clip 限制的是 `π_new/π_old`。两个 Q 取 min 是往下偏，用来压过估计。 | [L5](notes/lecture-05-off-policy-actor-critic.md) | [HW2](solutions/spring2026/homework2-online-rl/README.md) |
| Q-learning | `max` 作用在带噪声的估计上，`E[max Q̂] ≥ max E[Q̂]`。 | [L6](notes/lecture-06-q-learning.md) | [HW2](solutions/spring2026/homework2-online-rl/README.md) |
| 离线 | IQL 用 `V(s')`，不在下一状态采样动作。同一份 RND 数据对 IQL 更好，对强 CQL 更差。 | [L7](notes/lecture-07-offline-rl.md) | [2026](solutions/spring2026/homework3-offline-rl/README.md)、[2025](solutions/spring2025/homework3-offline-rl/README.md) |
| 偏好 | KL 约束下的最优奖励是 `β log(π/π_ref)`。配分函数在两条回答相减时消掉，所以损失可以直接写在策略上。`β=0.1` 且策略等于参考策略时，IPO 损失是 25。 | [L9](notes/lecture-09-rlhf-preference-optimization.md) | [Countdown](solutions/spring2026/default-project/README.md) |
| 推理 RL | RLOO 不除组内标准差，GRPO 除。低方差的题（几乎全对或全错）会被放大。损失再按长度平均，长的错误回答每个 token 分到的梯度更小。 | [L10](notes/lecture-10-rl-llm-reasoning.md) | [Q13](interview/QUESTION-BANK.md) |
| HER | 把目标换成后来实际到达的状态。确定性环境里转移不变。随机环境里，到达这个状态的转移是在原目标下发生的。 | [L12](notes/lecture-12-multi-task-goal-conditioned-rl.md) | [HW4](solutions/spring2025/homework4-goal-meta-rl/README.md) |
| 元学习 | DREAM 用信息增益减去每步代价来学探索。RL² 把探索和执行放进同一个 RNN，信用分配要穿过整段历史。 | [L13](notes/lecture-13-meta-rl.md) | [HW4](solutions/spring2025/homework4-goal-meta-rl/README.md) |
| 系统 | 0.5B、AdamW、bf16：权重 1 GB，fp32 主权重 2 GB，两个矩 4 GB，梯度 1 GB，共 8 GB。训练进程留下的约 89 GB 是缓存，不是 KV。vLLM 按整卡容量申请，同卡起不来。 | [工程手册](interview/ENGINEERING.md) | [故事 2](interview/PORTFOLIO.md) |

2026 版没有 exploration。 [L14](notes/lecture-14-exploration.md) 按 2025 版补了 RND、ICM 和计数，对应上面的探索数据实验。一句话答案的全表在 [课程问答](interview/COURSE-QA.md)，每节末尾的错答在 [题库](interview/QUESTION-BANK.md)。

## 怎么准备

90 分钟只准备两个故事，再默写对应公式。算法岗用 [故事 4 和 5](interview/PORTFOLIO.md)（采集器是 oracle，以及 IQL 和 CQL 要的数据相反）。系统岗用 [故事 1 和 2](interview/PORTFOLIO.md)（停止符的两种 BPE，以及同卡 vLLM 起不来）。

一个周末按上表从上到下，每行先讲开口的那句，再打开笔记里的追问链。对方要看原始数字时，进 `solutions/` 里该作业的 README，曲线由 CSV 和 JSONL 生成。

全部测试：`./scripts/run-all-tests.ps1`。

## 材料放在哪

| 要找什么 | 打开 |
| --- | --- |
| 90 秒故事、公式、错答 | [`interview/`](interview/README.md) |
| 一讲的推导和反例 | [`notes/`](notes/README.md)，从 [L1](notes/lecture-01-intro-mdps.md) 到 [L18](notes/lecture-18-frontiers.md) |
| 多种子曲线和两份报告 | [`solutions/`](solutions/README.md) |
| 官方题面和 starter | [`assignments/`](assignments/spring2026/)，这里不改 starter |
| 幻灯片 | [2026](slides/spring2026/) 17 讲，[2025](slides/spring2025/) 19 讲，[录像](https://www.youtube.com/playlist?list=PLoROMvodv4rPwxE0ONYRa_itZFdaKCylL)，[2023](slides/spring2023/) |
| 论文和工具索引 | [`experiments/`](experiments/README.md) |

| 主题 | 题面 | 实现 |
| --- | --- | --- |
| 模仿：BC、flow matching、DAgger | [HW1](assignments/spring2026/homework1-imitation/) | [5 个种子](solutions/spring2026/homework1-imitation/README.md) |
| 在线：PPO、GAE、高 UTD | [HW2](assignments/spring2026/homework2-online-rl/) | [样本效率](solutions/spring2026/homework2-online-rl/README.md) |
| 离线：AWAC、IQL、拼接 | [HW3](assignments/spring2026/homework3-offline-rl/) | [IQL 高于最好轨迹 −46](solutions/spring2026/homework3-offline-rl/README.md) |
| 后训练：SFT、IPO、RLOO、GRPO | [Default Project](assignments/spring2026/default-project/) | [扩展评测](solutions/spring2026/default-project/README.md) |
| CQL 与探索数据 | [2025 HW3](assignments/spring2025/) | [oracle 采集器](solutions/spring2025/homework3-offline-rl/README.md) |
| HER、RL²、DREAM | [2025 HW4](assignments/spring2025/) | [0 到 1，0.70 对 0.14](solutions/spring2025/homework4-goal-meta-rl/README.md) |

2025 的两份作业补的是 2026 大纲里没有的 CQL、HER 和 meta-RL。上游 commit 在 [UPSTREAM.md](UPSTREAM.md)。本地可以直接跑的是 HW1 的 Flappy Bird；其余实验的命令写在对应 README 里。

## 许可

官方材料的版权归原作者，来源记在 [UPSTREAM.md](UPSTREAM.md)。第三方内容只在 [`resources/`](resources/official.md) 里链接。笔记、面试材料和脚本使用 [MIT License](LICENSE)。
