# CS229-R

![面向面试的深度强化学习](assets/cs229r-cover.png)

面向面试的深度强化学习。材料来自 Stanford CS224R（Chelsea Finn，Spring 2026 / 2025 / 2023），
整理成可以当场推导、比较算法、并打开实验把数字讲清楚的个人仓库。

作者 [ShaneLiu04](https://github.com/ShaneLiu04)。与 Stanford University 及 CS224R 课程团队无隶属关系。
官方材料版权归原作者。正在修读该课的同学请遵守 honor code，不要把 `solutions/` 当作作业提交。

## 今晚怎么用

| 你有多少时间 | 做什么 | 打开 |
| --- | --- | --- |
| 90 分钟 | 只准备两个故事，默写对应公式 | 算法岗：[故事 4 和 5](interview/PORTFOLIO.md)；系统岗：[故事 1 和 2](interview/PORTFOLIO.md)。公式卡：[FORMULAS](interview/FORMULAS.md) |
| 一个周末 | 按下面的地图，每模块讲透一题 | 笔记里的「一页速览卡」和「面试追问链」 |
| 对方要看证据 | 打开曲线和多种子表 | 下表的作业 README |

全部测试：`./scripts/run-all-tests.ps1`。

四层深度，不要在一层里找全部内容：

| 层 | 回答什么 | 位置 |
| --- | --- | --- |
| 入口 | 今晚讲哪两题 | 本页 |
| 面试稿 | 90 秒故事、可手推公式、常见错答 | [`interview/`](interview/README.md) |
| 推导 | 一讲的定义、证明、反例 | [`notes/`](notes/README.md) |
| 证据 | 多种子曲线和报告 | [`solutions/`](solutions/README.md) |

## 面试地图

每一行是一道必须能讲完的题。一句话答案在 [课程问答索引](interview/COURSE-QA.md)，实验在作品集。

| 模块 | 讲完这一题 | 笔记 | 证据 |
| --- | --- | --- | --- |
| 模仿学习 | BC 的复合误差从哪来，多峰动作为什么不能用 MSE | [L2](notes/lecture-02-imitation-learning.md) | [HW1](solutions/spring2026/homework1-imitation/README.md) |
| 策略梯度 | baseline 为什么无偏，过去的奖励为什么能丢掉 | [L3](notes/lecture-03-policy-gradients.md) | [公式卡](interview/FORMULAS.md) |
| Actor-Critic | GAE 的 `λ` 在偏差和方差之间换什么 | [L4](notes/lecture-04-actor-critic.md) | [HW2](solutions/spring2026/homework2-online-rl/README.md) |
| PPO / SAC | clip 限制的是哪一个比值；double-Q 为什么取 min | [L5](notes/lecture-05-off-policy-actor-critic.md) | [HW2](solutions/spring2026/homework2-online-rl/README.md) |
| 值方法 | `max` 为什么造成过估计 | [L6](notes/lecture-06-q-learning.md) | [HW2 Gridworld](solutions/spring2026/homework2-online-rl/README.md) |
| 离线 RL | IQL 为什么不采样下一个动作；CQL 的 α 在压什么 | [L7](notes/lecture-07-offline-rl.md) | [2026 HW3](solutions/spring2026/homework3-offline-rl/README.md)、[2025 HW3](solutions/spring2025/homework3-offline-rl/README.md) |
| 偏好优化 | DPO 的隐式奖励怎么从 Bradley-Terry 推出来 | [L9](notes/lecture-09-rlhf-preference-optimization.md) | [Countdown](solutions/spring2026/default-project/README.md) |
| 推理 RL | RLOO、GRPO、Dr. GRPO 差在归一化的哪一项 | [L10](notes/lecture-10-rl-llm-reasoning.md) | [故事 1](interview/PORTFOLIO.md)、[题库 Q13](interview/QUESTION-BANK.md) |
| 目标条件 | HER 重标记了什么，什么时候有偏 | [L12](notes/lecture-12-multi-task-goal-conditioned-rl.md) | [2025 HW4](solutions/spring2025/homework4-goal-meta-rl/README.md) |
| 元学习 | RL² 和 DREAM 把探索放在哪里学 | [L13](notes/lecture-13-meta-rl.md) | [2025 HW4](solutions/spring2025/homework4-goal-meta-rl/README.md) |
| 系统 | 0.5B 的 8 GB 静态显存怎么乘出来；同卡 vLLM 为什么起不来 | [工程手册](interview/ENGINEERING.md) | [故事 2](interview/PORTFOLIO.md) |

2026 版课程没有 exploration 这一讲。L14 按 2025 版整理，补的是 RND、ICM 和计数。
[L14](notes/lecture-14-exploration.md) 对应 2025 HW3 的探索数据实验。

## 实验里能讲的数字

数字都由原始 CSV / JSONL 生成。细节、种子数和复现命令在各自的 README。

| 实验 | 可以讲的结果 |
| --- | --- |
| [2026 HW1](solutions/spring2026/homework1-imitation/README.md) | BC、flow matching、DAgger，5 个种子，含 Euler 步数和重规划消融 |
| [2026 HW2](solutions/spring2026/homework2-online-rl/README.md) | PPO 与高 UTD actor-critic 的样本效率，GAE `λ` 消融，Gridworld 上的 reward hacking |
| [2026 HW3](solutions/spring2026/homework3-offline-rl/README.md) | IQL 在拼接任务上的回报高于数据集最好轨迹 −46；Filtered BC 低于 −46 |
| [Default Project](solutions/spring2026/default-project/README.md) | 短 SFT 的 pass@1 为 0.188，参考策略处 IPO 损失为 25。长 think SFT 留出 pass@1 为 0.272；长程 RLOO / Dr. GRPO 把它抬到 0.52–0.54，pass@16 仍与 SFT 同量级 |
| [2025 HW3](solutions/spring2025/homework3-offline-rl/README.md) | starter 的 CQL 采集器是最短路贪心。换成真实探索数据后，IQL 在 RND 数据上更好（−50.9 对 −90.7），CQL 在随机数据上更好（α=0.1 时 −34.3±6.1） |
| [2025 HW4](solutions/spring2025/homework4-goal-meta-rl/README.md) | 无 HER 时 25-bit 成功率为 0，三种重标记都到 1.0。DREAM 0.696±0.007，RL² 0.143±0.073，最优为 0.7 |

CQL 的 α 在两种数据上方向相反。这是离线 RL 那一题的图。

![CQL 数据来源](solutions/spring2025/homework3-offline-rl/artifacts/figures/hw3_fig8_cql_data_source.png)

25-bit 上，没有重标记的曲线停在 0。

![HER bit-flip](solutions/spring2025/homework4-goal-meta-rl/artifacts/figures/hw4_fig1_bitflip_success.png)

Countdown 上，短程偏好优化和 RLOO 的差距小于种子之间的差距。

![Countdown pass@k](solutions/spring2026/default-project/artifacts/figures/fig7_passk.png)

## 仓库里各放什么

```text
interview/          故事、公式卡、错答、课程问答索引
notes/              18 讲推导，每讲有速览卡和追问链
solutions/          个人实现、多种子原始指标、图、两份报告
assignments/        官方题面和 starter，不在这里改
slides/ materials/  2026、2025、2023 官方幻灯片和课程材料
experiments/        论文与工具索引
scripts/            材料检查与 run-all-tests.ps1
```

官方幻灯片：2026 版 17 讲（[目录](slides/spring2026/)），2025 版 19 讲并有
[录像](https://www.youtube.com/playlist?list=PLoROMvodv4rPwxE0ONYRa_itZFdaKCylL)，
2023 版 19 讲。上游 commit 记在 [UPSTREAM.md](UPSTREAM.md)。

作业入口按「题面」和「个人实现」分开：

| 主题 | 官方题面 | 个人实现 |
| --- | --- | --- |
| 模仿学习 | [2026 HW1](assignments/spring2026/homework1-imitation/) | [solutions](solutions/spring2026/homework1-imitation/README.md) |
| 在线 RL | [2026 HW2](assignments/spring2026/homework2-online-rl/) | [solutions](solutions/spring2026/homework2-online-rl/README.md) |
| 离线 RL | [2026 HW3](assignments/spring2026/homework3-offline-rl/) | [solutions](solutions/spring2026/homework3-offline-rl/README.md) |
| LLM 后训练 | [Default Project](assignments/spring2026/default-project/) | [solutions](solutions/spring2026/default-project/README.md) |
| CQL 与探索 | [2025 HW3](assignments/spring2025/) | [solutions](solutions/spring2025/homework3-offline-rl/README.md) |
| HER 与 meta-RL | [2025 HW4](assignments/spring2025/) | [solutions](solutions/spring2025/homework4-goal-meta-rl/README.md) |

2025 的 HW3、HW4 是补 2026 大纲里没有的 CQL、HER 和 meta-RL。

## 沿课程顺序复习

| 阶段 | 看什么 | 做完能讲什么 |
| --- | --- | --- |
| 模仿 | [L1](notes/lecture-01-intro-mdps.md) [L2](notes/lecture-02-imitation-learning.md) | 复合误差、flow matching、DAgger |
| 在线 | [L3](notes/lecture-03-policy-gradients.md) 到 [L6](notes/lecture-06-q-learning.md) | 策略梯度、GAE、PPO、Double Q |
| 离线与奖励 | [L7](notes/lecture-07-offline-rl.md) 到 [L9](notes/lecture-09-rlhf-preference-optimization.md) | IQL、CQL、Bradley-Terry、DPO |
| 推理 | [L10](notes/lecture-10-rl-llm-reasoning.md) | RLOO、pass@k、test-time compute |
| 模型、目标、元学习 | [L11](notes/lecture-11-model-based-rl.md) 到 [L14](notes/lecture-14-exploration.md) | 短 rollout、HER、RL²、RND |
| 机器人 | [L15](notes/lecture-15-hierarchical-rl-il.md) 到 [L18](notes/lecture-18-frontiers.md) | 层次、sim-to-real、VLA |

本地能直接跑的是 HW1 的 Flappy Bird。HW2、HW3 和 Default Project 的完整实验在 GPU 上跑，
命令写在对应的 solution README。官方 starter 保持不动。

## 引用与许可

官方材料的来源和 commit 在 [UPSTREAM.md](UPSTREAM.md)。第三方内容只在 `resources/` 里链接。
笔记末尾标了所用幻灯片和论文。

原创笔记、面试材料和脚本使用 [MIT License](LICENSE)。官方材料沿用其各自版权。
本仓库用于个人学习和求职准备。
