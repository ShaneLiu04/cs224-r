# 实验卡片：HW2 Q-learning / PPO / Off-policy AC

> 状态：计划（待运行） · 代码：[starter-code/](../../assignments/spring2026/homework2-online-rl/starter-code/)
> 笔记：[L4](../../notes/lecture-04-actor-critic.md)、[L5](../../notes/lecture-05-off-policy-actor-critic.md)、[L6](../../notes/lecture-06-q-learning.md)

## 实验 1：Gridworld 奖励设计（P1）

- 变量：`(r_step, R1, R2) = (−1,10,5) / (−2,10,5) / (+1,1,1)`；
- 指标：选择的目标、路径、总回报；
- 记录：三条轨迹图 + 手算折扣回报；
- 讨论：为什么 `r_step` 的正负改变「尽快结束 vs 拖时间」的激励。

## 实验 2：PPO on hammer（P2）

- 命令：`modal run modal_on_policy.py`；
- 指标：`eval/episode_success`（至 1M steps），目标 ≥25%；
- 自变量（可选）：`clip_eps`、GAE `λ`、epochs、entropy coef、BC 预训练步数；
- 失败模式：`π_old` 未冻结、GAE 未逆序、ratio 未用 log 空间。

## 实验 3：Off-policy AC + UTD（P3）

- 命令：`modal run modal_off_policy.py`；
- 变量：`(num_critics, utd) = (2,1) / (10,5)`；
- 指标：至 100k / 40k 的 success（目标 ≥90%）；
- 假设：更高 UTD 加速收敛但更易不稳定；
- 失败模式：只更新被选中的两个 target critic、target 未 detach、BC 预训练缺失。

## 结果模板

| 实验 | 种子 | 关键指标 | 时间 | 备注 |
| --- | --- | --- | --- | --- |
| Gridworld 场景 1/2/3 | — | 轨迹/回报 | — |  |
| PPO (2 critics 不适用) |  | success@1M |  |  |
| AC (2,1) |  | success@100k |  |  |
| AC (10,5) |  | success@40k |  |  |

结论/失败分析：

-
