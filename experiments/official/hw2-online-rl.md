# 官方题目索引：HW2 Online RL

- 题面：[CS224R_2026_Homework_2.pdf](../../assignments/spring2026/homework2-online-rl/CS224R_2026_Homework_2.pdf)
- 代码：[starter-code/](../../assignments/spring2026/homework2-online-rl/starter-code/)
- 指南：[学习指南](../../assignments/spring2026/homework2-online-rl/README.md)

## 要求拆解

| 题 | 内容 | 分值 | 验收 |
| --- | --- | --- | --- |
| P1 | 5×4 gridworld 表格 Q-learning；三种奖励场景 | 3 | 轨迹 + 总回报 + 解释 |
| P2 | Meta-World hammer 上的 PPO（TruncatedNormal actor + GAE + clip + 参考策略 KL） | 3 | 1M steps 内 success ≥25% |
| P3 | off-policy AC（critic ensemble + double-Q + target net + BC 预训练），UTD 消融 | 4 | 100k 内 ≥90%；`10 critics/UTD=5` 40k 内 ≥90% |

## 关键实现点

- GAE：逆序递推，`returns = advantages + values`，`done` 截断；
- PPO：`ρ=exp(logπ_new−logπ_old)`，clip `[1−ε,1+ε]`，取 min；
- Off-policy：Bellman target `r+γ min(Q̄_i,Q̄_j)`（两个随机 target critic）；
- Actor：`−mean_i Q_i(s, π(s))`；BC 预训练提供稀疏奖励下的先验；
- UTD 提升 = 更多 critic 更新，需要配 target 更新与正则。

## 交付

- `gridworld_q_learning.py`、`on_policy.py`、`off_policy.py`；
- 三份 W&B CSV（`eval/episode_success`）：on_policy、off_policy(2 critics/UTD=1)、
  off_policy(10 critics/UTD=5）；
- PDF 报告（曲线截图 + 分析）。

## 隐藏的思考题

1. 为什么 PPO 在同一批数据上多轮更新仍然稳定？（clip + 小步长）
2. 为什么稀疏奖励下需要 BC 预训练？（随机探索几乎见不到成功）
3. critic ensemble 与 double-Q 分别解决什么？（误差与过估计）
4. UTD 提高为什么会加速但也更容易不稳定？（样本复用 vs bootstrap 误差）
