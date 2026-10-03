# 官方题目索引：HW3 Offline RL

- 题面：[CS224R_2026_Homework_3.pdf](../../assignments/spring2026/homework3-offline-rl/CS224R_2026_Homework_3.pdf)
- 代码：[starter-code/](../../assignments/spring2026/homework3-offline-rl/starter-code/)
- 指南：[学习指南](../../assignments/spring2026/homework3-offline-rl/README.md)
- 数据：AntMaze 自动下载（`~/.d4rl`）；PointMass stitching 数据在 `offline_datasets/`。

## 要求拆解

| 题 | 内容 | 分值 | 验收 |
| --- | --- | --- | --- |
| P1 | AWAC：策略、critic、agent；antmaze-umaze + antmaze-medium-diverse | 2 | 3-seed 均值±标准差表 |
| P2 | IQL：expectile 扫描（0.2/0.9）；与 AWAC 对比 | 5 | 表格 + 3 句解释 |
| P3 | PointMass stitching：IQL vs Filtered BC | 1 | 回报表 + 轨迹图 + 分析 |

## Autograder CSV

```text
P1/1/awac_umaze_seed1..3.csv        P1/2/awac_medium_maze_seed1..3.csv
P2/1/iql_umaze_seed1..3.csv         P2/2/iql_medium_maze_seed1..3.csv
```

来源必须是 W&B `Eval_AverageReturn` 图，逐 seed 导出，不合并。

## 关键实现点

- AWAC critic：`y = r + γ E_{a'~π}[min_i Q̄_i(s',a')]`；
- AWAC actor：`−logπ(a|s)·exp(A/λ)`（advantage detach、权重截断）；
- IQL `V`：expectile 回归 `|ζ−1{µ≤0}|µ²`；
- IQL `Q`：`(r+γV(s')−Q(s,a))²`，只用数据动作；
- IQL policy：`exp(β(Q̄−V))` 加权回归。

## 隐藏的思考题

1. IQL 的 `V` 与 `max` 的关系？为什么 `ζ=0.9` 常优于 0.5/0.2？
2. AWAC 的目标动作来自当前策略，为什么在 offline 下更危险？
3. 什么数据分布能体现 stitching？为什么 Filtered BC 做不到？
