# 实验卡片：HW3 AWAC / IQL / Stitching

> 状态：计划（待运行） · 代码：[starter-code/](../../assignments/spring2026/homework3-offline-rl/starter-code/)
> 笔记：[L7](../../notes/lecture-07-offline-rl.md)

## 实验 1：AWAC 两个环境

- 命令见 [HW3 指南](../../assignments/spring2026/homework3-offline-rl/README.md)；
- 指标：`Eval_AverageReturn` 末值，3 seeds 均值±标准差；
- 预期：umaze 较易，medium-diverse 明显更难；
- 失败模式：OOD 动作查询、advantage 权重爆炸、未归一化。

## 实验 2：IQL expectile 扫描

- 变量：`ζ = 0.2 / 0.5 / 0.9`（至少 0.2/0.9）；
- 指标：末值回报；在 umaze 选优后用于 medium 与 stitching；
- 假设：`ζ` 太低接近均值、太高被极值主导，`0.7–0.9` 常最优；
- 失败模式：expectile 方向写反、V 未 stop-gradient。

## 实验 3：PointMass stitching

- 数据：`offline_datasets/pointmass_stitching_dataset.npz`（max return −46，avg −104）；
- 对比：IQL（最优 ζ） vs Filtered BC（top 10%）；
- 指标：平均/最大回报（3 seeds）+ 轨迹可视化；
- 假设：IQL 超过 −46，Filtered BC 不超过；
- 记录：轨迹图 + 单条路径分析（是否拼接了不同片段）。

## 结果模板

| 数据集 | 算法 | 平均回报 ± std | 最大回报 ± std |
| --- | --- | --- | --- |
| antmaze-umaze | AWAC |  | — |
| antmaze-umaze | IQL (ζ*) |  | — |
| antmaze-medium-diverse | AWAC |  | — |
| antmaze-medium-diverse | IQL (ζ*) |  | — |
| PointMass stitching | IQL |  |  |
| PointMass stitching | Filtered BC |  |  |

结论/失败分析：

-
