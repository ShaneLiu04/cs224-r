# 实验卡片：HW1 BC / Flow Matching / DAgger

> 状态：计划（待运行） · 代码：[starter-code/](../../assignments/spring2026/homework1-imitation/starter-code/)
> 归属：[PROGRESS.md](../../PROGRESS.md) · 笔记：[L2](../../notes/lecture-02-imitation-learning.md)

## 实验 1：BC 回归在 easy/hard 的对比

- 假设：hard 的多模态缺口让 MSE 回归显著退化；
- 命令：

```bash
python main.py --method bc_reg --env easy
python main.py --method bc_reg --env hard
```

- 指标：50 episodes 平均存活长度 ± 标准差；
- 记录：`bc_reg_easy.txt`、`bc_reg_hard.txt`；
- 失败模式：输出未 clamp、动作块 shape 错、评估 episode 数不对。

## 实验 2：Flow Matching vs BC（hard）

- 假设：生成式策略能覆盖多峰，hard 上优于 MSE；
- 命令：`python main.py --method bc_flow --env hard`；
- 变量：集成步数 `n`（速度/质量权衡）、τ 采样分布；
- 记录：`bc_flow_hard.txt`；
- 失败模式：插值方向写反、Euler 步长写错、未 clamp。

## 实验 3：DAgger 学习曲线

- 假设：每轮成功率上升，且超过 BC 基线；
- 命令：`python main.py --method dagger --env hard`（默认 5 轮）；
- 自变量：轮数、每轮采样量、确定性专家策略；
- 记录：各轮评估均值/标准差；
- 失败模式：未聚合数据、专家不确定性导致标签歧义。

## 结果模板

| 方法 | easy 平均长度 | hard 平均长度 | 备注 |
| --- | --- | --- | --- |
| BC 回归 |  |  |  |
| Flow matching | — |  |  |
| DAgger（final） | — |  |  |

结论/失败分析：

-
