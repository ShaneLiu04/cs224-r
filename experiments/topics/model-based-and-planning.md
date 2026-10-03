# 主题：模型类 RL 与规划

> 对应讲次：[L11](../../notes/lecture-11-model-based-rl.md)；
> 历史作业：2023/2025 HW2 的 MJPC 部分。

## 问题脉络

1. 用监督学习学动力学模型；
2. 模型数据增强（Dyna）或纯规划；
3. 模型误差 → 分布偏移 → exploit；
4. 短 rollout、集成、不确定性惩罚、保守约束；
5. 规划算法：随机打靶、CEM、MPPI、树搜索；
6. 最新：latent world models、video world models。

## 必读论文

| 论文 | 年份 | 为什么读 |
| --- | --- | --- |
| Sutton, *Dyna* | 1991 | 模型数据增强雏形 |
| Nagabandi et al., *Neural Network Dynamics* | 2018 | 模型 + MPC |
| Chua et al., *PETS* | 2018 | 集成 + 不确定性规划 |
| Janner et al., *MBPO* | 2019 | 短 rollout 的经典分析 |
| Yu et al., *MOPO* | 2020 | 不确定性惩罚 |
| Hafner et al., *Dreamer(V3)* | 2019–2023 | latent world model |
| Schrittwieser et al., *MuZero* | 2020 | 学模型 + 搜索 |
| Williams et al., *MPPI* | 2017 | 采样控制 |

## 公式备忘

```text
模型学习:  L = −E log p_θ(s'|s,a)
Bellman with model: Q(s,a) = r + γ E_{s'~p_θ}[V(s')]
不确定性惩罚: r̃ = r − λ·u(s,a)
```

## 常见实验

- 模型 rollout 长度 `k` 对回报的影响；
- ensemble 大小与不确定性惩罚系数；
- 模型数据 vs 真实数据比例；
- CEM/MPPI 在 MuJoCo 任务上的规划质量；
- 模型误差可视化（预测 vs 真实）。

## 开放问题

- 高维像素模型的训练成本与稳定性；
- 模型与无模型方法的统一（hybrid）；
- 世界模型用于 LLM/机器人决策；
- 模型误差的在线检测与恢复。

## 本地材料

- slides 11；
- MJPC 源码与 videos：`assignments/spring2025/homework2-online-rl/starter-code/mujoco_mpc/`；
- 2023 HW2 同款源码。
