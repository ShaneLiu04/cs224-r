# 主题：模仿学习与生成式策略

> 对应讲次：[L1](../../notes/lecture-01-intro-mdps.md)、[L2](../../notes/lecture-02-imitation-learning.md)；
> 作业：[HW1](../../assignments/spring2026/homework1-imitation/README.md)。

## 问题脉络

1. 有专家演示、没有奖励 → 监督学习；
2. 闭环执行导致分布偏移 → compounding error；
3. 多模态演示 + MSE → 均值坍缩；
4. 生成式策略（diffusion / flow matching）→ 建模条件分布；
5. 交互式纠错（DAgger / HG-DAgger）→ 修正分布。

## 必读论文

| 论文 | 年份 | 为什么读 |
| --- | --- | --- |
| Ross et al., *DAgger* | 2011 | 聚合式模仿学习与理论保证 |
| Chi et al., *Diffusion Policy* | 2024 | 机器人 diffusion 策略的完整设计 |
| Zhao et al., *ALOHA / ACT* | 2023 | action chunking 与低成本遥操作 |
| Lipman et al., *Flow Matching* | 2023 | FM 的理论与实现 |
| Ho et al., *DDPM* | 2020 | 扩散模型基础 |
| Florence et al., *Implicit BC* | 2022 | energy-based 策略的另一条路线 |

## 关键对比

| 方法 | 多模态 | 似然可算 | 采样成本 | 训练稳定性 |
| --- | --- | --- | --- | --- |
| Gaussian BC | 否 | 是 | 1 步 | 高 |
| MoG | 是（K 有限） | 是 | 1 步 | 中 |
| 离散自回归 | 是 | 是 | 多维逐步 | 中 |
| Diffusion | 是 | 否（近似） | 多步 | 高 |
| Flow matching | 是 | 否 | 多步（可少） | 高 |

## 常见实验

- 数据多模态程度对 BC/diffusion 的影响（构造两种演示风格）；
- 动作块长度 `T` 与执行长度 `K` 的扫描；
- DAgger 轮数、专家干预比例、聚合方式（全部 vs 最近 N 轮）；
- 生成步数 `n` 与成功率的权衡（flow/diffusion 的推理成本）。

## 开放问题

- 人类视频（embodiment gap）如何大规模利用？
- 演示 + 在线 RL 的最优混合（RLPD、Cal-QL）？
- 长时域任务的演示切分与层次化（见 [hierarchy](hierarchy-and-robotics.md)）？

## 本地材料

- slides：[02_cs224r_imitation_2026.pdf](../../slides/spring2026/02_cs224r_imitation_2026.pdf)
- 作业：[HW1 starter code](../../assignments/spring2026/homework1-imitation/starter-code/)
- 历史：2023/2025 HW1 的 MuJoCo BC+DAgger。
