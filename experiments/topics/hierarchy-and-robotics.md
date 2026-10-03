# 主题：层次化、Sim-to-Real 与 VLA

> 对应讲次：[L15](../../notes/lecture-15-hierarchical-rl-il.md)、
> [L16](../../notes/lecture-16-sim-to-real.md)、[L17](../../notes/lecture-17-rl-for-vlas.md)。

## 问题脉络

1. 长时域任务 → 时间抽象（options、子目标）；
2. 子目标表示（语言、图像、位姿）与切换设计；
3. 仿真训练 → 真机部署（sim2real gap）；
4. 领域随机化、系统辨识、特权信息蒸馏；
5. 预训练 VLA + RL 后训练（离线迭代 / 轻量在线）。

## 必读论文

| 论文 | 年份 | 为什么读 |
| --- | --- | --- |
| Sutton et al., *Options* | 1999 | 层次化理论 |
| Eysenbach et al., *DIAYN* | 2018 | 技能发现 |
| Lynch et al., *HULC/LCBC* | 2020 | 语言条件层次 IL |
| Ahn et al., *SayCan* | 2022 | LLM 规划 + 可行性 |
| Tobin et al., *Domain Randomization* | 2017 | sim2real 基础 |
| Kumar et al., *RMA* | 2021 | 特权信息蒸馏 |
| Hwangbo et al., *Actuator Net* | 2019 | 电机建模 |
| Kim et al., *OpenVLA* | 2024 | 开源 VLA |
| Physical Intelligence, *π0.5* | 2025 | VLA + RL 后训练 |
| NVIDIA, *GR00T N1* | 2025 | 人形基础模型 |

## 关键设计表

| 设计点 | 选项 | 权衡 |
| --- | --- | --- |
| 子目标表示 | 语言 / 图像 / 状态 / 隐向量 | 语义泛化 vs 精确控制 |
| 层间监督 | 演示切分 / 规划器 / RL / HER | 数据成本 vs 最优性 |
| 切换 | 固定步数 / 分类器 / 学习终止 | 简单 vs 自适应 |
| sim2real | DR / system ID / residual / 蒸馏 | 鲁棒 vs 精确 |
| VLA RL | 全参 PPO / 离线条件 / 残差编辑 | 成本 vs 上限 |

## 常见实验

- 两层导航：高层选点 + 低层 HER；
- DR 参数宽度与真机/扰动鲁棒性；
- RMA 蒸馏后的性能损失；
- 残差策略 vs 全参微调的样本效率；
- 语言子目标 vs 图像子目标的泛化。

## 开放问题

- 长时域真机自主性与安全；
- 人形机器人的通用运动与操作统一；
- VLA 的 RL 后训练成本与稳定性；
- 世界模型 + VLA 的结合。

## 本地材料

- slides 15–17；
- 项目索引中的机器人方向优秀报告（`materials/spring2026/projects-index.html`）。
