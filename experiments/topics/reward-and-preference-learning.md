# 主题：奖励与偏好学习

> 对应讲次：[L8](../../notes/lecture-08-reward-learning.md)、[L9](../../notes/lecture-09-rlhf-preference-optimization.md)；
> 作业：项目 [SFT/IPO/RLOO](../../assignments/spring2026/default-project/README.md)。

## 问题脉络

1. 奖励难写 → 从演示/目标/偏好中学习；
2. 目标分类器被 exploit → 负例挖掘、约束、对抗训练；
3. 人类比较 → Bradley-Terry 奖励模型；
4. RLHF：偏好 → RM → PPO + KL；
5. DPO：闭式隐式奖励，跳过 RM；
6. 可验证奖励（RLVR）：规则打分，推理任务主流。

## 必读论文

| 论文 | 年份 | 为什么读 |
| --- | --- | --- |
| Christiano et al., *Deep RL from Human Preferences* | 2017 | 偏好学习开山 |
| Ng et al., *IRL* | 2000 | 逆向 RL 基础 |
| Fu et al., *VICE* | 2018 | 目标分类器与负例挖掘 |
| Ho & Ermon, *GAIL* | 2016 | 对抗式模仿与奖励 |
| Ouyang et al., *InstructGPT* | 2022 | RLHF 工业级管线 |
| Rafailov et al., *DPO* | 2023 | 隐式奖励推导 |
| Azar et al., *IPO* | 2024 | DPO 的确定性偏好修正 |
| Bai et al., *Constitutional AI / RLAIF* | 2022 | AI 反馈与自我批评 |
| Gao et al., *Scaling Laws for Reward Model Overoptimization* | 2023 | 过优化实证 |

## 常见实验

- 成对偏好数据的标注噪声对 RM 的影响；
- RM 准确率 vs 代理奖励 vs 人工评估（过优化曲线）；
- DPO `β` 扫描与对参考模型的 KL；
- IPO vs DPO 在重复偏好对上的稳定性；
- 可验证奖励 + 格式奖励的组合（项目）。

## 开放问题

- 偏好不可靠（标注噪声、长度偏差、谄媚）如何建模？
- 过程奖励 vs 结果奖励的最优组合？
- 多目标/多元价值下的对齐（pluralistic alignment）？
- 奖励模型与策略的共演化（co-evolution）？

## 本地材料

- slides 08–09；项目 `ipo_trainer/`、`evaluation/countdown.py`；
- [catalog/default-project.md](../catalog/default-project.md)。
