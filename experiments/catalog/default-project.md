# 实验卡片：Default Project SFT / IPO / RLOO

> 状态：计划（待运行） · 代码：[starter-code/](../../assignments/spring2026/default-project/starter-code/)
> 笔记：[L9](../../notes/lecture-09-rlhf-preference-optimization.md)、[L10](../../notes/lecture-10-rl-llm-reasoning.md)

## 实验 1：SFT 基线

- 命令：`bash sft_trainer/train_sft.sh`（或 modal 版）；
- 变量：学习率、batch、epochs、仅 response loss（必须）；
- 指标：SFT 后的 pass@1/pass@k、格式正确率、平均长度；
- 失败模式：把 prompt 计入 loss、OOM、学习率过大退化。

## 实验 2：IPO vs SFT

- 变量：`β`、偏好对构造（正确/错误答案）、训练轮数；
- 指标：held-out 偏好对准确率、pass@k、长度变化；
- 假设：IPO 提升 pass@k 但过度训练会降低多样性；
- 失败模式：参考模型未冻结、`Δ` 符号反、`β` 极端。

## 实验 3：RLOO 与课程

- 变量：`group_size`、温度、熵系数、KL 系数、课程（难度/长度）；
- 指标：pass@1/pass@k、响应长度、格式奖励、训练稳定性；
- 假设：组内奖励有方差时才有梯度；课程能显著提升难题表现；
- 失败模式：全对/全错组、长度爆炸、KL 过强无学习。

## 结果模板

| 阶段 | pass@1 | pass@k (k=?) | 平均长度 | 格式分 | 备注 |
| --- | --- | --- | --- | --- | --- |
| SFT |  |  |  |  |  |
| IPO |  |  |  |  |  |
| RLOO |  |  |  |  |  |
| RLOO + 扩展 |  |  |  |  |  |

扩展实验设计（选一）：

- 难度课程调度：
- 部分解答探索：
- 工具/验证器：

结论/失败分析：

-
