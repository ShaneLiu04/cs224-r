# 官方题目索引：HW1 Imitation Learning

- 题面：[CS224R_2026_Homework_1.pdf](../../assignments/spring2026/homework1-imitation/CS224R_2026_Homework_1.pdf)
- 模板：[CS224R_2026_Homework_1.tex](../../assignments/spring2026/homework1-imitation/CS224R_2026_Homework_1.tex)
- 代码：[starter-code/](../../assignments/spring2026/homework1-imitation/starter-code/)
- 指南：[学习指南](../../assignments/spring2026/homework1-imitation/README.md)

## 要求拆解

| 题 | 要求 | 分值 | 交付 |
| --- | --- | --- | --- |
| P1 | 实现 BCPolicy（3 层 MLP + sigmoid）与 MSE loss；easy/hard 各训练一次 | 2 | 结果表 + 2–3 句解释 |
| P2 | 实现 flow matching schedule（interpolate/sample）与 FM loss；hard 训练 | 2 | 结果表 + 2–3 句解释 |
| P3 | 实现 DeterministicExpert、rollout_episode、rollout_and_relabel；DAgger 5 轮 | 2 | 学习曲线 + 三方法对比 + 解释 |

## 评估与提交

- 指标：50 个评估 episode 的 episode 长度均值 ± 标准差；
- 代码提交：`bc_reg_easy.txt`、`bc_reg_hard.txt`、`bc_flow_hard.txt`、`dagger_hard.txt`；
- 代码文件：`networks.py`、`losses.py`、`expert.py`、`dagger.py`（保持原格式，填 TODO）。

## 隐藏的思考题

1. hard mode 为什么让 MSE 回归退化？（多峰 → 均值坍缩）
2. flow matching 如何缓解多峰？（生成式建模条件分布）
3. DAgger 的确定性专家为什么必要？（消除专家随机性带来的标签歧义）
4. 为什么 DAgger 能打破 compounding error？（访问分布 + 专家纠正）

## 关联论文

- Ross et al., *DAgger*, 2011 → [topics/imitation-learning.md](../topics/imitation-learning.md)
- Chi et al., *Diffusion Policy*, 2024；Lipman et al., *Flow Matching*, 2023
- Zhao et al., *ALOHA/ACT*（action chunking）, 2023
