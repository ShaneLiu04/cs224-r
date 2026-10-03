# Lecture 10 — RL for LLMs: Reasoning

- 课程版本：Spring 2026（2025 有录像；2026 主讲 Noam Brown）
- 日期：2026-05-01
- 官方材料：[10_cs224r_rl_for_llms_reasoning_2026.pdf](../slides/spring2026/10_cs224r_rl_for_llms_reasoning_2026.pdf) ·
  [2025 版](../slides/spring2025/10_cs224r-rl_for_reasoning_lecture.pdf) ·
  [Lecture 10 视频](https://www.youtube.com/watch?v=O2VpNnwB4lM) ·
  论文：GRPO (DeepSeekMath)、DeepSeek-R1
- 状态：初稿

## 学习目标

- 解释「推理是新的 scaling 维度」：训练算力之外，测试时算力同样可扩展；
- 写出 CoT、self-consistency、verifier 重排序的流程与优劣；
- 推导 GRPO 与 RLOO 的组内优势，说明为什么可以去掉 critic；
- 描述 R1-Zero 的现象（准确率与回答长度同增）及其启示；
- 理解评估范式转变：`score vs compute`、pass@k、长程安全评估。

## 先修知识

- L3–L5 策略梯度/PPO；L9 偏好优化；语言模型采样与对数概率。

## 核心概念

### 1. 两代 scaling

- 训练时 scaling：更多参数、更多数据、更多算力（预训练 scaling law）；
- **测试时 scaling**：给模型更多思考时间——更长的 CoT、更多采样、搜索、验证。

Noam Brown 用扑克与围棋的历史说明：搜索/推理算力可以在**不增加模型**的情况下提升决策质量
（Libratus、AlphaGo 都是实例）。LLM 的 CoT 是把「潜空间搜索」显式化为 token 序列。

### 2. 推理增强的基本手段

| 方法 | 做法 | 代价 |
| --- | --- | --- |
| CoT prompting | 让模型先写推理步骤再给答案 | 采样成本、可能编造推理 |
| Self-consistency | 采样多条 CoT，取多数答案 | K 倍采样 |
| Verifier / 重排序 | 用验证器对候选答案打分，选最优 | 需要验证器或答案可验证 |
| 搜索（树/束） | 对推理步骤做搜索 | 实现复杂、奖励设计难 |

在数学、代码这类**答案可验证**的任务上，多数投票与验证器效果显著。

### 3. 可验证奖励（RLVR）

奖励来自规则或程序：答案是否正确、代码是否通过测试、格式是否合法。
Countdown（默认项目）就是最简例子：

```text
0.0  无 <answer> 标签
0.1  格式合法但等式错误
1.0  用给定数字构成正确等式
```

可验证奖励避免了 L8 的偏好噪声，是当前推理 RL 的主流。

### 4. GRPO（Group Relative Policy Optimization）

PPO 需要 critic 估计 baseline，成本高。GRPO 对同一 prompt 采样一组 `K` 个回答，
用组内统计构造优势：

```text
Â_i = (r_i − mean(r_1…r_K)) / std(r_1…r_K)
```

然后套用 PPO 的 clip 目标（可选加上与参考模型的 KL 惩罚）：

```text
L_GRPO = −(1/K) Σ_i min( ρ_i Â_i, clip(ρ_i,1±ε) Â_i ) + β KL(π_θ || π_ref)
```

- 不需要 value network，显存与实现都简化；
- 组内对比天然去除 prompt 难度差异（同一 prompt 内比较）；
- 缺点：所有回答奖励相同（如全对/全错）时优势为零，没有学习信号，需要课程/难度控制。

### 5. RLOO（REINFORCE Leave-One-Out）

另一种无 critic 的组内基线：

```text
Â_i = r_i − (1/(K−1)) Σ_{j≠i} r_j
```

与 GRPO 的区别：RLOO 用 leave-one-out 均值（无偏），GRPO 用全组均值/标准差归一化
（有偏但方差更小）。两者都只需采样 + 奖励，工程上都比 PPO 轻。

### 6. R1-Zero 现象与课程学习

DeepSeek-R1-Zero：纯 RL（无 SFT 冷启动）在数学任务上，**准确率与回答长度同时上升** ——
模型学会用更长的推理链换正确率。启示：

- 推理行为可以被 RL 激发，而非必须示范；
- 长度增长带来成本与上限问题，需要长度惩罚/预算控制；
- 奖励设计（格式奖励、答案奖励、语言一致性）影响行为。

### 7. 评估范式的变化

- 不再只看单次准确率，而是 `score vs compute/time` 曲线；
- `pass@k`：采样 k 次至少一次成功的比例，衡量探索上限；
- 安全评估也要考虑 test-time compute：能力随推理预算增长，短评估会低估风险；
- 推理算力逐渐与权重同等重要（inference capacity 被低估）。

### 8. 多智能体与长程

CoT 是串行的；多智能体/多轮交互可以并行探索与分工。长程任务（数小时到数天）的
信用分配、记忆、验证与安全仍是开放问题（配合 L18）。

## 关键推导：pass@k 的无偏估计

从 `n` 个样本中随机取 `k` 个，至少一个正确的概率：

```text
pass@k = 1 − C(n−c, k) / C(n, k)
```

其中 `c` 是正确的样本数。代码评估常用 `n=200, k=1,10,100`。
它衡量「策略覆盖多少正确路径」，是推理 RL 的关键指标之一。

## 实现映射（Default Project）

| 概念 | 文件 | 说明 |
| --- | --- | --- |
| 可验证奖励 | `starter-code/evaluation/countdown.py` | 0.0 / 0.1 / 1.0 三档 |
| RLOO 采样 | `rloo_trainer/rloo.py` + `sampling_worker.py` | Ray + vLLM 采样 `group_size` 个回答 |
| LOO 优势 | `rloo_trainer/rloo_update_worker.py` | `update(...)` 中实现 |
| pass@k | `evaluation/view_passk.ipynb` | 对比 SFT / IPO / RLOO |
| 熵/KL 正则 | `rloo_update_worker.py` | 稳定训练 |

## 易错点与反例

- **组内奖励全同**：优势为 0，无梯度；需要混合难度或过滤；
- **长度偏差**：长回答累积更多 log-prob 项，KL/熵项需要按 token 平均；
- **只优化答案奖励**：可能得到「乱写过程、碰对答案」，应加格式/过程奖励；
- **pass@k 与 greedy 准确率混淆**：前者衡量多样性上限，后者衡量单次质量；
- **忽略推理成本**：报告结果时需同时给 token/时延预算。

## 实验或练习

1. 在 Countdown 上对比 SFT / IPO / RLOO 的 pass@1、pass@k 与平均长度；
2. 实现 GRPO 与 RLOO 的优势（几行代码），用同一批组样本比较数值；
3. 设计一个「全对/全错时无优势」的场景，提出两种课程策略缓解。

## 高频问题

- 问：GRPO 与 PPO 的核心差别？答：省略 critic，用组内相对奖励做优势。
- 问：RLOO 与 GRPO 的基线差异？答：LOO 均值（无偏） vs 组均值/方差归一化（低方差）。
- 问：为什么 CoT 能提升准确率？答：把计算展开成中间 token，等效于增加串行推理算力。
- 问：可验证奖励的优势？答：无偏好噪声、抗 reward hacking（相对 RM），适合数学/代码。
- 问：test-time scaling 对评估的挑战？答：需要按算力预算报告，否则不可比。

## 本讲小结

推理 RL 把 RL 从「连续控制」带到「语言 token」：目标可验证、组内比较、测试时算力可扩展。
GRPO/RLOO 是当前主力算法。下一讲回到经典问题：能不能学一个环境模型来提升样本效率？

## 延伸阅读

- Shao et al., *DeepSeekMath* (GRPO), 2024；DeepSeek-AI, *DeepSeek-R1*, 2025.
- Wei et al., *Chain-of-Thought Prompting*, 2022；Wang et al., *Self-Consistency*, 2022.
- Brown & Sandholm, *Superhuman AI for multiplayer poker*, Science 2019.
- 官方 slides 与默认项目 README。
