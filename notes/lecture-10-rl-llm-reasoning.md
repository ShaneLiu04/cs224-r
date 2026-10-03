# Lecture 10 — RL for LLMs: Reasoning

- 课程版本：Spring 2026（2025 有录像；2026 主讲 Noam Brown）
- 日期：2026-05-01
- 官方材料：[10_cs224r_rl_for_llms_reasoning_2026.pdf](../slides/spring2026/10_cs224r_rl_for_llms_reasoning_2026.pdf) ·
  [2025 版](../slides/spring2025/10_cs224r-rl_for_reasoning_lecture.pdf) ·
  [Lecture 10 视频](https://www.youtube.com/watch?v=O2VpNnwB4lM) ·
  论文：GRPO (DeepSeekMath)、DeepSeek-R1
- 状态：初稿

## 摘要

推理是继训练算力之后的新 scaling 维度：长 CoT、多采样、验证与搜索都可在推理时加算力。
本讲介绍可验证奖励（RLVR）与两种无 critic 的组内策略优化：GRPO（组归一化优势）与
RLOO（leave-one-out 基线），并用 R1-Zero 现象（准确率与回答长度同增）说明推理行为可被
RL 激发。最后给出评估范式转变：`score vs compute`、pass@k 与长程安全评估。

## 一页速览卡（面试前 5 分钟）

**必须记住的五件事**

1. 测试时算力四件套：CoT、self-consistency、验证器/重排序、搜索。
2. RLVR：奖励由规则/程序判定（Countdown 的 0/0.1/1.0），无偏好噪声。
3. GRPO：同 prompt 采 K 个回答，`Â_i=(r_i−mean)/std`，去掉 value network。
4. RLOO：`Â_i=r_i−mean_{j≠i}r_j`，leave-one-out 无偏基线。
5. 评估：`pass@k`、`score vs compute`；组内奖励全同时没有学习信号。

**必须能默写的公式**

```text
GRPO:  Â_i = (r_i − mean(r_1..K)) / std(r_1..K)
RLOO:  Â_i = r_i − (1/(K−1)) Σ_{j≠i} r_j
pass@k = 1 − C(n−c, k) / C(n, k)
GRPO loss ≈ −E[ min(ρ_i Â_i, clip(ρ_i,1±ε) Â_i) ] + β KL(π_θ‖π_ref)
```

**一句话区分**

- GRPO vs RLOO：组均值/方差归一化（有偏低方差） vs LOO 均值（无偏）。
- pass@1 vs pass@k：单次质量 vs 多样性覆盖上限。
- RLVR vs 偏好 RL：规则奖励 vs 学出来的奖励模型；前者更难被 hack 但覆盖任务有限。

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

## 深入推导

### 推导 1：RLOO 的无偏性

设同 prompt 组内回报 `r_1..r_K` 独立同分布，`E[r_i]=μ`。RLOO 基线
`b_i=(1/(K−1))Σ_{j≠i}r_j`。对任意 `i`：

```text
E[ (r_i − b_i) | r_i ] = r_i − μ
⇒ E[ r_i − b_i ] = μ − μ = 0，且 E[ b_i ] = μ
```

即基线无偏，优势的期望等于真实优势；同时 `b_i` 与 `r_i` 负相关，降低策略梯度方差。
它是 REINFORCE 在「同 prompt 多样本」场景下的方差最小化变体之一。

### 推导 2：GRPO 的归一化偏差

GRPO 用 `Â_i=(r_i−mean)/std`。分子同样无偏（`mean` 是基线），但除以随组变化的 `std`
引入了比率估计的偏差（类似 1/x 的 Jensen 偏差），且组内奖励全同时 `std=0` 未定义。
工程处理：加 `eps`、过滤零方差组、动态采样补足有效组。取舍是「有偏但更稳定的尺度」。

### 推导 3：pass@k 的无偏估计

从 `n` 个样本（`c` 个正确）中不放回抽 `k`，全错的概率：

```text
P(全错) = C(n−c, k) / C(n, k)  ⇒  pass@k = 1 − C(n−c,k)/C(n,k)
```

无偏性来自组合计数的对称性；实现时用对数 gamma 函数避免大数溢出。它是「策略在第 k 次
尝试内成功」的概率，衡量探索覆盖；greedy 准确率衡量单次质量。

### 推导 4：为什么训练会拉长回答

- 可验证任务里更长的推理链往往提高正确率（更多计算换准确率）；
- RL 只优化结果奖励，若长度与成功相关，策略会增加长度；
- KL 与熵项按 token 累积，存在结构性不对称；需要长度惩罚/预算或按 token 归一化。
这解释了 R1-Zero 的「准确率与长度同增」现象，以及为什么需要控制推理成本。

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

## 面试追问链（由浅入深）

1. 为什么 LLM 的 RL 能去掉 critic？→ 同一 prompt 的多采样天然构成自比较基线，
   不需要学 `V(s)` 来降方差。
2. GRPO 的 `std` 归一化有什么风险？→ 零方差组无定义；比率偏差；对异常值敏感。
3. RLOO 与 GRPO 谁更好？→ 理论 RLOO 无偏、GRPO 方差更低；实践差异取决于奖励分布与
   组大小，任务相关。
4. 可验证奖励有哪些坑？→ 规则漏洞、格式刷分、奖励稀疏（组内全对/全错）、
   reward hacking 的变体。
5. 为什么测试时算力需要单独评估？→ 能力随预算增长，单次准确率会低估高风险；
   应画 `score vs compute` 曲线。
6. CoT 的「推理」一定是忠实的吗？→ 不一定；可能是事后合理化，验证器与过程奖励
   是缓解方向。

## 自测题（附要点）

1. 证明 RLOO 基线无偏。
   - 要点：`E[b_i]=μ`；条件期望。
2. 构造一个组内奖励使 GRPO 优势全为零，并给出三种处理。
   - 要点：`r` 全相同；难度混合、过滤零方差组、动态采样。
3. 给定 `n=8, c=2`，计算 `pass@1` 与 `pass@2`。
   - 要点：0.25；`1−C(6,2)/C(8,2)=1−15/28≈0.464`。
4. 解释 R1-Zero 的长度增长，并设计一个长度控制奖励。
   - 要点：长度与成功相关；奖励改为 `r − λ·len/length_budget` 或长度分桶归一化。
5. 比较 GRPO 的 KL 项与 DPO 的 β 在训练中的作用。
   - 要点：都约束偏离参考模型；GRPO 是每步正则，DPO 是隐式奖励缩放。

## 关联笔记

- 前置：[L9 RLHF/DPO](lecture-09-rlhf-preference-optimization.md)
- 后续：[L14 探索与内在奖励](lecture-14-exploration.md)、
  [L18 前沿与研究方法](lecture-18-frontiers.md)

## 本讲小结

推理 RL 把 RL 从「连续控制」带到「语言 token」：目标可验证、组内比较、测试时算力可扩展。
GRPO/RLOO 是当前主力算法。下一讲回到经典问题：能不能学一个环境模型来提升样本效率？

## 延伸阅读

- Shao et al., *DeepSeekMath* (GRPO), 2024；DeepSeek-AI, *DeepSeek-R1*, 2025.
- Wei et al., *Chain-of-Thought Prompting*, 2022；Wang et al., *Self-Consistency*, 2022.
- Brown & Sandholm, *Superhuman AI for multiplayer poker*, Science 2019.
- 官方 slides 与默认项目 README。
