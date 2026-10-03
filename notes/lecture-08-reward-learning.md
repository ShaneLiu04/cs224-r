# Lecture 08 — Reward Learning

- 课程版本：Spring 2026（2025 有录像）
- 日期：2026-04-24
- 官方材料：[08_cs224r_reward_learning_2026.pdf](../slides/spring2026/08_cs224r_reward_learning_2026.pdf) ·
  [2025 版](../slides/spring2025/08_cs224r_reward_learning_2025.pdf) ·
  [Lecture 8 视频](https://www.youtube.com/watch?v=PDIxDhA9Z6Y) ·
  论文：Christiano et al. (2017)、VICE、GAIL
- 状态：初稿

## 学习目标

- 说清奖励从哪来：手写、goal classifier、人类偏好、AI 反馈；
- 解释 reward classifier 被 RL exploit 的机制与缓解方法；
- 写出基于 Bradley-Terry 的偏好奖励模型损失；
- 画出 RLHF 循环（采样 → 偏好 → 奖励模型 → RL），知道每步作用；
- 认识 reward hacking / over-optimization 与评估困难。

## 先修知识

- L4–L7：RL 目标、离线数据、BC；二分类交叉熵；GAN 的对抗直觉。

## 核心概念

### 1. 奖励是最大的瓶颈

- 游戏：奖励由规则给出；
- 真实世界：只有成功/失败信号，稀疏且延迟；
- 复杂任务（对话、写代码）：没有可解析的奖励函数。

三条常见路径：**演示**（BC 不知道奖赏）、**目标状态**（goal classifier）、
**人类反馈**（偏好学习）。

### 2. Goal classifier（目标分类器）

思路：给「目标达成」的状态分类器 `p(goal | s)`，用 `log p(goal|s)` 当奖励。

- 正例：专家达到目标的状态；负例：其他状态；
- 问题：RL 会**主动搜索分类器的弱点**：对抗样本、分布外状态得到高分类分；
- 缓解：VICE 用对抗训练 + 负样本挖掘，把 RL 访问的、分类为高的状态加入负例；
  或用约束优化限制策略偏离数据分布。

这与 GAN 的结构同源：判别器与「生成器」（策略）互相博弈。判别器同时充当奖励，
也同时会传授演示行为（GAIL：`D(s,a)` 区分专家与策略，`log D` 当奖励）。

### 3. 从人类比较学奖励

让人类对两条轨迹（或两段回复）排序：`τ_w ≻ τ_l`。假设 Bradley-Terry 模型：

```text
P(τ_w ≻ τ_l) = σ( R_φ(τ_w) − R_φ(τ_l) ) = exp(R_w) / (exp(R_w) + exp(R_l))
```

奖励模型损失（交叉熵）：

```text
L_R(φ) = −E_{(w,l)}[ log σ( R_φ(w) − R_φ(l) ) ]
```

- 轨迹奖励可分解为状态奖励和 `R_φ(τ) = Σ_t r_φ(s_t, a_t)`（或末状态/token 级）；
- 比较比绝对打分更可靠、标注更便宜（Christiano et al., 2017 的原始动机）。

### 4. RLHF 循环

```text
1. 预训练 LM
2. 指令微调（SFT）
3. 采样多个回复，人类/标注员排序 → 偏好数据
4. 训练奖励模型 R_φ
5. 用 PPO 优化 π_θ 最大化 R_φ − β KL(π_θ || π_ref)
6. 迭代 3–5 收集新数据
```

KL 项的作用：保持语言质量、防止过度优化奖励模型。L9 会详细推导与 DPO 对比。

### 5. RLAIF / Constitutional AI

用 AI（通常也是 LLM）代替人类标注偏好，成本低、可扩展；风险是继承 AI 的偏差。
Constitutional AI 用一套原则让模型自我批评与修订，再对修订对做偏好训练。

### 6. 从偏好到人类意图：局限

- 人类偏好不可靠：标注者噪声、不一致、被表面风格影响（更长、更自信的回答被偏好）；
- **reward over-optimization**：PPO 迭代越久，proxy reward 上升但真实偏好先升后降；
- 评测困难：训练奖励与真实目标脱节时，需要人工评估或独立 benchmark；
- 社会影响：Goodhart 定律「当指标变成目标，它就不再是好指标」。

### 7. 无监督 RL / 自提出目标

没有外部奖励时，可以用内在目标：

- 目标生成（goal generation）：以「当前能力边缘」的状态为目标（self-play、curriculum）；
- 技能发现：最大化状态覆盖 `I(s; z)`（DIAYN，L15）；
- 好奇心/预测误差（L14）。

这就是「agent 自己提出目标、自己评价」的无人监督 RL 方向。

## 关键推导：偏好损失 = 二分类注意力加权

把 `L_R = −log σ(R_w − R_l)` 拆开：只有当 `R_w − R_l` 很大时梯度消失；
错误排序（`R_w < R_l`）时梯度较大，把质量沿两条轨迹反方向推。注意奖励只能确定到**常数平移**
（`R + c` 不影响成对概率），因此实践中给奖励模型加正则或归一化（如期望为零）。

## 实现映射

| 场景 | 数据 | 奖励来源 | 对应位置 |
| --- | --- | --- | --- |
| HW2/HW3 MuJoCo | 手工 shaping / 环境稀疏奖励 | 环境 | 无需学习奖励 |
| Default Project | Countdown 正确答案 | 规则验证（verifiable reward） | `evaluation/countdown.py` |
| Default Project IPO | 正确/错误回答对 | 偏好对（chosen/rejected） | `ipo_trainer/` |
| 期末项目 | 自定义 | 可能需要偏好/分类器奖励 | 见 `experiments/topics/reward-learning.md` |

Default Project 的 Countdown 评分：`0.0` 无 `<answer>`；`0.1` 格式合法但答案错；
`1.0` 用给定数字构成等式且结果正确。这是**可验证奖励**（RLVR）的最简形态。

## 易错点与反例

- **把分类器概率当真实奖励**：策略会 hack；需要负例挖掘与分布约束；
- **偏好对顺序写反**：`σ(R_w − R_l)`，不是 `σ(R_l − R_w)`；
- **奖励模型过拟合小数据**：需要正则/早停，并用 held-out 比较准确率；
- **忽略 KL 惩罚**：策略退化成「刷分器」，语言质量崩坏；
- **只看训练 reward 曲线**：必须看真实偏好/人工评估，防止 over-optimization。

## 实验或练习

1. 实现 `L_R` 的两个版本（log-sigmoid vs 直接回归分数差），在玩具偏好数据上比较；
2. 用 Flappy Bird 状态训练一个 goal classifier（存活 1000 步为目标状态），
   观察 PPO 用它当奖励时是否出现 exploit；
3. 读 InstructGPT 论文的 3.3–3.6 节，记录 RLHF 各阶段的指标变化。

## 高频问题

- 问：为什么用比较而不是绝对打分？答：更稳定、标注成本低、尺度无关。
- 问：GAIL 与 goal classifier 的关系？答：都是「判别器当奖励」的对抗式逆 RL。
- 问：reward hacking 的本质？答：优化 proxy 导致真实目标下降（Goodhart 定律）。
- 问：RLAIF 的风险？答：继承模型偏见与误差，难以发现标注错误。
- 问：KL 正则为什么放在奖励里？答：约束策略不要为刷分偏离参考模型太远。

## 本讲小结

奖励工程是 RL 落地的关键：从演示、目标、偏好三类信号各有所求。偏好学习引出 RLHF，
下一讲把 RLHF 与 DPO 放在一起做完整推导，并讨论奖励不可靠时的对策。

## 延伸阅读

- Christiano et al., *Deep RL from Human Preferences*, 2017.
- Fu et al., *VICE*, 2018；Ho & Ermon, *GAIL*, 2016.
- Ouyang et al., *InstructGPT*, 2022；Bai et al., *Constitutional AI*, 2022.
