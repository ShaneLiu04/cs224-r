# Lecture 09 — RL for LLMs: RLHF & Preference Optimization

- 课程版本：Spring 2026（2025 有录像；2025 主讲 Archit Sharma）
- 日期：2026-04-29
- 官方材料：[09_cs224r_rlhf_2026.pdf](../slides/spring2026/09_cs224r_rlhf_2026.pdf) ·
  [2025 版](../slides/spring2025/09_cs224r-2025-rlhf.pdf) ·
  [Lecture 9 视频](https://www.youtube.com/watch?v=XKLGuwvSKvI) ·
  论文：InstructGPT、DPO、IPO
- 状态：初稿

## 摘要

LLM 后训练管线：预训练 → SFT → 偏好对齐。本讲把 RLHF 的 KL 约束目标解出闭式最优策略，
证明「策略与参考模型的对数比就是隐式奖励」，再把隐式奖励代入 Bradley-Terry 似然得到 DPO——
一个不需要奖励模型、不需要在线采样的分类式损失。最后比较 RLHF / DPO / IPO 的假设与失效
模式，讨论 reward over-optimization、风格偏差与对齐税。

## 一页速览卡（面试前 5 分钟）

**必须记住的五件事**

1. RLHF 目标：`max_π E[R] − βKL(π‖π_ref)`；PPO 优化，RM 固定。
2. 闭式最优：`π*(y|x) ∝ π_ref(y|x) exp(R(x,y)/β)`；反解 `R = β log(π*/π_ref) + β log Z(x)`。
3. DPO：把隐式奖励代入 BT 似然，`Z(x)` 在成对差中抵消，得到对数比的 log-sigmoid 损失。
4. IPO：用间隔平方损失防止 DPO 在重复偏好对上过拟合。
5. 现象：RLHF 改变风格、reward over-optimization、长度偏差、对齐税。

**必须能默写的公式**

```text
π*(y|x) = π_ref(y|x) exp(R(x,y)/β) / Z(x)
L_DPO = −E_{(x,y_w,y_l)}[ log σ( β[ (logπ_θ(y_w)−logπ_ref(y_w)) − (logπ_θ(y_l)−logπ_ref(y_l)) ] ) ]
L_IPO = E[ ( (Δ_w − Δ_l) − 1/(2β) )² ],   Δ = logπ_θ(y) − logπ_ref(y)
```

**一句话区分**

- RLHF vs DPO：显式 RM + 在线 RL vs 隐式奖励 + 离线分类。
- DPO vs SFT：偏好对相对信号 vs 单条正确示范。
- DPO vs IPO：log-sigmoid vs 间隔平方；后者对确定性偏好更稳健。

## 学习目标

- 画出 LLM 后训练管线：预训练 → SFT → 偏好数据 → 奖励模型 → PPO；
- 从 KL 约束目标推导最优策略的封闭形式与隐式奖励；
- 推导 DPO 损失，解释它为什么不需要单独训练 reward model；
- 比较 RLHF、DPO、IPO 的假设与失效模式；
- 认识 reward over-optimization、风格偏差与对齐税。

## 先修知识

- L3 策略梯度、L8 偏好奖励；语言模型基础（下一个 token 预测、对数似然）。

## 核心概念

### 1. LLM 训练三阶段

1. **预训练**：海量文本，学世界知识、语法、代码与推理雏形；
2. **指令微调（SFT）**：`(instruction, response)` 对，学「以助手方式回答」；
3. **偏好对齐**：让输出更符合人类偏好（有用、无害、诚实）。

### 2. 为什么 SFT 不够

- 收集「正确示范」昂贵，且示范只告诉模型「怎么做」不告诉「别怎么做」；
- 很多质量维度（语气、简洁、安全）难以示范，但容易比较；
- SFT 目标是模仿数据分布，不直接优化人类效用。

于是用**成对比较**：`(x, y_w, y_l)`，其中 `y_w` 优于 `y_l`。

### 3. RLHF 管线

```text
1. 用 SFT 模型初始化 π_θ
2. 对 prompt x 采样多个回复，人类排序 → 偏好对
3. 训练奖励模型 R_φ(x, y)（Bradley-Terry + log-sigmoid）
4. PPO 优化：
   max_θ  E_{x~D, y~π_θ}[ R_φ(x,y) ] − β · KL( π_θ(y|x) || π_ref(y|x) )
```

- `π_ref` 通常是冻结的 SFT 模型；
- KL 用逐 token 的估计：`log π_θ(y_t|x,y_<t) − log π_ref(...)`；
- 奖励常只在序列末尾给一次（可加 per-token 奖励）。

PPO 的 actor 是 LM，critic 也常是 LM 规模，显存成本高；奖励模型则固定不更新。

### 4. 从 KL 目标到隐式奖励

RLHF 目标：

```text
max_π  E_{y~π}[ R(x,y) ] − β D_KL( π(·|x) || π_ref(·|x) )
```

在任意 `x` 上对该目标做变分求解（忽略配分函数）：

```text
π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp( R(x,y) / β )
```

反解奖励：

```text
R(x,y) = β log( π*(y|x) / π_ref(y|x) ) + β log Z(x)
```

关键洞察：**最优策略与参考策略的对数比就等于奖励**（差一个只依赖 `x` 的常数）。

### 5. DPO（Direct Preference Optimization）

把隐式奖励代入 Bradley-Terry 似然，`Z(x)` 在成对比较中抵消：

```text
P(y_w ≻ y_l | x) = σ( β[ log(π_θ(y_w|x)/π_ref(y_w|x)) − log(π_θ(y_l|x)/π_ref(y_l|x)) ] )

L_DPO(θ) = −E_{(x,y_w,y_l)}[ log σ( β[ Δ_w − Δ_l ] ) ]
```

其中 `Δ = log π_θ(y|x) − log π_ref(y|x)`。

- 无需奖励模型、无需采样、无需 critic，只是「带对数比的两类分类」；
- 梯度会同时提高 `y_w`、降低 `y_l` 的似然，并按错误程度自适应加权；
- `β` 控制偏离参考模型的程度（等价于 KL 强度）。

### 6. DPO vs RLHF vs IPO

| 方法 | 奖励模型 | 采样 | 优点 | 风险 |
| --- | --- | --- | --- | --- |
| RLHF (PPO) | 需要 | 在线 | 可持续迭代、上限高 | 复杂、不稳定、过优化 |
| DPO | 不需要 | 离线 | 简单稳定、省资源 | 对偏好数据质量敏感；可能过拟合 `y_w` |
| IPO | 不需要 | 离线 | 修正 DPO 在确定性偏好下的过拟合 | 目标改为比分回归 |
| RLAIF | AI 标注 | 离线/在线 | 可扩展 | 偏差继承 |

**IPO**（Identity Preference Optimization）把 log-sigmoid 换成带间隔的平方损失
`(Δ_w − Δ_l − 1/(2β))²`，避免 DPO 在「同一对反复训练」时把概率推向极端。

### 7. 现象与局限

- RLHF 显著改变风格：更详细、列表更多、更礼貌（不一定更正确）；
- **reward over-optimization**：PPO 步数增加时 proxy reward 持续上升，真实偏好先升后降；
- 人类偏好有噪声与偏差（长度偏好、谄媚）；
- 对齐税（alignment tax）：对齐后某些能力/多样性可能下降；
- 单一奖励容易被钻空子，需要多目标、过程奖励、验证器（L10）。

## 关键推导：DPO 损失从哪来

1. 偏好模型：`P(y_w≻y_l) = σ(R(y_w) − R(y_l))`；
2. 隐式奖励：`R(y) = β log(π_θ(y)/π_ref(y)) + β log Z`；
3. 代入并对 `θ` 做极大似然 ⇒
   `L = −log σ( β[(logπ_θ(y_w) − logπ_ref(y_w)) − (logπ_θ(y_l) − logπ_ref(y_l))] )`。

注意：DPO 的「奖励」是策略自身的隐式量，**没有独立的通用奖励函数**；
但训练完成后可用 `β log(π_θ/π_ref)` 当 reward model 给新样本打分（有分布限制）。

## 深入推导

### 推导 1：KL 约束目标的闭式解

对固定 `x`，考虑

```text
max_{π(·|x)}  Σ_y π(y|x) R(x,y) − β Σ_y π(y|x) log( π(y|x)/π_ref(y|x) )
s.t.  Σ_y π(y|x) = 1
```

拉格朗日量对 `π(y|x)` 求导置零：

```text
R(x,y) − β( log(π/π_ref) + 1 ) + λ = 0
⇒ π*(y|x) = π_ref(y|x) exp( R(x,y)/β ) · exp( (λ−β)/β )
```

归一化常数 `Z(x)=Σ_y π_ref(y|x)exp(R/β)` 吸收 `exp((λ−β)/β)`，得
`π*(y|x)=π_ref(y|x)exp(R/β)/Z(x)`。反解即隐式奖励（差一个 `β log Z(x)`，只依赖 `x`）。

### 推导 2：DPO 损失的完整推导

BT 似然：

```text
P(y_w≻y_l|x) = σ( R(x,y_w) − R(x,y_l) )
```

代入隐式奖励 `R = β log(π_θ/π_ref) + β log Z(x)`：`β log Z(x)` 在差分中抵消，

```text
R_w − R_l = β[ log(π_θ(y_w)/π_ref(y_w)) − log(π_θ(y_l)/π_ref(y_l)) ]
```

对 `θ` 极大化 `log σ(R_w−R_l)` 即得 `L_DPO`。注意证明只用了「最优策略」的关系，
但 DPO 直接把它当作参数化损失使用——这是它的核心「偷换」。

### 推导 3：DPO 的梯度在做什么

记 `Δ = β(Δ_w − Δ_l)`，`σ = σ(−Δ)` 为排序错误概率：

```text
∇_θ L_DPO = −β σ(−Δ) [ ∇logπ_θ(y_w) − ∇logπ_θ(y_l) ]
```

- 提高 chosen 的对数似然、降低 rejected；
- 权重 `σ(−Δ)` 在排序错误时大、排序正确且差距大时趋于 0（自动难例挖掘）；
- 与 RLHF 的区别：没有采样、没有 RM，但也没有新数据探索。

### 推导 4：IPO 为什么更稳

DPO 在 `y_w` 与 `y_l` 可区分时会把 `Δ_w−Δ_l` 推向无穷（log-sigmoid 无饱和点的目标）。
IPO 换成

```text
L_IPO = ( (Δ_w − Δ_l) − 1/(2β) )²
```

目标存在有限最优间隔 `1/(2β)`，避免概率比发散；代价是需要调节 `β` 与偏好确定性匹配。

## 实现映射（Default Project）

| 阶段 | 文件 | TODO | 验收 |
| --- | --- | --- | --- |
| SFT | `default-project/starter-code/sft_trainer/sft.py` | `train(...)`：只在 response token 上算 loss | loss 正常下降 |
| IPO | `ipo_trainer/ipo.py` | `train(...)`：chosen/rejected 对数比 + 参考模型 | 与 SFT 基线对比 |
| RLOO | `rloo_trainer/rloo_update_worker.py` | `update(...)`：LOO 优势 + 熵/KL | pass@k 提升 |
| 评估 | `evaluation/countdown_eval.py` | pass@k、格式分 | 见 README |

Countdown 是**可验证任务**：奖励由规则给出（0/0.1/1.0），因此 IPO 的偏好对可由
「答案正确性」自动构造；RLOO 甚至不需要偏好对，直接用验证奖励。

## 易错点与反例

- **KL 项方向写反**：是 `KL(π_θ || π_ref)`，注意采样在 `π_θ` 上；
- **奖励模型只看 `y_w`**：必须是成对差分；
- **DPO 忘记除以参考模型**：没有 `π_ref` 项时会退化成偏好数据的 BC；
- **β 太小**：偏离参考过远、输出退化；**β 太大**：几乎学不动偏好；
- **把 DPO 当万能**：偏好数据有偏时，DPO 会放大偏差；
- **RLHF 奖励只给 EOS**：长回答累积 KL 惩罚，可能激励更短输出——需注意长度偏差。

## 实验或练习

1. 在 Countdown 上实现 SFT 与 IPO，比较 pass@k 与输出长度变化；
2. 推导 `π* = π_ref exp(R/β)/Z`：用 KKT 或变分法补齐每一步；
3. 用同一偏好数据训练 DPO 与 IPO，在 held-out 偏好对上看准确率与过度优化。

## 高频问题

- 问：RLHF 与 DPO 的本质联系？答：DPO 是 RLHF 闭式解代入偏好似然后的等价分类目标。
- 问：DPO 需要在线采样吗？答：不需要；这是它的优势也是上限（无新数据探索）。
- 问：为什么 RLHF 要用 KL？答：防止策略为刷奖励远离语言模型分布。
- 问：reward over-optimization 怎么发现？答：训练 reward 与独立的人工/模型评估曲线分叉。
- 问：什么任务更适合 DPO？答：有高质量静态偏好对、算力受限、追求稳定；探索需求强时用 PPO/RLOO。

## 面试追问链（由浅入深）

1. DPO 真的完全等价于 RLHF 吗？→ 在「最优策略关系 + 偏好数据分布 + BT 假设」下等价；
   实践中有分布偏移（DPO 离线、RLHF 在线）与优化差异。
2. `β` 在 DPO 与 RLHF 中含义一样吗？→ 都控制偏离参考模型的程度；DPO 中它还缩放
   隐式奖励，影响损失曲率。
3. 为什么 DPO 也需要参考模型？→ 隐式奖励定义为对数比；去掉 `π_ref` 会退化成偏好对上的 BC。
4. 没有在线采样会带来什么局限？→ 无法探索新回答、无法修正偏好数据未覆盖的行为；
   迭代式 DPO/在线 DPO 是常见补救。
5. reward over-optimization 如何在训练中检测？→ 代理奖励上升、独立评估/人工偏好下降；
   看二者的分叉点决定早停。
6. 为什么 RLHF 的输出变得更长？→ 长度与人类偏好相关 + KL 项按 token 累积，
   存在短/长回答的不对称激励；需要长度归一化或惩罚。

## 自测题（附要点）

1. 完整推导 `π*(y|x) ∝ π_ref exp(R/β)`。
   - 要点：拉格朗日乘子 + 归一化。
2. 从 BT 与隐式奖励推导 DPO 损失，指出在哪一步消去 `Z(x)`。
   - 要点：成对差分。
3. 写出 DPO 梯度并解释 `σ(−Δ)` 权重。
   - 要点：错误越大梯度越大。
4. 比较 SFT、DPO、PPO 的数据与计算需求。
   - 要点：示范 vs 偏好对离线 vs 在线采样 + RM + critic。
5. 给定 `Δ_w=0.2, Δ_l=0.1, β=0.5`，计算 DPO 的 logit 与损失。
   - 要点：logit `0.5×0.1=0.05`；`−logσ(0.05)`。

## 关联笔记

- 前置：[L8 奖励学习](lecture-08-reward-learning.md)
- 后续：[L10 LLM 推理 RL](lecture-10-rl-llm-reasoning.md)、
  [L17 VLA 的 RL 微调](lecture-17-rl-for-vlas.md)（优势条件监督同源）

## 本讲小结

偏好对齐的两条主线：RLHF 显式学奖励再 RL；DPO 用隐式奖励一步到位。默认项目把
SFT → IPO → RLOO 串起来，正好覆盖「监督 → 偏好 → 可验证奖励」。L10 深入推理任务的 RL。

## 延伸阅读

- Ouyang et al., *InstructGPT*, 2022.
- Rafailov et al., *DPO*, 2023；Azar et al., *IPO*, 2024.
- Bai et al., *Constitutional AI*, 2022.
- 官方 slides：`slides/spring2026/09_cs224r_rlhf_2026.pdf`。
