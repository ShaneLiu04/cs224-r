# Lecture 13 — Meta-Reinforcement Learning

- 课程版本：Spring 2026（2025 有录像）
- 日期：2026-05-13
- 官方材料：[13_cs224r_metarl_2026.pdf](../slides/spring2026/13_cs224r_metarl_2026.pdf) ·
  [2025 版](../slides/spring2025/13_cs224r_metarl_2025.pdf) ·
  [Lecture 13 视频](https://www.youtube.com/watch?v=wSiyEpvoGkA) ·
  论文：RL²、MAML、PEARL、VariBAD
- 状态：初稿

## 学习目标

- 形式化 meta-RL：任务分布 + 少量交互内的快速适应；
- 描述黑盒 meta-RL（RL²）的输入输出与训练方式；
- 推导 MAML 的二阶梯度与一阶近似（FOMAML/Reptile）差异；
- 写出 PEARL 的隐变量上下文、变分下界与 posterior sampling 探索；
- 理解 meta-RL 中探索难题（探索与执行的耦合）及两类解法。

## 先修知识

- 多任务（L12）、策略梯度（L3）、变分推断基础（ELBO）。

## 核心概念

### 1. 问题设定

元学习把「任务」当作从分布 `p(T)` 中采样的样本，目标是：

```text
max_θ  E_{T_i ~ p(T)} [ J_{T_i}( adapt(θ, D_i^{support}) ) ]
```

- **meta-train**：在大量任务上训练「学习算法/初始化/上下文编码器」；
- **meta-test**：面对新任务，用少量交互或少量轨迹快速适应。

与多任务的区别：多任务测试时任务标识已知（`z` 直接给），meta-RL 测试时任务未知，
必须**从经验中推断任务**。

### 2. 形式化对比

| 设定 | 训练时 | 测试时 | 任务信息 |
| --- | --- | --- | --- |
| Multi-task | 多任务数据 | 已知 `z` | 显式给出 |
| Transfer | 源任务 | 目标任务微调 | 部分 |
| Meta-RL | 任务分布 | 新任务 + 少量交互 | 需推断 |

### 3. 黑盒 Meta-RL（RL²）

把任务推断交给序列模型：策略输入是完整历史 `(s_1,a_1,r_1,…,s_t)`，
用 RNN/Transformer 隐式估计任务并输出动作：

```text
a_t ~ π_θ( a_t | s_1, a_1, r_1, …, s_t )
```

- 训练目标就是普通 RL（如 PPO/TRPO）在任务分布上的期望回报；
- 「适应」发生在隐状态更新中，不需要梯度更新；
- 与上下文条件策略（Decision Transformer、in-context RL）同源；
- 成功依赖探索：模型必须主动收集「区分任务」的信息。

### 4. MAML（Model-Agnostic Meta-Learning）

学一个**初始化** `θ`，使得在新任务上做少量梯度步就能得到好策略：

```text
内循环（每个任务）：θ'_i = θ − α ∇_θ L_{T_i}(θ)
外循环（meta）：    θ ← θ − β ∇_θ Σ_i L_{T_i}(θ'_i)
```

外循环需要对内循环求导（二阶梯度），实现上简化：

- **FOMAML**：忽略二阶项，用 `∇_{θ'} L(θ'_i)` 近似；
- **Reptile**：`θ ← θ + ε (θ'_i − θ)`，不需要显式二阶。

MAML 对策略梯度任务较昂贵，实践中更适合监督式元学习或小网络。

### 5. 上下文式 Meta-RL：PEARL

PEARL 用一个隐变量 `z` 编码任务，做**离线/在线 off-policy** 元训练：

- 推断网络 `q_φ(z | c)`，`c` 是最近若干 `(s,a,r,s')` 组成的上下文；
- 策略与 Q 条件于 `(s, a, z)`；
- 训练用变分下界（ELBO）：

```text
L = E_{z~q_φ(z|c)}[ L_critic + L_actor ] + β D_KL( q_φ(z|c) || p(z) )
```

- 测试时用少量数据更新 `q(z|c)`，策略立即适应。

### 6. Meta-RL 中的探索

Meta-RL 的独特难点：**收集什么数据决定了任务推断的质量**。
奖励可能诱导策略只顾执行而不去「试探」任务参数。两类解法：

1. **端到端优化探索**：把整个 episode 的回报当目标，让探索自然出现
   （RL²、VariBAD；需要长时域信用分配）；
2. **后验采样（posterior sampling）**：维护任务后验，按「从后验采样一个假设任务、
   最优执行」行动。这天然带有信息收集行为（PEARL 在测试时用后验更新实现类似效果）。

Cheap exploration 的例子：走廊任务中不同门后面奖励不同，智能体需要先试探
（走进不同门）再执行最优策略。

### 7. 应用：CS 教育

讲者展示了 meta-RL 用于学生编程作业的 bug 定位与反馈：

- 把「学生/题目」当任务，少量交互后推断学生错误模式；
- 学到的探索行为会主动测试边界条件（如球撞墙、球出界）；
- 在 CS106A 的真实评分中带来更快（44%）且更准（6%）的辅助。

## 关键推导：PEARL 的 ELBO 结构

最大化 `log p(c)` 的变分下界：

```text
log p(c) ≥ E_{q(z|c)}[ log p(c | z) ] − D_KL( q(z|c) || p(z) )
```

在 RL 中 `p(c | z)` 由 `(s,a,r,s')` 的似然近似，实践上把 critic/actor 损失
当作 `log p(c|z)` 的代理项（probabilistic 视角），再加 KL。这个框架解释了
「上下文编码 + 条件策略 + 后验正则」三件套。

## 实现映射

| 概念 | 材料 | 说明 |
| --- | --- | --- |
| 黑盒 meta-RL | `assignments/spring2025/homework4-goal-meta-rl/starter-code/meta_rl/` | RNN 策略 + 任务分布环境 |
| 任务采样 | `meta_rl/envs/`、`meta_rl/configs/` | 训练/测试任务划分 |
| 目标条件基线 | `.../goal_conditioned_rl/` | 对照组（任务已知） |

## 易错点与反例

- **meta-train / meta-test 任务泄漏**：测试任务不能出现在训练分布；
- **把 multi-task 当 meta-RL 评估**：测试时必须隐藏任务标识；
- **上下文太短**：任务推断不准；太长则样本效率低；
- **内循环步数过多**：MAML 退化为普通微调，失去「快速适应」意义；
- **探索不给额外激励**：端到端方法可能收敛到「不探索也能拿中等奖励」的局部最优。

## 实验或练习

1. 在 2D 导航任务分布（目标位置随机）上比较 RL² 与 GCRL（条件于目标）的少样本适应；
2. 实现 FOMAML 与 Reptile 的一步更新，比较与 MAML 的差距；
3. 阅读 VariBAD 论文，写出它如何用 RNN 同时做任务推断与价值估计。

## 高频问题

- 问：meta-RL 与 multi-task 的核心差别？答：任务是否已知；meta-RL 要从经验推断任务。
- 问：RL² 为什么能适应？答：RNN 隐状态隐式实现了「任务后验」。
- 问：MAML 的代价？答：二阶梯度、内循环需可微，且对 RL 的高方差敏感。
- 问：meta-RL 的探索难在哪？答：探索行为要在 meta-train 学到，信用分配跨越多个 episode。
- 问：posterior sampling 为什么有效？答：随机选一个「信念中的任务」并最优执行，自然产生信息收集。

## 本讲小结

Meta-RL 把「快速适应」当作学习目标。黑盒法（RL²）用序列模型隐式推断；优化法（MAML）
学好初始化；概率法（PEARL）显式维护任务后验并支持后验采样探索。下一讲补上 2025 版的
exploration 专题，理解「主动收集信息」的通用工具。

## 延伸阅读

- Duan et al., *RL²*, 2016；Finn et al., *MAML*, 2017；Nichol et al., *Reptile*, 2018.
- Rakelly et al., *PEARL*, 2019；Zintgraf et al., *VariBAD*, 2020.
- Liu et al., *Decoupling Exploration and Exploitation*, 2021.
- 官方 slides 与 2025/2023 HW4。
