# Lecture 12 — Multi-Task and Goal-Conditioned RL

- 课程版本：Spring 2026（2025 有录像）
- 日期：2026-05-08
- 官方材料：[12_cs224r_mtrl_gcrl_2026.pdf](../slides/spring2026/12_cs224r_mtrl_gcrl_2026.pdf) ·
  [2025 版](../slides/spring2025/12_cs224r_mtrl_gcrl_2025.pdf) ·
  [Lecture 12 视频](https://www.youtube.com/watch?v=qNdsI_4AQJw) ·
  论文：HER (Andrychowicz et al., 2017)
- 状态：初稿

## 学习目标

- 把多任务学习形式化为任务分布 `p(T)` 上的期望，并说明条件 MDP 的构造；
- 写出多任务模仿学习与多任务 RL 的目标，理解任务条件化（task id、语言、目标状态）；
- 推导 HER 的重标记流程，解释它为什么能把失败变成功；
- 区分 weight sharing 与 data sharing，理解 GCRL 是多任务的特例；
- 认识负迁移与任务表示设计。

## 先修知识

- L1–L2 的 MDP 与模仿学习；L6–L7 的值方法；L11 的模型视角。

## 核心概念

### 1. 任务分布

多任务 RL 假设任务 `T_i = (S, A, P_i, r_i, ρ_i)` 来自分布 `p(T)`，目标是：

```text
max_θ  E_{T_i ~ p(T)} [ J_{T_i}(θ) ]
```

不同任务的差异可以体现在初始状态、目标、奖励、动力学甚至动作空间。

**条件 MDP**：把所有任务放进一个 MDP，状态里加入任务标识 `z_i`：

```text
s' ~ P(s' | s, a, z_i),   r ~ r(s, a, z_i),   π_θ(a | s, z)
```

这样多任务 RL 就变成「带条件的单任务 RL」。`z` 的表示可以是 one-hot、语言指令、
目标图像/状态、隐变量。

### 2. 多任务模仿学习

数据 `D = {(s, a, z)}`，目标仍是极大似然：

```text
L = −E_{(s,a,z)~D}[ log π_θ(a | s, z) ]
```

实践要点：

- 任务条件要足够表达（例如语言指令 + 目标图像）；
- 大模型（Transformer）擅长吸收多任务数据，任务泛化更好；
- 数据量与任务多样性决定泛化，而不是单任务的训练技巧。

### 3. 多任务 RL

直接对联合 MDP 做 PPO/SAC/AWR 等，只是 `π(a|s,z)`、`Q(s,a,z)`、`V(s,z)`
都条件于任务。主要挑战：不同任务的奖励尺度、难度、梯度方向冲突（负迁移）。

- **weight sharing**：共享网络参数，用任务编码区分；
- **data sharing**：不同任务的数据混合训练；
- 任务采样策略：均匀 vs 按难度/不确定性（curriculum）。

### 4. Goal-Conditioned RL（GCRL）

GCRL 是多任务的特例：`z = s_g`（目标状态）。奖励常见形式：

```text
稀疏：r(s, a, s_g) = 1[ ||φ(s) − φ(s_g)|| ≤ ε ]       （到达目标）
稠密：r = −||φ(s) − φ(s_g)||                          （距离惩罚）
```

策略 `π(a | s, s_g)`，值函数 `V(s, s_g)` 或 `Q(s, a, s_g)`。
关键收益：**一套参数可以泛化到任意目标**，包括训练时没见过的目标。

### 5. Hindsight Experience Replay（HER）

稀疏奖励下失败轨迹没有信号。HER 的做法：把轨迹**实际到达的状态**当成目标重新标注：

```text
原始：τ = (s_1…s_T, a_1…a_T, g, r≈0)
重标：选 g' = s_T（或途中某状态），得到新数据 (s_t, a_t, s_{t+1}, g', r')
```

- 只要用相对目标表示，重标后的轨迹**必然成功**（到达了自己）；
- 数据集从「几乎全失败」变成「大量成功样例」；
- 变体：future（从未来状态选）、final（只用终点）、episode（任意同轨迹状态）、
  random/hybrid。

代价：目标分布偏向实际可达状态；需要与课程学习或目标采样配合。

## 关键推导：HER 的重标概率

设目标从「未来轨迹状态」采样（future strategy）。对第 `t` 步，
选 `g' = s_{t'}, t' ≥ t`，此时从 `(s_t, a_t)` 出发的可达目标包含 `s_{t'}`，
奖励 `r' = 1`。因此每条长度 `T` 的失败轨迹能提供 `O(T)` 个成功转移，
数据效率大幅提升。同时目标分布是行为策略可达分布，策略学会「从任意状态到任意可达状态」。

## 实现映射（2025/2023 HW4）

| 概念 | 文件 | TODO/说明 |
| --- | --- | --- |
| Goal-conditioned Q | `assignments/spring2025/homework4-goal-meta-rl/starter-code/goal_conditioned_rl/q_network.py` | 目标拼接进输入 |
| HER 逻辑 | 同目录训练脚本 | `run_episode` + 重标 |
| Bit-flip 环境 | `goal_conditioned_rl/bit_flip_env.py` | 离散状态目标 |
| 元学习部分 | `.../meta_rl/` | L13 使用 |

2026 版没有 HW4，但期末项目仍可基于 GCRL/HER 选题。

## 易错点与反例

- **目标表示**：用绝对状态 vs 相对差分；HER 生效的前提是「目标可从当前状态达成」，
  用了绝对目标但状态编码不包含目标时很难学；
- **奖励尺度**：稠密距离奖励便于学习但可能被 hack（原地抖动）、稀疏奖励依赖 HER/课程；
- **重标记忘记改 reward**：只改 `g` 不改 `r` 的数据是错的；
- **任务采样失衡**：简单任务主导梯度，难任务学不动；
- **把 GCRL 当单任务**：评估要覆盖多目标（含未见目标），不能只测一个 `g`。

## 实验或练习

1. 在 Bit-flip 环境实现 HER，比较有/无 HER 的成功率（2023 HW4 有完整脚手架；
   2025 HW4 使用更新的 meta_rl 结构）；
2. 设计三种目标采样策略（final/future/episode），比较收敛速度；
3. 思考：语言指令作为 `z` 与目标状态作为 `z` 的优缺点（表达力、监督来源、泛化）。

## 高频问题

- 问：多任务 RL 与单任务 RL 的本质差别？答：目标变成任务分布上的期望，策略条件于任务。
- 问：GCRL 与 multi-task 的关系？答：GCRL 是 `z = s_g` 的特例。
- 问：HER 为什么有效？答：用实际到达状态重标目标，把失败转成成功并扩大目标覆盖。
- 问：HER 的代价？答：目标分布偏向可达状态；可能忽略长程/稀有成功。
- 问：什么时候用密集奖励？答：学习初期；后期可退火到稀疏以避免被 hack。

## 本讲小结

多任务 = 条件 MDP + 任务编码；GCRL = 目标条件化；HER = 用未来状态重标目标。
这三个概念在语言条件机器人与 VLA（L15/L17）中无处不在。下一讲进一步扩展到「任务未知、
必须快速适应」的元学习设定。

## 延伸阅读

- Andrychowicz et al., *Hindsight Experience Replay*, 2017.
- Kaelbling, *Learning to Achieve Goals*, 1993；Schaul et al., *UVFA*, 2015.
- 官方 slides 与 2025/2023 HW4 handout。
