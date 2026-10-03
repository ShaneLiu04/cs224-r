# Lecture 07 — Offline Reinforcement Learning

- 课程版本：Spring 2026（2025 有录像）
- 日期：2026-04-22
- 官方材料：[07_cs224r_offline_rl_2026.pdf](../slides/spring2026/07_cs224r_offline_rl_2026.pdf) ·
  [2025 版](../slides/spring2025/07_cs224r_offline_rl_2025.pdf) ·
  [Lecture 7 视频](https://www.youtube.com/watch?v=lRDaXnPIzks) ·
  论文：AWR、IQL、CQL、AWAC；作业：HW3
- 状态：初稿

## 摘要

离线 RL 只给固定数据集、不能探索。核心困难是：策略查询数据分布外的动作（OOD）时 Q 值
不可信，而 bootstrap 会把误差沿轨迹放大。三类解法：加权模仿（AWR/AWAC）、隐式避免 OOD
（IQL 用 expectile 学上分位数并只用数据动作）、显式保守（CQL 压低 OOD 动作价值）。
本讲还引入 stitching——离线 RL 能拼接次优轨迹片段，是它与模仿学习的本质区别。

## 一页速览卡（面试前 5 分钟）

**必须记住的五件事**

1. OOD 动作 + bootstrap = extrapolation error：Q 高估 → 策略偏向 OOD → 误差继续放大。
2. AWR/AWAC 策略目标：`−E[logπ(a|s)·exp(A/λ)]`；`λ→∞` 退化为 BC。
3. Expectile：`L₂^ζ(µ)=|ζ−1{µ≤0}|µ²`；`ζ>0.5` 学上分位数，可近似 `max` 而无需查询 OOD。
4. IQL：`V` 拟合 `Q` 的上 expectile；`Q` 目标用 `r+γV(s')`，只在数据动作上更新。
5. Stitching：把数据中不同轨迹的优质片段拼成更优策略；Filtered BC 做不到。

**必须能默写的公式**

```text
AWR:       L_π = −E_{(s,a)~D}[ log π_θ(a|s) exp( A(s,a)/λ ) ]
Expectile: argmin_m E[ |ζ − 1{X≤m}| (X−m)² ]
IQL:       L_V = E[ L₂^ζ( Q̄(s,a) − V_ψ(s) ) ]
           L_Q = E[ ( r + γ V_ψ(s') − Q_θ(s,a) )² ]        （a 来自数据）
CQL:       α( E_{s,a~µ}[Q] − E_{(s,a)~D}[Q] ) + L_TD
```

**一句话区分**

- BC vs AWR：均匀模仿 vs 按优势指数加权模仿。
- AWAC vs IQL：Bellman 目标采样 `a'~π`（有 OOD 风险） vs 用 `V(s')`（只用数据动作）。
- IQL vs CQL：结构性避免查询 OOD vs 显式惩罚 OOD 取值。

## 学习目标

- 定义离线 RL 问题，解释为什么不能直接用 off-policy actor-critic；
- 说清 OOD 动作上的 Q 值误差如何被 bootstrap 放大；
- 写出 AWR/AWAC 的 advantage 加权回归目标，理解温度 `λ` 的作用；
- 推导 expectile 回归，写出 IQL 的 `V`/`Q`/策略三件套；
- 比较 IQL 与 AWAC/CQL，理解 stitching 现象以及在 PointMass 上如何检验。

## 先修知识

- L4–L6：actor-critic、Q-learning、Bellman 方程、BC 与 AWR 的关系。

## 核心概念

### 1. 问题设定

给定固定数据集 `D = {(s, a, r, s')}`（由未知行为策略 `µ` 收集），
不与环境交互，学习最大化期望回报的策略。典型来源：日志数据、人类演示、旧策略回放、
其他任务数据。

### 2. 为什么不能直接用 off-policy 方法

以 SAC 为例，critic 目标包含 `a' ~ π_θ(·|s')`：当 `π_θ` 偏离 `µ` 时，`a'` 可能是数据中
从未出现过的动作（OOD），网络对它的 Q 估计没有监督信号。Bootstrapping 让错误沿轨迹
传播：`Q` 高估 OOD 动作 → 策略偏向 OOD 动作 → 下轮又查询 OOD 动作 → 误差放大（extrapolation error）。

模仿学习只学数据内动作，所以没有 OOD 问题，但**无法超越数据**（不能 stitching，
也不理解奖励）。离线 RL 要同时利用奖励并约束在数据支撑集附近。

### 3. 简单基线：加权模仿

**BC**：只匹配动作，无视奖励。

**Filtered BC**：只用高回报轨迹。数据利用率低、难处理次优轨迹拼接。

**AWR（Advantage-Weighted Regression）**：用指数的优势权重：

```text
L_π(θ) = −E_{(s,a)~D}[ log π_θ(a|s) · exp( (1/λ) · A^{π_k}(s,a) ) ]
```

- 优势越高（比当前策略平均好）的动作权重越大；
- `λ → ∞` 退化为 BC；`λ → 0` 只模仿最优动作；
- 优点：完全不需要 OOD 查询，天然离线；
- 缺点：需要估计 `A`。若用单步/MC 回报，优势精度有限；若用 TD，则又回到 OOD 问题。

### 4. 用 TD 估计优势，避免 OOD：IQL

IQL（Implicit Q-Learning）的关键技巧是**expectile 回归**：不学条件均值，而学条件分布的
上分位数。

Expectile 定义：

```text
L_2^ζ(µ) = |ζ − 1{µ ≤ 0}| · µ²
argmin_m E[ L_2^ζ(X − m) ]
```

- `ζ = 0.5` 时退化为 MSE（均值）；
- `ζ > 0.5` 时对 `X > m` 的样本加权更多，解为上 expectile（大数据侧）；
- 直觉：让 `V_ψ(s)` 逼近 `max_{a' ~ 数据} Q(s,a')` 的近似上界，而无需查询 OOD 动作。

三件套：

```text
L_V(ψ) = E_{(s,a)~D}[ L_2^ζ( Q_θ̄(s,a) − V_ψ(s) ) ]          # 拟合 Q 的上 expectile
L_Q(θ) = E_{(s,a,s')~D}[ ( r(s,a) + γ V_ψ(s') − Q_θ(s,a) )² ]  # 只对数据动作回归，用 V 做目标
L_π(θ) = −E_{(s,a)~D}[ log π_θ(a|s) · exp( β (Q_θ̄(s,a) − V_ψ(s)) ) ]  # AWR 抽取策略
```

- `Q̄` 是 target Q 网络；
- critic 只在数据动作 `a ~ D` 上训练，**不采样 `a'`**，因此没有 OOD 动作；
- `V` 只是「数据内优势的 amortized 估计」，不做 `max` 的显式优化；
- `ζ` 是核心超参：`ζ=0.9` 常用，太低接近均值，太高会被数据中的极端值主导。

IQL 的训练目标可以理解为「算策略改进而不真的在 OOD 动作上 query」。

### 5. CQL（Conservative Q-Learning）

另一条路线：显式压低 OOD 动作的 Q 值，让 Q 成为真实值的下界：

```text
L_CQL = α ( E_{s~D, a~µ}[Q(s,a)] − E_{(s,a)~D}[Q(s,a)] ) + L_TD
```

- 第一项压 OOD 分布（如当前策略 `µ=π_θ`）的 Q；
- 第二项抬数据动作的 Q；
- `α` 控制保守程度。CQL 简单但需要采样动作，计算量较大。

AWAC（HW3 使用）介于 AWR 与 SAC 之间：critic 用 Bellman 更新（但目标动作来自当前策略，
带来 OOD 风险），策略用 advantage 加权回归。实践中常加 BC 正则或使用数据动作近似。

### 6. 离线 RL 的评估与 stitching

- **评估**：只用数据内回报会高估；主流做法是在相同环境做少量在线评估，
  报告 `Eval_AverageReturn`；
- **stitching**：数据中可能只有「A→B 的最优片段」与「B→C 的最优片段」，
  离线 RL 应能拼出 A→C 的更好策略。IQL 由于 `Q` 的目标只基于数据内转移并传播，
  在 PointMass 这类任务上通常表现出 stitching；Filtered BC 做不到。

## 关键推导：IQL 为什么能避开 OOD

如果直接学 `max`：`Q(s,a) = r + γ max_{a'} Q(s',a')`，就需要评估 `argmax` 动作，
它可能不在数据中。IQL 用 `V_ψ(s')` 代替 `max_{a'} Q(s',a')`：

```text
Q target = r + γ V_ψ(s')
```

而 `V_ψ(s)` 通过 expectile 回归从 `Q(s,a)`（a 来自数据）隐式逼近上界。
因此整条更新链只涉及数据动作。代价是 expectile 只是 max 的近似：`ζ` 越大越接近 max，
方差也越大。

## 深入推导

### 推导 1：AWR 从哪里来（KL 正则策略迭代）

从「改进策略但不离开旧策略太远」的目标出发：

```text
max_π  E_{s~D, a~π}[ A^{π_k}(s,a) ] − λ D_KL( π(·|s) || π_k(·|s) )
```

在 `s` 固定、`Σ_aπ(a|s)=1` 的约束下用拉格朗日法，最优解为

```text
π*(a|s) ∝ π_k(a|s) · exp( A^{π_k}(s,a) / λ )
```

把它投影回参数化策略（对数据分布做加权极大似然）就得到 AWR 损失。`λ` 是「信任域」宽度：
越小越激进（只模仿高优势动作），越大越接近 BC。

### 推导 2：Expectile 的最优解

对 `X` 的分布，目标 `E[L₂^ζ(X−m)]` 对 `m` 求导令零：

```text
ζ · E[X>m] · (X−m) − (1−ζ) · E[X≤m] · (m−X) = 0
```

即 `m` 是加权均值：`m = [ζE(X|X>m) + (1−ζ)E(X|X<m)]` 的隐式解。
`ζ>0.5` 时大值权重更大，解位于分布上半部。极端 `ζ→1` 趋近最大值，`ζ=0.5` 是均值。
IQL 用 `ζ≈0.7–0.9` 在「接近 max」与「不被极值主导」之间折中。

### 推导 3：IQL 为何不查询 OOD 动作

- Q 的 Bellman 目标是 `r+γV(s')`，不需要对 `s'` 采样动作；
- V 的回归目标 `Q̄(s,a)` 中 `(s,a)` 全部来自数据；
- 策略只在「抽取」阶段用 `exp(β(Q̄−V))` 加权数据动作。
整条链路只读到数据内动作，因此没有 OOD 查询；`V` 的 expectile 隐式承担了 `max` 的角色。

### 推导 4：CQL 惩罚项的梯度直觉

对 `L = α(E_{a~µ}[Q] − E_{a~D}[Q])`：

```text
∂L/∂Q(s,a~µ) = +α      （压低策略/OOD 动作）
∂L/∂Q(s,a~D) = −α      （抬高数据动作）
```

因此 Q 在 OOD 区域被系统性低估、在数据区域被抬高，得到保守下界；`α` 越大越保守，
越大越难「爬出」次优数据。

## 实现映射（HW3）

| 概念 | 文件 | TODO | 验收 |
| --- | --- | --- | --- |
| AWAC critic | `assignments/spring2026/homework3-offline-rl/starter-code/cs224r/critics/awac_critic.py` | clipped double-Q 目标 | 3 seeds 平均回报 |
| AWAC agent | `.../agents/awac_agent.py` | 加权 NLL 策略更新 | antmaze-umaze/medium-diverse |
| IQL critic | `.../critics/iql_critic.py` | expectile loss、V 更新、Q 更新 | `ζ=0.2/0.9` 对比 |
| IQL agent | `.../agents/iql_agent.py` | AWR 策略抽取 | 同上 |
| Stitching | `offline_datasets/pointmass_stitching_dataset.npz` | 对比 Filtered BC | 回报是否超过数据最优 −46 |

运行命令（官方要求 Modal）：

```bash
modal run --detach modal_train_para.py --algo awac \
  --env-name antmaze-umaze-v0 --exp-name awac_antmaze_umaze --use-wandb

modal run --detach modal_train_para.py --algo iql \
  --env-name antmaze-umaze-v0 --iql-expectile 0.9 \
  --exp-name iql_zeta_0.9_umaze --use-wandb
```

## 易错点与反例

- **critic 对策略采样动作做 Bellman 目标（经典 off-policy）**：离线场景会 OOD 爆掉；
- **expectile 实现方向写反**：`ζ>0.5` 应给 `µ>0`（预测偏低）更大权重，最终学上分位数；
- **`V` 忘了 stop-gradient**：`V` 的目标必须 detach；
- **策略更新用了错误的 A**：IQL 的策略权重是 `exp(β(Q−V))`，其中 `Q` 用 target 网络；
- **评估只看训练 loss**：必须看环境评估回报；
- **将 AWAC 的 `a'` 来自当前策略**：这正是它与 IQL 的差别，也是它在困难任务上更脆弱的原因。

## 实验或练习

1. 在 PointMass stitching 数据上跑 IQL 与 Filtered BC，比较最大回报与轨迹图，判断是否出现
   「拼接出数据中不存在的路径」；
2. 扫 IQL expectile `{0.2, 0.5, 0.9}`，画回报曲线，解释趋势；
3. AWAC 与 IQL 在 antmaze-medium 上的差距，从 OOD 动作角度给出 3 句话解释。

## 高频问题

- 问：离线 RL 的核心困难？答：OOD 动作上的价值误差被 bootstrap 放大。
- 问：AWR 与 BC 的区别？答：按优势指数加权，好动作多模仿。
- 问：IQL 的 expectile 在近似什么？答：数据内动作的 `Q` 上分位数，用来代替 `max`。
- 问：为什么 IQL 不需要采样 `a'`？答：用 `V(s')` 做 Bellman 目标，训练只用数据动作。
- 问：CQL 与 IQL 的差别？答：CQL 显式惩罚 OOD 动作价值；IQL 结构性避免查询 OOD。
- 问：什么是 stitching？答：把数据中不同轨迹的优质片段拼成更优策略。

## 面试追问链（由浅入深）

1. 为什么不能直接在离线数据上跑 SAC？→ Bellman 目标采样 `a'~π_θ`，OOD 动作的 Q 无监督，
   高估被 bootstrap 放大。
2. AWR 里 `A` 怎么来？用 MC 还是 TD？→ MC 精度低、TD 又引入 OOD；这正是 IQL 要解决的
   矛盾——用 expectile V 获得「数据内的近似 max」。
3. `ζ` 太小/太大分别发生什么？→ 小：接近均值，策略错过高回报动作；大：被极值/噪声主导，
   V 不稳。
4. 为什么 IQL 的策略抽取仍可能出现 OOD？→ 策略是参数化的，`exp` 加权训练后可能外推；
   但训练信号全部来自数据，外推程度有限。
5. 如何定义 stitching？为什么 Filtered BC 不会 stitching？→ 拼接不同轨迹的片段；
   Filtered BC 只在单条轨迹内模仿，无法组合。
6. 离线 RL 如何评估？→ 环境交互评测（D4RL 协议）或 OPE；只报训练 loss 没有意义。

## 自测题（附要点）

1. 从 KL 正则策略迭代推导 AWR 的最优策略形式。
   - 要点：`π* ∝ π_k exp(A/λ)`；拉格朗日乘子。
2. 手算：`X=[0,1,10]`，`ζ=0.9` 的 expectile 大致落在哪个区间？
   - 要点：大于均值 3.67、靠近 10；具体由加权方程定。
3. 写出 IQL 三个损失并标出哪些量 stop-gradient。
   - 要点：`V`、`V(s')` 在 Q 目标中 detach；`Q̄` 在 V 目标中 detach。
4. 解释 AWAC 在 antmaze-medium 上通常弱于 IQL 的原因。
   - 要点：目标动作来自当前策略，OOD 程度随任务变难而上升。
5. 在 HW3 stitching 实验中，如何判断「拼接」确实发生？
   - 要点：回报超过数据最优（−46），轨迹图显示组合了不同演示段。

## 关联笔记

- 前置：[L6 Q-learning](lecture-06-q-learning.md)、[L8 奖励学习](lecture-08-reward-learning.md)
- 后续：[L17 VLA 的离线 RL 微调](lecture-17-rl-for-vlas.md)（advantage conditioning 同源）

## 本讲小结

离线 RL = 「利用奖励」+「不离开数据支撑」。AWR 用加权模仿，IQL 用 expectile 把 `max` 隐式化，
CQL 用惩罚压 OOD。HW3 的 AWAC/IQL 对比与 PointMass stitching 是理解这些设计的最佳实验。
下一讲追问一个更根本的问题：奖励本身从哪来？

## 延伸阅读

- Kostrikov et al., *Offline RL with Implicit Q-Learning* (IQL), 2021.
- Nair et al., *AWAC*, 2020；Peng et al., *AWR*, 2019.
- Kumar et al., *Conservative Q-Learning* (CQL), 2020.
- Fu et al., *D4RL*, 2020；官方 HW3 handout。
