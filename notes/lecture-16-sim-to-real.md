# Lecture 16 — Sim-to-Real Robot Learning（2026 客座：Guanya Shi）

- 课程版本：Spring 2026（2025 对应讲为 Autonomous Learning + Sim2Real）
- 日期：2026-05-22
- 官方材料：[16_cs224r_sim2real_robot_learning_2026.pdf](../slides/spring2026/16_cs224r_sim2real_robot_learning_2026.pdf) ·
  [2025 版 autonomy](../slides/spring2025/16_cs224r_autonomy_2025.pdf) ·
  [2025 sim2real](../slides/spring2025/17_rl_for_robotics.pdf) ·
  参考：RMA、Domain Randomization、OmniRetarget、SPIDER、FPO
- 状态：初稿

## 摘要

Sim-to-real 是机器人 RL 的落地桥梁：仿真提供并行数据与免费标签，但与真实存在动力学
（摩擦/延迟/电机）与感知（渲染/噪声）两类失配。四条路线：领域随机化（学分布鲁棒策略）、
学习适应（特权信息 → 历史蒸馏，RMA）、real2sim 校准（系统辨识/残差模型/actuator net）、
人类数据（运动重定向 + 跟踪）。PPO 仍是主力，近期趋势是「仿真基础设施 + 少量真机数据微调」
（Sim2Real 4.0）。

## 一页速览卡（面试前 5 分钟）

**必须记住的五件事**

1. 两类 gap：动力学失配与感知失配，必须分别处理。
2. Domain randomization：把参数 `e` 随机化，`x'=f_sim(x,u,e)`，学共享鲁棒策略。
3. 学习适应：RMA 用特权信息训练 teacher，再用历史观测蒸馏出 student。
4. Real2sim：system ID 拟合参数、残差网络补仿真误差、actuator net 建模电机。
5. 人类数据：retargeting（运动学/动力学）+ tracking 策略，是近期人形进度的关键。

**必须能默写的公式**

```text
DR 目标:  max_π E_{e~p(e)} E_{τ~π,e}[ Σ γ^t r_t ]
RMA 蒸馏: L(φ) = E[ || μ_φ( history ) − z_t ||² ],  z 为特权隐变量
残差模型: s' ≈ f_sim(s,a) + g_φ(s,a)
```

**一句话区分**

- DR vs System ID：把不确定性随机化求鲁棒 vs 用真机数据缩小不确定性。
- Teacher-Student vs 非对称 AC：测试时用 student 策略 vs critic 专属特权信息。
- 运动学 vs 动力学重定向：几何可行 vs 物理可行（SPIDER）。

## 学习目标

- 解释为什么用仿真（并行、廉价标签、可复现）以及 sim2real gap 的两类来源；
- 描述 domain randomization、system identification、residual model 三条路线；
- 写出 RMA / asymmetric actor-critic 的特权信息蒸馏流程；
- 理解人类动作数据（retargeting + tracking）在机器人学习中的角色；
- 知道 sim2real 中常用的 RL 算法选择与前沿趋势（Sim2Real 4.0）。

## 先修知识

- L5 PPO/SAC；L12 目标条件；L15 层次化；基础控制（PD、MPC）概念。

## 核心概念

### 1. 机器人仿真器

常见仿真器：MuJoCo（接触、控制）、Isaac Sim/Lab（GPU 并行）、PyBullet、Genesis、
Drake 等。分类维度：刚体/软体、接触模型、渲染真实度、可并行度、可微性。

用仿真的好处：

- 并行收集海量数据（成千上万环境）；
- 奖励/标签（接触力、位姿、真实状态）免费可得；
- 安全、可复现、可 reset。

### 2. Sim2Real 的两类失配

1. **动力学失配**：质量、摩擦、延迟、电机模型、接触参数与真实不同；
2. **感知/视觉失配**：渲染外观、光照、纹理、传感器噪声与真实不同。

### 3. Domain Randomization（DR）

不追求「精确仿真」，而是让参数 `e` 随机化，训练对分布鲁棒的策略：

```text
x_{t+1} = f_sim(x_t, u_t, e),    e ~ p(e)
π(x) 在大量 e 上共享（系统辨识 + 随机化）
```

- 代表：OpenAI 魔方（ADT）、四足/人形运动（Agile But Safe、RPL）；
- 优点：不需要精确建模；缺点：分布设计困难，过宽导致保守、过窄导致不鲁棒；
- 变体：课程化随机化（curriculum DR）、对抗式 DR（ADR）。

### 4. Learning to Adapt

随机化参数不可观测时，策略需要**在线适应**：

- **RMA**：训练阶段用特权信息（接触、地形、外力等）估计隐参数 `z`；
  测试阶段从历史 `(s, a)` 用监督学习预测 `z`，喂给策略；
- **Asymmetric Actor-Critic**：critic 在仿真中用特权信息，actor 只用可观测信息；
- 学生网络可以在 latent 空间学（RMA 的核心技巧），避免直接拟合不可观测量。

### 5. Real2Sim 与残差模型

- **System ID**：用真机数据拟合仿真参数（SPI-Active：采样 + 主动探索）；
- **Residual dynamics**：仿真预测 + 学习残差 `f_real ≈ f_sim + g_φ`；
  或用真实数据校准 actuator net（电机力矩模型）；
- 这类方法让仿真更准，减少对随机化的依赖。

### 6. 人类数据：Retargeting + Tracking

- 动捕/视频 → **运动重定向**到机器人关节（几何/运动学层，如 OmniRetarget）；
- 训练策略跟踪参考动作（tracking policy），再用 RL 微调；
- **SPIDER** 等做动力学级重定向，保证可行性；
- 数据规模化（人形、灵巧手）是近期进展主因之一。

### 7. Sim2Real 常用的 RL 算法

- **PPO**：on-policy、稳定、可大规模并行仿真，仍是主力；
- **FPO / FPO++**：面向 flow matching 策略的 policy gradient；
- **BFM-Zero**：无监督 RL + forward-backward 表示；
- 常见组合：PPO 训 tracking/运动策略 + 课程 DR + 蒸馏到可部署网络。

### 8. 历史与趋势（Sim2Real 4.0）

控制社区做 sim2real 已数十年：经典 MPC/system ID → 深度 RL + DR → 特权学习/残差 →
现在进入「在线 + 真实数据 + 基础模型」的 4.0 阶段：预训练策略 + 少量真机数据微调，
仿真成为训练基础设施的一部分而非终点。

## 关键推导：RMA 的蒸馏目标

训练时特权编码器 `φ(privileged)` 与策略联合训练；蒸馏阶段通过监督回归
`ẑ = μ_φ(s_{t-k:t}, a_{t-k:t})`，让历史观测预测 `z`：

```text
L_distill(φ) = E[ || μ_φ(history) − z_t ||² ]
```

测试时不再需要特权信息。关键点：预测的是 latent `z`（而非具体物理参数），
更易泛化且对噪声鲁棒。

## 深入推导

### 推导 1：DR 是期望鲁棒还是最坏情况鲁棒

DR 优化 `max_π E_{e~p(e)}[J(π,e)]`，即对参数分布求**期望**。若分布尾部有灾难场景，
期望会稀释它。最坏情况鲁棒优化 `max_π min_{e∈E}[J(π,e)]` 更保守但更安全。
实践折中：在 `p(e)` 中显式加入失效模式（课程化 DR、ADR 用对抗生成 `e`）。

### 推导 2：RMA 蒸馏的信息论视角

特权信息 `p_t`（质量、摩擦、外力）不可观测。RMA 不直接预测 `p_t`，而是训练
隐变量 `z_t=E(p_t)` 作为策略输入，再用历史 `h_t=(s,a)_{t-k:t}` 预测 `z_t`：

```text
L = E[ || μ_φ(h_t) − z_t ||² ] + L_RL(π(a | s, z))
```

历史是特权信息的**充分统计量**的近似：只要历史中包含与 `p_t` 相关的信息，
蒸馏就能恢复它；否则无论网络多大都学不到。这解释了为什么 RMA 的观测历史长度很重要。

### 推导 3：残差模型的偏差-方差

`ŝ' = f_sim(s,a) + g_φ(s,a)`。若 `f_sim` 已解释大部分动力学，`g` 只需拟合小残差：
目标方差小、学习更稳；若 `f_sim` 很差，`g` 退化为完整模型学习，失去先验优势。
同时残差模型在真机数据分布外仍不可信，需要与 DR/约束配合。

### 推导 4：actuator net 的辨识方程

电机力矩 `τ = h(q,q̇,command)` 通常非线性。用真机轨迹收集 `(q,q̇,q̈)`，
由刚体动力学反推 `τ`（需要标定质量/惯量），再监督回归

```text
L(φ) = E[ || h_φ(q,q̇,command) − τ_measured ||² ]
```

替代方案：把 `h_φ` 直接放进仿真做可微学习，让下游任务损失反向校准电机模型。

## 实现映射

| 概念 | 材料 | 说明 |
| --- | --- | --- |
| MuJoCo 仿真 | `assignments/spring2023/homework2-online-rl/starter-code/mujoco_mpc/` | 接触/控制仿真 |
| PPO 并行训练 | HW2 的 PPO（Modal 上并行环境） | sim2real 常用同款 |
| 人类数据/retarget | 期末项目方向 | OmniRetarget 等开源工具 |

## 易错点与反例

- **DR 分布拍脑袋设计**：过宽使策略保守，过窄无法迁移；需要课程/真实数据校准；
- **特权信息泄漏到测试**：actor 只能用可观测信息，critic 可以；
- **只对齐视觉不对齐动力学**：反过来也一样，两个 gap 要分别处理；
- **tracking 参考动作动力学不可行**：需动力学级重定向或 relax 约束；
- **把仿真成功当真实成功**：必须有真机评测与失效分析。

## 实验或练习

1. 用 2023/2025 HW2 的 MJPC 环境做奖励 shaping 与参数随机化实验；
2. 阅读 RMA 论文，画出「特权编码 → 策略 → 蒸馏」三阶段数据流；
3. 选一个 MuJoCo locomotion 环境，比较无 DR / 温和 DR / 激进 DR 的鲁棒性（扰动测试）。

## 高频问题

- 问：sim2real 的两类失配？答：动力学（摩擦/延迟/电机）与感知（渲染/噪声）。
- 问：DR 的核心思想？答：不追求精确参数，而是在参数分布上学鲁棒策略。
- 问：特权信息怎么用？答：训练时给 critic/teacher，蒸馏到只用观测的 student。
- 问：为什么 PPO 仍是主力？答：并行仿真、稳定、对奖励设计容忍度高。
- 问：人形机器人近期进展的关键？答：大规模并行仿真 + 人类动作数据 + 课程随机化。

## 面试追问链（由浅入深）

1. 为什么不能只靠提高仿真精度？→ 接触/摩擦/软体的建模误差与传感器噪声无法完全消除；
   且精确建模成本高、不可微、难以泛化到新场景。
2. DR 的参数分布应该多宽？→ 覆盖真实参数但不宽到让策略保守；
   可用真机数据校准（curriculum/ADR）。
3. 特权信息在测试时怎么获得？→ 不可获得；RMA 用历史观测预测隐参数，
   或非对称 AC 让 critic 专用（critic 只在训练用）。
4. 为什么 PPO 在 sim2real 中这么主流？→ 大规模并行仿真 + on-policy 稳定 +
   对奖励 shaping 容忍；SAC 类在真机微调时更省样本。
5. 人类动作数据的价值在哪？→ 提供难以用奖励描述的技能先验与运动多样性；
   需要重定向解决形态差异。
6. 如何判断一个策略可以上真机？→ 多随机种子 + 扰动/地形泛化测试 +
   失效模式分析；不要只看仿真平均回报。

## 自测题（附要点）

1. 区分两类 sim2real 失配并各举两个来源。
   - 要点：动力学（摩擦、延迟、电机）；感知（纹理、光照、相机噪声）。
2. 写出 DR 的优化目标，说明它与普通 RL 目标的区别。
   - 要点：外层对参数分布求期望。
3. 画出 RMA 三阶段流程图并标出每阶段的输入。
   - 要点：特权编码 → 策略训练 → 历史蒸馏。
4. 解释 actuator net 为什么需要扭矩标签，以及无标签替代思路。
   - 要点：回归真实力矩；可用无监督一致性/状态预测替代。
5. 设计一个测试 DR 策略鲁棒性的方案。
   - 要点：参数网格外推、质量/摩擦扰动、延迟注入、地形变化。

## 关联笔记

- 前置：[L11 模型类 RL](lecture-11-model-based-rl.md)、
  [L12 GCRL](lecture-12-multi-task-goal-conditioned-rl.md)
- 后续：[L17 VLA 与 RL 后训练](lecture-17-rl-for-vlas.md)

## 本讲小结

Sim2real 是机器人 RL 的落地桥梁：DR、学习适应、real2sim 校准、人类数据与策略蒸馏
是四块基石。下一讲看看当机器人已有强预训练 VLA 时，RL 应该怎么做。

## 延伸阅读

- Tobin et al., *Domain Randomization*, 2017；Kumar et al., *RMA*, 2021.
- Tan et al., *Sim-to-Real: Agile Locomotion*, 2018；Hwangbo et al., *Actuator Net*, 2019.
- OmniRetarget, SPIDER, FPO 相关最新论文（见 slides 引文）。
- 官方 slides：`slides/spring2026/16_cs224r_sim2real_robot_learning_2026.pdf`。
