# 阅读路线

按目标与时间预算组织。每讲复习闭环：不看笔记写 5 个关键词 → 手推一个公式 →
写出关键 shape/目标函数 → 在仓库里找一个实现/实验验证 → 给出一个失效边界。

## 0. 前置关系图

```text
L1 MDP ──► L2 模仿学习 ──► L3 策略梯度 ──► L4 Actor-Critic ──┬─► L5 Off-policy AC（PPO/SAC）
   │                                                         └─► L6 Q-learning
   │                                                                  │
   │                        ┌─────────────────────────────────────────┘
   │                        ▼
   │                 L7 离线 RL（AWAC/IQL/CQL）──► L8 奖励学习 ──► L9 RLHF/DPO ──► L10 LLM 推理 RL
   │                                                                                   │
   └──────────────► L12 多任务/GCRL ──► L13 元学习 ◄── L14 探索 ◄──────────────────────┘
                          │                 │
                          ▼                 ▼
                    L15 层次化 ◄──── L11 模型类 RL ──► L16 Sim-to-Real ──► L17 VLA RL ──► L18 前沿
```

依赖关系：L3 是 L4–L7 的前提；L7 是 L8–L9 的前提；L12/L13/L14 构成一组；
L15–L17 是应用层，可最后读。

## 1. 每讲掌握标准与时间预算

| # | 笔记 | 精读 | 必须能默写 | 判断「学会了」的标准 |
| ---: | --- | --- | --- | --- |
| 1 | [L1](lecture-01-intro-mdps.md) | 40min | 轨迹概率、`J(θ)` | 能解释为何期望对策略求导 |
| 2 | [L2](lecture-02-imitation-learning.md) | 90min | BC loss、FM 插值与速度、DAgger 数据集 | 能画多模态分布并解释 MSE 坍缩 |
| 3 | [L3](lecture-03-policy-gradients.md) | 120min | `∇J` 推导、`R_t`、baseline 无偏 | 能独立完成四步推导并解释每步 |
| 4 | [L4](lecture-04-actor-critic.md) | 90min | TD 残差、GAE 递推、`returns=adv+values` | 能手算 3 步 GAE |
| 5 | [L5](lecture-05-off-policy-actor-critic.md) | 120min | PPO clip、SAC Q/策略损失 | 能解释 replay 的三处修正 |
| 6 | [L6](lecture-06-q-learning.md) | 90min | Q 更新、Double Q、压缩论证 | 能推导过估计并解释 Double Q |
| 7 | [L7](lecture-07-offline-rl.md) | 150min | AWR、expectile、IQL 三式、CQL | 能解释 OOD 误差如何被放大 |
| 8 | [L8](lecture-08-reward-learning.md) | 60min | BT 损失、RLHF 循环 | 能举出分类器被 exploit 的例子 |
| 9 | [L9](lecture-09-rlhf-preference-optimization.md) | 120min | 闭式 `π*`、DPO 损失与梯度 | 能完整推导 DPO 并说明 `Z(x)` 消去 |
| 10 | [L10](lecture-10-rl-llm-reasoning.md) | 90min | GRPO/RLOO 优势、pass@k | 能证明 RLOO 无偏、构造零优势组 |
| 11 | [L11](lecture-11-model-based-rl.md) | 60min | 模型损失、MBPO 流程 | 能解释短 rollout 的原因 |
| 12 | [L12](lecture-12-multi-task-goal-conditioned-rl.md) | 90min | 条件 MDP、HER 重标 | 能计数 HER 候选并说明偏差 |
| 13 | [L13](lecture-13-meta-rl.md) | 90min | MAML 内外循环、PEARL ELBO | 能解释后验采样为何是探索 |
| 14 | [L14](lecture-14-exploration.md) | 60min | UCB、RND、最大熵 | 能解释噪声电视问题 |
| 15 | [L15](lecture-15-hierarchical-rl-il.md) | 60min | options、三个设计问题 | 能设计两层系统并指出接口 |
| 16 | [L16](lecture-16-sim-to-real.md) | 60min | DR 目标、RMA 蒸馏 | 能区分两类 gap 与四条路线 |
| 17 | [L17](lecture-17-rl-for-vlas.md) | 60min | 优势条件 BC、残差策略 | 能说明 IL 平台期的原因 |
| 18 | [L18](lecture-18-frontiers.md) | 60min | 算法五要素、研究模板 | 能写出项目提案的四段式 |

合计约 26 小时精读；若只看速览卡与自测题，约 3 小时。

## 2. 路线一：完整学（约 10 周，每周 8–12 小时）

| 周 | 内容 | 产出 |
| --- | --- | --- |
| 1 | L1–L2 + HW1 前半（BC） | Flappy Bird BC easy 跑通 |
| 2 | L2 后半（FM/DAgger）+ HW1 报告 | 四条命令 + 学习曲线 |
| 3 | L3–L4 + HW2 P1 | gridworld 三场景 |
| 4 | L5–L6 + HW2 P2 | PPO ≥25% |
| 5 | L5 复习 + HW2 P3 | off-policy AC ≥90% + UTD 消融 |
| 6 | L7 + HW3 前半 | AWAC 跑通 |
| 7 | L7 复习 + HW3 后半 | IQL + stitching |
| 8 | L8–L10 + Default Project 前半 | SFT + IPO |
| 9 | L10 复习 + RLOO | pass@k 对比 |
| 10 | L11–L18 泛读 + 项目报告 | 最终报告与 poster |

## 3. 路线二：作业优先（约 6 周）

1. L2 → HW1（1.5 周）；
2. L3–L6 → HW2（2 周）；
3. L7–L8 → HW3（1.5 周）；
4. L9–L10 → Default Project（1 周，之后滚动训练）；
5. L15–L17 泛读选项目方向 + L18 写 proposal。

## 4. 路线三：面试速查（2–3 天）

| 时段 | 内容 | 目标 |
| --- | --- | --- |
| Day 1 上午 | L3 推导 + L4 GAE | 白板默写 |
| Day 1 下午 | L5 PPO/SAC + L6 Double Q | 公式 + 直觉 |
| Day 2 上午 | L7 三算法 + L8 偏好 | 对照表 |
| Day 2 下午 | L9 DPO 推导 + L10 GRPO/RLOO | 推导 + pass@k |
| Day 3 | 公式速查表 + 高频题自答 + L11–L18 速览卡 | 全场串讲 |

面试深水区三连（务必准备）：

1. 为什么 policy gradient 需要 `∇log` 换元，环境项为何消失？
2. IQL 的 expectile 在估计什么，为什么能避免 OOD？
3. DPO 与 RLHF 的数学关系，DPO 的失效模式？

## 5. 路线四：科研入门（一个季度）

1. 复现三个最小版本：PPO（HW2）、IQL（HW3）、DPO/IPO（项目）；
2. 选一个 [experiments/topics](../experiments/topics/) 主题，读 5 篇论文并做对比表；
3. 用 [templates/experiment.md](../templates/experiment.md) 记录 3 个单变量实验；
4. 用 L18 的四段式写 proposal，找反馈后扩展为完整研究。

## 6. 复习闭环（每讲执行一遍）

1. **回忆**：合上笔记写 5 个关键词与 1 个公式；
2. **手推**：推导本讲最核心的公式（见每讲「深入推导」）；
3. **落地**：打开对应作业/实验卡片，说清每个 TODO 的输入输出；
4. **找边界**：举一个该方法的失效场景（如「λ=1 高方差」「ζ 太大被极值主导」）；
5. **串联**：在 [关联笔记](lecture-18-frontiers.md#关联笔记) 处补一条跨讲连接。

## 7. 进度追踪模板

在 [PROGRESS.md](../PROGRESS.md) 中为每讲维护三列：

```text
L07 离线 RL   [ ] 精读  [ ] 公式默写  [ ] 实验验证（HW3）  [ ] 可讲 5 分钟
```

「可讲 5 分钟」的标准：能用速览卡的结构，从问题、公式、算法、失效边界四个层次讲清楚一讲。
