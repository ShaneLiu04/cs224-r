# 阅读路线

按目标与时间预算组织。每讲复习闭环：不看笔记写 5 个关键词 → 手推一个公式 →
写出关键 shape/目标函数 → 在仓库里找一个实现/实验验证 → 给出一个失效边界。

## 路线一：跟着课程完整学（约 10 周，每周 8–12 小时）

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

## 路线二：作业优先（约 6 周，适合已学过 RL 的人）

1. 只精读 **L2**（HW1 需要）→ 完成 HW1；
2. 精读 **L3–L6** → 完成 HW2；
3. 精读 **L7 + L8** → 完成 HW3；
4. 精读 **L9–L10** → 完成默认项目；
5. 做 **L15–L17** 的泛读，选一个期末项目方向；
6. 用 **L18** 的方法论写 research proposal。

## 路线三：面试速查（2–3 天）

| 时段 | 内容 |
| --- | --- |
| Day 1 上午 | L3 策略梯度完整推导、L4 GAE 递推 |
| Day 1 下午 | L5 PPO clip/SAC 三件套、L6 Double Q 与过估计 |
| Day 2 上午 | L7 离线 RL 三算法对比表、expectile/IQL 公式 |
| Day 2 下午 | L8–L9 偏好学习与 DPO 推导、L10 GRPO/RLOO |
| Day 3 | 公式速查表默写 + 高频问题自问自答 + L18 总结 |

面试常被追问的三个「深水区」：

1. 为什么 policy gradient 需要 `∇log` 换元，环境项为什么消失？
2. IQL 的 expectile 到底在估计什么，为什么能避免 OOD？
3. DPO 与 RLHF 在数学上是什么关系，DPO 的失效模式是什么？

## 路线四：科研入门（1 个季度）

1. 复现三篇论文的最小版本：PPO（HW2）、IQL（HW3）、DPO/IPO（项目）；
2. 选一个 `experiments/topics/` 主题，读完其中 5 篇论文并做一张对比表；
3. 提出可证伪假设，用 `templates/experiment.md` 记录单变量实验；
4. 写 research proposal，征求反馈，再扩展到完整实验。

## 每讲自检问题

学完一讲后，尝试回答（不翻笔记）：

- L1：写出轨迹概率与目标函数；state/observation 的区别与处理方案。
- L2：BC 的两个失败模式；flow matching 的插值、目标与采样。
- L3：完整推导 policy gradient；说明 baseline 无偏。
- L4：写 GAE；解释 `λ` 两端极限。
- L5：写 PPO clip；写 SAC 的 Q 与策略损失。
- L6：写 Q-learning 更新与 Double Q 目标；解释过估计。
- L7：写 AWR/IQL 损失；解释 expectile 与 stitching。
- L8：写偏好奖励损失；解释 reward hacking。
- L9：推导 DPO 损失；比较 RLHF/DPO/IPO。
- L10：写 GRPO/RLOO 优势；解释 pass@k。
- L11：MBPO 为什么用短 rollout；规划算法对比。
- L12：HER 重标流程；GCRL 与多任务关系。
- L13：RL²/MAML/PEARL 三者差异；元 RL 探索难点。
- L14：RND/ICM/UCB；噪声电视问题。
- L15：三个设计问题；高层 DAgger 作用。
- L16：两类失配；RMA 蒸馏。
- L17：VLA RL 三种路线；advantage conditioning。
- L18：算法地图；做研究的六个步骤。
