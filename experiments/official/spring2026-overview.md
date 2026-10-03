# Spring 2026 课程与作业总览

## 课程结构

- 授课：Wed/Fri 09:30–11:00，Chelsea Finn（部分客座：Archit Sharma、Noam Brown、Guanya Shi）；
- 评分：HW1 10% + HW2 15% + HW3 15% + 期中 25% + 项目 35%；
- 无 HW4；无 exploration 讲；
- 项目分默认（Countdown RL）与自定义；均有 proposal / milestone / poster / report。

## 讲次与作业的对应

| 讲次 | 主题 | 对应作业/实验 |
| --- | --- | --- |
| L1–L2 | MDP、模仿学习 | HW1 |
| L3–L6 | 策略梯度、AC、off-policy AC、Q-learning | HW1/Q-learning 热身、HW2 |
| L7–L8 | 离线 RL、奖励学习 | HW3 |
| L9–L10 | RLHF/DPO、LLM 推理 RL | Default Project |
| L11–L17 | MBRL、多任务、元学习、层次、sim2real、VLA | 项目选题 |
| L18 | 前沿与研究方法 | 项目报告 |

## 三份作业的难度与工作量（经验估计）

| 作业 | 代码量 | 主要难点 | 时间 |
| --- | --- | --- | --- |
| HW1 | 小（4 个文件，7 个 TODO） | flow matching 实现、DAgger 确定性专家设计 | 1–2 天 |
| HW2 | 中（3 个文件） | GAE/PPO 正确性、critic ensemble 与 UTD 稳定性 | 3–5 天（含训练） |
| HW3 | 中（5 个文件） | expectile 方向、Bellman 目标、Modal/D4RL 流程 | 4–6 天（含训练） |
| Project | 大 | 大规模训练管线、评估与扩展实验 | 3–4 周 |

## 官方材料位置

- 讲义：[`slides/spring2026/`](../../slides/spring2026/)
- 作业：[`assignments/spring2026/`](../../assignments/spring2026/)
- 其他：[`materials/spring2026/`](../../materials/spring2026/)

## 版本提示

2026 与 2025 的作业接口不同（详见 [resources/versions.md](../../resources/versions.md)）：
不要混用 2025 的 handout/代码与 2026 的模板。
