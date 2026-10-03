# Spring 2023 作业（存档）

4 次作业 + 项目；与 Spring 2025 结构基本一致（2025 是其修订版）。
课程讲义含 `practical deep RL`、`RL in the real world`、`skill discovery` 等专题。

| 作业 | 主题 | 目录 | 主要内容 |
| --- | --- | --- | --- |
| HW1 | Imitation Learning | [homework1-imitation/](homework1-imitation/) | BC 与 DAgger（MuJoCo 专家 `.pkl` 数据） |
| HW2 | Online RL | [homework2-online-rl/](homework2-online-rl/) | MJPC 奖励设计 + actor-critic/Q-learning（含 MuJoCo MPC 源码与演示视频） |
| HW3 | Offline RL | [homework3-offline-rl/](homework3-offline-rl/) | IQL 与 CQL（含 PointMass、Reacher 等自定义环境） |
| HW4 | Goal-Conditioned & Meta-RL | [homework4-goal-meta-rl/](homework4-goal-meta-rl/) | Bit-flip HER 与 meta_rl 框架 |

项目指南：[project/CS224R_Project_Guidelines.pdf](project/CS224R_Project_Guidelines.pdf)。

## 注意

- HW2 的 `mujoco_mpc/` 是完整 MJPC 源码 + 演示视频（约 90MB），主要作为理解规划器的材料；
- 数据接口（Gym/MuJoCo 版本）较旧，运行需要匹配版本；
- 如需完整复现，优先参考 2025 版（官方提供了更新的环境与说明）。
