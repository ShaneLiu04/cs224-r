# Spring 2025 作业（存档）

4 次作业 + 项目；与 2026 版相比多 HW4、内容更偏 MuJoCo 经典流程。
**有完整官方录像**（见 [slides/README.md](../../slides/README.md)）。

| 作业 | 主题 | 目录 | 主要内容 |
| --- | --- | --- | --- |
| HW1 | Imitation Learning | [homework1-imitation/](homework1-imitation/) | MuJoCo 专家数据（Ant/HalfCheetah/Hopper/Walker2d）上的 BC 与 DAgger |
| HW2 | Online RL | [homework2-online-rl/](homework2-online-rl/) | MuJoCo MPC（MJPC）奖励设计 + Meta-World 在线 RL（含 72MB 演示与仿真资源） |
| HW3 | Offline RL | [homework3-offline-rl/](homework3-offline-rl/) | IQL 与 CQL；含 `starter-code`（updated 版）与 `starter-code-v1` 两个版本 |
| HW4 | Goal-Conditioned & Meta-RL | [homework4-goal-meta-rl/](homework4-goal-meta-rl/) | HER（goal_conditioned_rl）与 RNN 元学习（meta_rl） |

项目指南：[project/](project/)（Default 与 Custom 两版）。

## 与 2026 的关系

- 2025 的 HW1（BC+DAgger）与 HW2 的在线 RL 被 2026 重写：Flappy Bird 替代 MuJoCo IL，
  PPO/off-policy AC 替代 MJPC；HW3 从 IQL+CQL 改为 AWAC+IQL；
- 2025 HW4 的主题（HER、Meta-RL）在 2026 只出现在课堂与项目中；
- 做作业请使用对应版本的 handout 与 starter code，接口互不兼容。

## 使用建议

- 想补 GCRL/Meta-RL 动手经验：用 HW4；
- 想理解「奖励设计对行为的影响」：HW2 的 MJPC 部分；
- 想比较 CQL 与 IQL：HW3 的 `starter-code-v1` 与 updated 版。
