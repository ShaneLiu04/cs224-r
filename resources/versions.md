# 版本差异：2023 vs 2025 vs 2026

## 课程结构

| 维度 | Spring 2023 | Spring 2025 | Spring 2026 |
| --- | --- | --- | --- |
| 讲师 | Chelsea Finn + Karol Hausman | Chelsea Finn | Chelsea Finn + 客座 |
| 作业数 | 4 | 4 | 3 |
| 考试 | 无 | 无 | 期中（25%） |
| 项目权重 | 50% | 50% | 35% |
| 视频 | 无公开 | 有完整录像 | Canvas（未公开） |
| Exploration 讲 | 无独立讲 | 有（14） | 无 |
| LLM RL | 无 | 有（9–10） | 有（9–10） |

## 讲次映射

| 主题 | 2023 | 2025 | 2026 |
| --- | --- | --- | --- |
| 模仿学习 | cs224r_imitation | 02 | 02 |
| 策略梯度 | cs224r-mdp-pg | 03 | 03 |
| Actor-Critic | cs224r-actor-critic-split | 04 | 04 |
| Q-learning | cs224r-q-learning + practical deep RL | 05–06 | 05–06 |
| 离线 RL | offline_rl_1/2 | 07 | 07 |
| 奖励学习 | cs224r_reward_learning | 08 | 08 |
| RLHF/推理 | 无 | 09–10 | 09–10 |
| 模型类 RL | cs224r_mbrl | 11 | 11 |
| 多任务/GCRL | cs224r-mtrl | 12 | 12 |
| 元学习 | metarl1/2 | 13 | 13 |
| 探索 | 无 | 14 | 无 |
| 层次化/技能 | skill-discovery | 15 | 15 |
| 机器人 | RLLeggedRobots / autonomy | 16–17 | 16–17（sim2real / VLA） |
| 前沿 | review_frontiers | 18 | 18 |

## 作业接口变化

| 作业 | 2023 | 2025 | 2026 |
| --- | --- | --- | --- |
| HW1 | BC + DAgger（MuJoCo，`.pkl` 专家数据） | 同 2023 | **Flappy Bird**：BC 回归 + flow matching + DAgger |
| HW2 | MJPC 奖励设计 + actor-critic/Q-learning | 同 2023 | **Gridworld Q-learning + PPO + off-policy AC**（Meta-World hammer） |
| HW3 | IQL + CQL（自建 env） | IQL + CQL（更新版） | **AWAC + IQL**（D4RL AntMaze + PointMass stitching） |
| HW4 | GCRL + Meta-RL | GCRL + Meta-RL | 无（主题进入项目） |

## 迁移提示

- 课程编号不同：2025 的视频第 6 讲对应 2026 slides 06，但 2023 的 practical deep RL 已合并/删除；
- 代码接口不同：不要用 2025 的 starter code 做 2026 作业；
- 算法术语一致：AWR/AWAC、IQL、GRPO/RLOO 的公式跨版本通用。

## 版本选择建议

| 目标 | 建议 |
| --- | --- |
| 完成当前作业 | 2026 |
| 跟视频系统学 | 2025 |
| 补 exploration/技能发现 | 2025/2023 |
| 想做 GCRL/Meta-RL 动手 | 2025/2023 HW4 |
