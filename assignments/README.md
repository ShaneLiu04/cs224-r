# 作业材料总览

本目录是官方作业的镜像：题面（PDF/LaTeX 模板）+ starter code（含专家数据、演示、配置）。
**不含答案**。每份作业目录内有一份我写的 `README.md` 作为学习指南。

## 版本与作业

| 版本 | 作业 | 主题 | 关键实现 | 指南 |
| --- | --- | --- | --- | --- |
| Spring 2026 | HW1 | Imitation Learning（Flappy Bird） | BC 回归 / flow matching / DAgger | [指南](spring2026/homework1-imitation/README.md) |
| Spring 2026 | HW2 | Online RL（gridworld + Meta-World hammer） | tabular Q / PPO+GAE / off-policy AC | [指南](spring2026/homework2-online-rl/README.md) |
| Spring 2026 | HW3 | Offline RL（D4RL AntMaze + PointMass） | AWAC / IQL / stitching | [指南](spring2026/homework3-offline-rl/README.md) |
| Spring 2026 | Project | Countdown 推理 RL | SFT / IPO / RLOO / pass@k | [指南](spring2026/default-project/README.md) |
| Spring 2025 | HW1–HW4 | MuJoCo BC+DAgger / MJPC+在线 RL / IQL+CQL / GCRL+Meta-RL | 存档 | [总览](spring2025/README.md) |
| Spring 2023 | HW1–HW4 | 与 2025 基本一致 | 存档 | [总览](spring2023/README.md) |

## 我该怎么用

1. 先读对应目录的 `README.md`（TODO 地图 + 实现顺序 + 调试清单 + 验收标准）；
2. 打开官方 handout，对照目录里的 starter code 找到 `TODO` / `YOUR CODE HERE`；
3. **独立实现**，不要参考网上他人解答（honor code）；
4. 自己的实现在 `solutions/spring2026/<homework>/` 下另建目录，不要直接改 `starter-code/`
   （保持官方材料干净，便于随时重置）。

## 计算资源

- HW1（2026）：本地 CPU 可跑 Flappy Bird；
- HW2/HW3（2026）：官方要求 Modal 计算 + W&B 记录（课程曾提供 credits）；
- Default Project：本地 GPU 或 Modal（H100）。

## 目录结构约定

每个作业目录：

```text
homeworkX-name/
├── README.md                       # 我写的学习指南
├── CS224R_20XX_Homework_X.pdf      # 官方题面
├── CS224R_20XX_Homework_X.tex      # LaTeX 报告模板
├── starter-code/                   # 官方 starter code（已解压，清理过缓存）
└── template/                       # 旧版作业的报告模板（如有）
```
