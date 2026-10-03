# CS224R Experiments 学习资料库

以 Spring 2026 的三次作业、默认项目与历史 HW4 为骨架，把题目要求、核心论文、工具文档、
社区材料与本仓库的实验卡片组织成可检索的自学知识库。

> Official-first。先读本地 handout 并独立实现，再打开标为 spoiler 的社区材料。
> 本目录只提供原创摘要、引用与链接，不是答案集，也不复制第三方内容。

## 四种入口

### 按作业

- [Spring 2026 总览](official/spring2026-overview.md)
- [HW1 Imitation Learning](official/hw1-imitation.md)
- [HW2 Online RL](official/hw2-online-rl.md)
- [HW3 Offline RL](official/hw3-offline-rl.md)
- [Default Project](official/default-project.md)

### 按主题

- [模仿学习与生成式策略](topics/imitation-learning.md)
- [策略梯度与 Actor-Critic](topics/policy-gradients-actor-critic.md)
- [值方法与离线 RL](topics/value-based-and-offline-rl.md)
- [奖励与偏好学习](topics/reward-and-preference-learning.md)
- [LLM 的 RL](topics/llm-rl.md)
- [模型类 RL 与规划](topics/model-based-and-planning.md)
- [多任务、元学习与探索](topics/multi-task-meta-exploration.md)
- [层次化与机器人](topics/hierarchy-and-robotics.md)

### 按社区资料

- [社区资料使用规则](community/README.md)
- [HW1](community/hw1.md) · [HW2](community/hw2.md) · [HW3](community/hw3.md) ·
  [Project/其他](community/project.md)

### 按可复现实验

- [HW1：BC / Flow Matching / DAgger](catalog/hw1-imitation.md)
- [HW2：Q-learning / PPO / Off-policy AC](catalog/hw2-online-rl.md)
- [HW3：AWAC / IQL / Stitching](catalog/hw3-offline-rl.md)
- [Project：SFT / IPO / RLOO](catalog/default-project.md)

## 资源预算路线

- **CPU only**：Flappy Bird 的 BC/DAgger 小规模、论文精读、Q-learning gridworld；
- **单张 GPU**：Flappy Bird flow matching、默认项目的小模型 SFT、玩具 RL 环境；
- **Modal / 多卡**：HW2 的 PPO/AC、HW3 的 AWAC/IQL、默认项目 RLOO 全量训练。
  没有课程 credits 时需自付 Modal 费用（见 tools 文档中的替代方案）。

## Provenance

- [`SOURCES.yaml`](SOURCES.yaml)：URL、作者、版本、主题、license 与可信度；
- [`LICENSES.md`](LICENSES.md)：镜像与引用政策；
- [`source-report.md`](source-report.md)：来源统计与检查结果；
- 官方快照 commit 见根目录 [`UPSTREAM.md`](../UPSTREAM.md)。

大型项目报告 PDF、训练产物与视频不入库；只保存配置、命令、汇总指标与小型图表。
