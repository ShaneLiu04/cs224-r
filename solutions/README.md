# 个人实现

作者在修完/旁听课程后完成的作业实现、多种子实验与报告，用于自学与求职展示。
官方 starter code 保留在 `assignments/` 下不改动，这里是复制后的独立工作区。

| 作业 | 主题 | 说明 |
| --- | --- | --- |
| 2026 HW1 | BC、Flow Matching、DAgger | [README](spring2026/homework1-imitation/README.md) |
| 2026 HW2 | Q-learning、PPO+GAE、高 UTD actor-critic | [README](spring2026/homework2-online-rl/README.md) |
| 2026 HW3 | AWAC、IQL、拼接 | [README](spring2026/homework3-offline-rl/README.md) |
| 2026 Default Project | Countdown：SFT、IPO/DPO、RLOO/GRPO、pass@k | [README](spring2026/default-project/README.md) |
| 2025 HW3 | IQL、CQL、RND 探索数据 | [README](spring2025/homework3-offline-rl/README.md) |
| 2025 HW4 | HER、RL² vs DREAM | [README](spring2025/homework4-goal-meta-rl/README.md) |

每个作业目录结构一致：

```text
<homework>/
├── code/                 # starter 副本 + 实现
├── tests/ 或 code/tests/ # 单元测试
├── analysis/             # 从原始指标生成图和 LaTeX 宏（2025 共用 ../analysis/）
├── artifacts/            # raw 指标与 figures
└── reports/              # assignment_report/ 与 interview_guide/（XeLaTeX，PDF 已提交）
```

运行全部测试：`./scripts/run-all-tests.ps1`（各套件依赖不同，缺包时显示 `ERROR (deps)`）。

## 学术诚信

CS224R honor code 禁止参考他人解答、禁止公开自己的课程解答。如果你正在修读该课程，
请不要阅读或使用本目录；这里的代码不是、也不应被当作课程提交。
