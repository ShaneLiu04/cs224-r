# CS224R 十八讲自学笔记

原创中文笔记，保留英文术语；不逐页翻译，也不替代官方 lecture materials。
每讲按统一结构深度组织，面向「能推导、能比较、能落地、能应对追问」的学习目标。

## 课程索引

| # | 日期（2026） | 官方主题 | 笔记 | 作业/实践 |
| ---: | --- | --- | --- | --- |
| 1 | 04-01 | Course Intro, MDPs | [L1](lecture-01-intro-mdps.md) | A1 环境与 MDP |
| 2 | 04-03 | Imitation Learning | [L2](lecture-02-imitation-learning.md) | A1 全部 |
| 3 | 04-08 | Policy Gradients | [L3](lecture-03-policy-gradients.md) | A2 基础 |
| 4 | 04-10 | Actor-Critic Methods | [L4](lecture-04-actor-critic.md) | A2 GAE |
| 5 | 04-15 | Off-Policy Actor-Critic | [L5](lecture-05-off-policy-actor-critic.md) | A2 PPO/AC |
| 6 | 04-17 | Q-Learning | [L6](lecture-06-q-learning.md) | A2 P1 gridworld |
| 7 | 04-22 | Offline RL | [L7](lecture-07-offline-rl.md) | A3 AWAC/IQL |
| 8 | 04-24 | Reward Learning | [L8](lecture-08-reward-learning.md) | A3 / 项目 |
| 9 | 04-29 | RLHF & Preference Optimization | [L9](lecture-09-rlhf-preference-optimization.md) | 项目 IPO |
| 10 | 05-01 | RL for LLM Reasoning | [L10](lecture-10-rl-llm-reasoning.md) | 项目 RLOO |
| 11 | 05-06 | Model-Based RL | [L11](lecture-11-model-based-rl.md) | 项目选题 |
| 12 | 05-08 | Multi-Task & Goal-Conditioned RL | [L12](lecture-12-multi-task-goal-conditioned-rl.md) | 旧版 HW4 |
| 13 | 05-13 | Meta-RL | [L13](lecture-13-meta-rl.md) | 旧版 HW4 |
| 14 | — | Exploration（2025 版补课） | [L14](lecture-14-exploration.md) | 项目选题 |
| 15 | 05-20 | Hierarchical RL & IL | [L15](lecture-15-hierarchical-rl-il.md) | 项目选题 |
| 16 | 05-22 | Sim-to-Real（客座：Guanya Shi） | [L16](lecture-16-sim-to-real.md) | 项目选题 |
| 17 | 05-27 | RL for VLAs | [L17](lecture-17-rl-for-vlas.md) | 项目选题 |
| 18 | 05-29 | Frontiers & How to Research | [L18](lecture-18-frontiers.md) | 项目报告 |

> 2026 版课程没有第 14 讲；L14 依据 2025 版 slides 整理，用于补全探索主题。

## 每讲结构（深度版）

| 部分 | 作用 |
| --- | --- |
| 摘要 | 3–5 句话概括问题、核心结果与主线张力 |
| 一页速览卡 | 面试前 5 分钟：五件事 + 公式 + 一句话区分 |
| 学习目标 / 先修知识 | 定位本讲在课程中的位置 |
| 核心概念 | 概念、公式、shape、直觉 |
| 关键推导 | 主要公式的推导主线与边界条件 |
| 深入推导（L3–L7、L9–L13） | 完整数学步骤、证明、偏差来源与代码含义 |
| 实现映射 | 概念 → 作业文件 → 验收标准 |
| 易错点与反例 | 常见 bug 与失效场景 |
| 实验或练习 | 可执行的最小实验设计 |
| 高频问题 | 基础问答 |
| 面试追问链 | 由浅入深的连续追问与回答框架 |
| 自测题 | 带要点的手算/证明/设计题 |
| 关联笔记 | 前后讲与概念复用关系 |
| 本讲小结 / 延伸阅读 | 收束与论文入口 |

## 官方链接

- 课程主页：https://cs224r.stanford.edu/
- 课程网站源码（材料快照来源）：https://github.com/cs224r/cs224r.github.io
- 2025 视频：https://www.youtube.com/playlist?list=PLoROMvodv4rPwxE0ONYRa_itZFdaKCylL
- 官方 head TA 笔记（Jubayer Ibn Hamid）：
  https://jubayer-ibn-hamid.github.io/data/RL_Notes__final_.pdf

## 辅助导航

- [术语与符号表 + 公式速查 + 易混对照](GLOSSARY.md)
- [按目标/时间预算的阅读路线](READING-ROADMAP.md)
- [论文、工具与社区资料库](../experiments/README.md)
- [三代作业材料](../assignments/README.md)

## 笔记使用法

1. 首读「摘要 → 一页速览卡 → 学习目标」，建立本讲的定位；
2. 二读「核心概念 → 关键推导 → 深入推导」，合上笔记手推核心公式；
3. 按「实现映射」打开对应 starter code，填写 TODO 前先口述算法；
4. 用「易错点」自检，跑「实验或练习」，记录结果；
5. 用「高频问题 → 面试追问链 → 自测题」做输出式复习；
6. 面试/考试前只过「速览卡 + 公式速查 + 追问链」。

## 校验

```bash
python notes/check_notes.py            # 结构 / 公式 / 链接 / 索引 / 术语强校验
python notes/check_notes.py --verbose  # 每讲行数、公式数、问题数统计
```

笔记只写原创摘要与短引用，不复制官方材料内容。
