# CS224R 十八讲自学笔记

原创中文笔记，保留英文术语；不逐页翻译，也不替代官方 lecture materials。
每讲固定结构：目标 → 概念 → 推导 → 实现映射 → 易错点 → 实验 → 高频问题 → 小结。

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

## 官方链接

- 课程主页：https://cs224r.stanford.edu/
- 课程网站源码（材料快照来源）：https://github.com/cs224r/cs224r.github.io
- 2025 视频：https://www.youtube.com/playlist?list=PLoROMvodv4rPwxE0ONYRa_itZFdaKCylL
- 官方 head TA 笔记（Jubayer Ibn Hamid）：
  https://jubayer-ibn-hamid.github.io/data/RL_Notes__final_.pdf

## 辅助导航

- [术语与符号表](GLOSSARY.md)
- [按目标/时间预算的阅读路线](READING-ROADMAP.md)
- [论文、工具与社区资料库](../experiments/README.md)
- [三代作业材料](../assignments/README.md)

## 笔记使用法

1. 首读「目标 → 核心概念 → 小结」，对照官方 slides 标记不懂的地方；
2. 二读推导并手算公式（策略梯度、GAE、expectile、DPO）；
3. 按「实现映射」打开对应 starter code，填写 TODO 前先口述算法；
4. 用「易错点」自检，再跑「实验或练习」；
5. 面试/考试前只看「高频问题」与公式速查。

笔记只写原创摘要与短引用，不复制官方材料内容。
