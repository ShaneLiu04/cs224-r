# 主题：多任务、元学习与探索

> 对应讲次：[L12](../../notes/lecture-12-multi-task-goal-conditioned-rl.md)、
> [L13](../../notes/lecture-13-meta-rl.md)、[L14](../../notes/lecture-14-exploration.md)；
> 历史作业：2023/2025 HW4（HER + Meta-RL）。

## 问题脉络

1. 任务分布上的泛化 → 条件 MDP / 多任务 RL；
2. 目标状态作为任务 → GCRL + HER；
3. 测试时任务未知 → meta-RL（RL²、MAML、PEARL）；
4. 主动收集信息 → 探索：内在奖励、后验采样、UCB。

## 必读论文

| 论文 | 年份 | 为什么读 |
| --- | --- | --- |
| Andrychowicz et al., *HER* | 2017 | 目标重标 |
| Schaul et al., *UVFA* | 2015 | 目标条件值函数 |
| Duan et al., *RL²* | 2016 | 黑盒元 RL |
| Finn et al., *MAML* | 2017 | 学初始化 |
| Rakelly et al., *PEARL* | 2019 | 概率上下文元 RL |
| Zintgraf et al., *VariBAD* | 2020 | 贝叶斯任务推断 |
| Eysenbach et al., *DIAYN* | 2018 | 无监督技能 |
| Burda et al., *RND* | 2019 | 内在奖励 |
| Pathak et al., *ICM* | 2017 | 好奇心 |
| Osband et al., *Bootstrapped DQN* | 2016 | 深度探索 |

## 公式备忘

```text
多任务目标: E_{T~p(T)}[J_T(θ)]
MAML: θ' = θ − α∇L_T(θ);  θ ← θ − β∇Σ L_T(θ')
PEARL: max E_{q(z|c)}[log p(c|z)] − KL(q(z|c)||p(z))
UCB: a = argmax Q + c√(log t / N(a))
```

## 常见实验

- HER 的目标采样策略（final/future/episode）；
- 任务数量与泛化的 scaling；
- RL² 的上下文长度对适应速度的影响；
- PEARL 的 KL 系数与探索行为；
- RND 在稀疏导航中的探索覆盖。

## 开放问题

- 任务分布如何自动发现/生成（open-endedness）？
- 元学习的分布外任务泛化；
- 探索与安全的平衡；
- LLM 作为任务推断器/规划器的角色。

## 本地材料

- slides 12–14；
- 2025/2023 HW4：`assignments/spring2025/homework4-goal-meta-rl/`（HER + meta_rl）。
