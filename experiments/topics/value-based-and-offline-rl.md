# 主题：值方法与离线 RL

> 对应讲次：[L6](../../notes/lecture-06-q-learning.md)、[L7](../../notes/lecture-07-offline-rl.md)；
> 作业：[HW2 P1](../../assignments/spring2026/homework2-online-rl/README.md)、
> [HW3](../../assignments/spring2026/homework3-offline-rl/README.md)。

## 问题脉络

1. 隐式策略 + Bellman 最优 → Q-learning；
2. 过估计 → Double Q / clipped double-Q；
3. 稳定训练 → replay + target network + n-step；
4. 固定数据 + OOD 动作 → 离线 RL；
5. 策略约束/保守化：AWR → IQL → CQL → 决策 Transformer 路线。

## 必读论文

| 论文 | 年份 | 为什么读 |
| --- | --- | --- |
| Watkins & Dayan, *Q-learning* | 1992 | 收敛性 |
| Mnih et al., *DQN* | 2015 | 深度值方法的工程要点 |
| van Hasselt et al., *Double DQN* | 2016 | 过估计修正 |
| Peng et al., *AWR* | 2019 | 加权模仿 |
| Nair et al., *AWAC* | 2020 | 离线 actor-critic |
| Kostrikov et al., *IQL* | 2021 | expectile 与避免 OOD |
| Kumar et al., *CQL* | 2020 | 保守价值下界 |
| Fu et al., *D4RL* | 2020 | 数据集与评测协议 |
| Levine et al., *Offline RL survey* | 2020 | 综述 |

## 公式备忘

```text
Q-learning:  Q ← Q + α[r + γ max Q − Q]
Double:      y = r + γ Q̄(s', argmax Q(s',·))
AWR:         L = −E[log π · exp(A/λ)]
IQL-V:       L = E[L₂^ζ(Q̄ − V)],  IQL-Q: (r + γV(s') − Q)²
CQL:         α(E_{µ}[Q] − E_D[Q]) + TD
```

## 常见实验

- gridworld 奖励设计（HW2 P1）；
- Double Q vs Q-learning 的 Q 值偏差；
- IQL expectile 扫描（0.1–0.9）；
- AWAC vs IQL 在 umaze/medium 的对比（HW3）；
- PointMass stitching：IQL vs Filtered BC（HW3 P3）；
- 离线预训练 + 在线微调（cal-QL / RLPD 方向）。

## 开放问题

- 离线评估（OPE）如何可信？
- 何时「保守」是必要的，何时只是拖慢学习？
- 决策 Transformer / diffusion planner 与 TD 方法的边界？

## 本地材料

- HW2 `gridworld_q_learning.py`；HW3 `cs224r/critics/`、`offline_datasets/`；
- slides 06–07；
- [catalog/hw3-offline-rl.md](../catalog/hw3-offline-rl.md)。
