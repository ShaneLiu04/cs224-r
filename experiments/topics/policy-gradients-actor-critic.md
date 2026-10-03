# 主题：策略梯度与 Actor-Critic

> 对应讲次：[L3](../../notes/lecture-03-policy-gradients.md)、[L4](../../notes/lecture-04-actor-critic.md)、
> [L5](../../notes/lecture-05-off-policy-actor-critic.md)；作业：[HW2](../../assignments/spring2026/homework2-online-rl/README.md)。

## 问题脉络

1. 只有奖励，没有专家 → 直接优化期望回报；
2. `∇log` 换元 + causality + baseline → REINFORCE；
3. 用值函数做 baseline / 引入 bootstrap → actor-critic；
4. GAE 调偏差-方差；
5. 多轮复用与重放 → PPO / SAC。

## 必读论文

| 论文 | 年份 | 为什么读 |
| --- | --- | --- |
| Williams, *REINFORCE* | 1992 | 策略梯度原点 |
| Sutton et al., *Policy Gradient with Function Approximation* | 1999 | 兼容函数逼近的理论 |
| Schulman et al., *GAE* | 2016 | 现代 actor-critic 的标准组件 |
| Schulman et al., *PPO* | 2017 | 工程上最常用的策略优化 |
| Haarnoja et al., *SAC* | 2018 | 最大熵 off-policy AC |
| Mnih et al., *A3C* | 2016 | 并行 on-policy 训练范式 |

## 公式备忘

```text
∇J = E[Σ_t ∇log π_θ(a_t|s_t) · Â_t]
Â_t = δ_t + γλÂ_{t+1},   δ_t = r_t + γV(s_{t+1}) − V(s_t)
PPO: L = −E[min(ρÂ, clip(ρ,1±ε)Â)]
SAC: J = Σ E[r + αH(π)],  y = r + γ(min Q̄ − α log π)
```

## 常见实验

- baseline 类型（无 / 均值 / V 网络）对方差的影响；
- GAE `λ ∈ {0, 0.9, 0.95, 1.0}` 的偏差-方差曲线；
- PPO clip `ε`、epochs、minibatch 大小的交互；
- SAC 熵温度固定 vs 自动；
- on-policy vs off-policy 的样本效率对比（HW2 P2 vs P3）。

## 开放问题

- 如何让 off-policy 方法在大模型上稳定（RLHF 中的 critic 成本）？
- 无 critic 的优势估计（GRPO/RLOO）的理论性质？
- 分布式/异步训练下的 staleness 与重要性修正？

## 本地材料

- HW2 starter code（PPO + off-policy AC）；
- slides 03–05；
- [catalog/hw2-online-rl.md](../catalog/hw2-online-rl.md)。
