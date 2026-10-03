# HW2 学习指南 — Online Reinforcement Learning

> 官方题面：[CS224R_2026_Homework_2.pdf](CS224R_2026_Homework_2.pdf) ·
> LaTeX 模板：[CS224R_2026_Homework_2.tex](CS224R_2026_Homework_2.tex) ·
> starter code：[starter-code/](starter-code/)

## 任务速览

三部分：

1. **P1（3 分）**：5×4 gridworld 上的表格 Q-learning，比较三种奖励函数；
2. **P2（3 分）**：Meta-World `hammer-v2`（4-DoF Sawyer，稀疏奖励）上的 **PPO + GAE**，
   要求至少 25% success（1M steps）；
3. **P3（4 分）**：同一任务上的 **off-policy actor-critic**（critic ensemble + BC 预训练 +
   双 Q + target network），要求 100k steps 内 ≥90%；再做 `10 critics / UTD=5` 的消融。

官方要求 Modal 计算 + W&B 记录；本地 GPU 可作开发但不被支持。

## TODO 地图

| 问题 | 文件 | TODO | 要点 |
| --- | --- | --- | --- |
| P1 | `gridworld_q_learning.py` | `choose_action` | `ε`-greedy；平局任意 |
| P1 | `gridworld_q_learning.py` | `train_q_learning` | 选动作→step→更新 Q→done 终止 |
| P2 | `on_policy.py` | `compute_gae` | 逆序递推 `δ_t`、`Â_t`、`returns = Â + V` |
| P2 | `on_policy.py` | `update` 两处 `### YOUR CODE HERE ###` | `no_grad` 算 GAE；PPO clip loss |
| P3 | `off_policy.py` | `bc` | 对演示做 `−log π(a\|s)` 预训练 |
| P3 | `off_policy.py` | `update_critic` | 采样两个 target critics 取 min；soft update |
| P3 | `off_policy.py` | `update_actor` | `−mean_i Q_i(s, π(s))`（可加 BC 正则） |

**只允许改这三个文件**（官方说明：只有它们含 `YOUR CODE HERE`）。

关键公式：

```text
GAE:      δ_t = r_t + γ(1−d_t)V(s_{t+1}) − V(s_t)
          Â_t = δ_t + γλ(1−d_t)Â_{t+1}
PPO:      ρ_t = exp(logπ_θ − logπ_old),  L = −E[min(ρ_tÂ_t, clip(ρ_t,1±ε)Â_t)]
Critic:   y = r + γ min(Q̄_i(s',a'), Q̄_j(s',a')),  a' ~ π_θ(s')
Target:   Q̄ = (1−ρ)Q̄ + ρQ
```

## 运行命令

```bash
# 本地（开发用）
conda env create -f conda_env_modal.yml && conda activate cs224r-hw2
wandb login

# P1
python gridworld_q_learning.py

# P2 / P3（Modal，长任务可加 --detach）
modal run modal_on_policy.py
modal run modal_off_policy.py

# P3 消融：修改 modal_off_policy.py 子进程参数
#   agent.num_critics=10  utd=5
```

需要下载的 CSV：W&B 的 `eval/episode_success` 图 → 右上角导出。

## 报告要求

- P1：三个奖励场景的轨迹与总回报，每场景一句话解释；
- P2：`eval/episode_success` 截图（至 1M steps），≥25%；
- P3：主实验截图（至 100k，≥90%）+ UTD 消融截图（至 40k，≥90%）+ 一句话解释；
- 对比 P2 与 P3 学习曲线（样本效率、最终性能）并解释至少两个差异原因。

## 调试清单

- [ ] GAE 逆序；`done` 处不 bootstrap；`returns = advantages + values`；
- [ ] PPO 比率用 log 差再 exp；`π_old` 是收集数据时的策略，先算好并冻结；
- [ ] critic loss 全 N 个网络都更新（只 min 两个 target 不等于只更新两个）；
- [ ] Bellman target 要 `detach`；target network 用 EMA（`soft_update_params`）；
- [ ] BC 预训练要真正跑（稀疏奖励下随机策略几乎无正样本）；
- [ ] UTD=5 时训练约 2 小时/50k steps，注意 Modal timeout 与 `--detach`；
- [ ] 观测是最近两个环境 step 的拼接，形状对齐；
- [ ] W&B secret：`modal secret create wandb-secret WANDB_API_KEY=... --force`。

## 验收标准

- P1：三场景都能到达某个目标且解释与回报计算一致；
- P2：PPO ≥25%；
- P3：2 critics/UTD=1 时 ≥90%（100k 内）；10 critics/UTD=5 更快（40k 内）。

## 相关笔记

[L3 策略梯度](../../../notes/lecture-03-policy-gradients.md) ·
[L4 Actor-Critic / GAE](../../../notes/lecture-04-actor-critic.md) ·
[L5 Off-policy AC / PPO / SAC](../../../notes/lecture-05-off-policy-actor-critic.md) ·
[L6 Q-Learning](../../../notes/lecture-06-q-learning.md)
