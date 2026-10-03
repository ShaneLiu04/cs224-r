# HW1 学习指南 — Imitation Learning（Flappy Bird）

> 官方题面：[CS224R_2026_Homework_1.pdf](CS224R_2026_Homework_1.pdf) ·
> LaTeX 模板：[CS224R_2026_Homework_1.tex](CS224R_2026_Homework_1.tex) ·
> starter code：[starter-code/](starter-code/)（官方 `README.md`、`installation.md`、
> `colab_instructions.md` 都在里面）

## 任务速览

在物理 Flappy Bird 环境上用**动作块（action chunking）**做模仿学习：

- 观测（4 维）：`[dist to pipe, gap1 y, gap2 y, bird y]`；
- 动作：目标 y 位置 `∈ [0,1]`（PD 控制器转推力）；
- 难度：easy（每根管子一个缺口）/ hard（交替单/双缺口）；
- 每 episode 最多 1000 步，存活 1000 步算成功；
- 动作块：预测 `ACTION_CHUNK=20` 个动作，执行 `EXECUTE_STEPS=10` 后重查（receding horizon）。

三个问题：

1. **BC 回归**（MSE）——easy + hard；
2. **Flow matching**——hard；
3. **DAgger**——hard，5 轮迭代。

## TODO 地图

| 文件 | TODO | 要点 | 依赖 |
| --- | --- | --- | --- |
| `networks.py` | `BCPolicy.__init__/forward` | 3 层 MLP：Linear→ReLU→Linear→ReLU→Linear→Sigmoid；输出 `(B,20)` | 无 |
| `losses.py` | `bc_loss` | `MSE(π(s), a*)`，对 batch 求平均 | BCPolicy |
| `networks.py` | `FlowMatchingSchedule.interpolate` | 采样 τ、噪声；`a_τ = τa₁+(1−τ)a₀`；返回 `(a_τ, v=a₁−a₀)` | 无 |
| `networks.py` | `FlowMatchingSchedule.sample` | 从 `a₀~N(0,I)` 用 Euler 积分 `n` 步；clamp 到 `[0,1]` | interpolate |
| `losses.py` | `flow_matching_loss` | `MSE(v_θ(s,a_τ,τ), a₁−a₀)`（tip：调用 `schedule.interpolate`） | schedule |
| `dagger.py` | `DeterministicExpert.act` | hard mode 下确定化专家（消除多峰歧义） | 读 `expert.py` |
| `dagger.py` | `rollout_episode` | 当前策略 rollout，收集 `(s,a)`，用 action chunk + 执行 K 步 | BCPolicy |
| `dagger.py` | `rollout_and_relabel` | rollout 后用专家重标，返回新训练对 | 上两个 |

推荐顺序（官方）：1→2→3→4→5→6→7。

## 运行命令

```bash
cd starter-code
pip install -e .          # 或按 installation.md

python main.py --method bc_reg  --env easy
python main.py --method bc_reg  --env hard
python main.py --method bc_flow --env hard
python main.py --method dagger  --env hard          # 默认 5 轮
python main.py --plot                                # 对比图
```

输出文件（提交用）：`bc_reg_easy.txt`、`bc_reg_hard.txt`、`bc_flow_hard.txt`、`dagger_hard.txt`。

## 报告要求

- P1：easy/hard 的 episode 长度均值 ± 标准差（50 个评估 episode）；解释 hard 上 MSE 的表现；
- P2：hard 上 flow matching 的结果 + 2–3 句解释；
- P3：DAgger 学习曲线（x=轮次，y=平均 episode 长度，带误差棒，叠加 BC 水平线）；
  三方法柱状/表格对比；解释 DAgger 为什么有效、确定性专家的作用。

## 调试清单

- [ ] `BCPolicy` 输出是 `(B,20)` 而不是 `(B,)`：最后一个 Linear 的 out_features 别写错；
- [ ] flow 的 τ 维度：广播到 `a` 的形状，速度目标不能反；
- [ ] Euler 步长是 `1/n`，不是 `n`；采样步数 `n` 别设太小；
- [ ] DAgger 要**聚合**每轮数据（`D ← D ∪ D_i`），否则灾难性遗忘；
- [ ] rollout 里别忘了 `env.reset()` 和 `done` 处理；action chunk 只执行前 K 个；
- [ ] `DeterministicExpert` 的确定性规则要与 hard mode 的随机专家行为一致；
- [ ] 评估口径：50 episodes、存活长度；训练 loss 不等于性能。

## 验收标准

- easy 上 BC 回归接近满分（存活接近 1000）；
- hard 上 MSE 显著退化（多峰问题），flow matching 明显更好；
- DAgger 曲线随轮次上升并超过 BC 基线。

## 相关笔记

[L1 环境与 MDP](../../../notes/lecture-01-intro-mdps.md) ·
[L2 模仿学习、flow matching、DAgger](../../../notes/lecture-02-imitation-learning.md)
