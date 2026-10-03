# HW3 学习指南 — Offline RL（AWAC + IQL）

> 官方题面：[CS224R_2026_Homework_3.pdf](CS224R_2026_Homework_3.pdf) ·
> LaTeX 模板：[CS224R_2026_Homework_3.tex](CS224R_2026_Homework_3.tex) ·
> starter code：[starter-code/](starter-code/)

## 任务速览

在 D4RL **AntMaze**（连续控制）与 **PointMass**（离散控制）上实现两种离线 RL 算法：

1. **P1（2 分）AWAC**：`antmaze-umaze-v0` 与 `antmaze-medium-diverse-v0`，3 seeds；
2. **P2（5 分）IQL**：扫 expectile `ζ=0.2/0.9`，在 umaze 选优并用于 medium + AntMaze 对比 AWAC；
3. **P3（1 分）stitching**：PointMass 子优数据上比较 IQL 与 Filtered BC，
   检验是否能拼出优于数据最优（max return −46）的策略。

奖励约定：到达目标 0，其他每步 −1（所以回报越接近 0 越好）。

## TODO 地图

| 问题 | 文件 | TODO | 公式 |
| --- | --- | --- | --- |
| P1 | `cs224r/policies/MLP_policy.py` | 策略网络/对数概率 | 高斯策略 |
| P1 | `cs224r/critics/awac_critic.py` | clipped double-Q 目标 | `y = r + γ E_{a'~π}[min_i Q̄_i(s',a')]` |
| P1 | `cs224r/agents/awac_agent.py` | advantage 加权 NLL | `L_π = −E[log π(a\|s) exp(A/λ)]` |
| P2 | `cs224r/critics/iql_critic.py` | expectile V + Q | `L_V = E[L₂^ζ(Q̄−V)]`；`L_Q = E[(r+γV(s')−Q)²]` |
| P2 | `cs224r/agents/iql_agent.py` | AWR 策略抽取 | `exp(β(Q̄−V))` 加权 |
| P3 | 无需新代码 | 运行两种算法对比 | 观察 max return |

## 运行命令

```bash
conda create -n cs224r-hw3 python=3.10.19 && conda activate cs224r-hw3
pip install modal && modal setup

# AWAC
modal run --detach modal_train_para.py --algo awac \
  --env-name antmaze-umaze-v0 --exp-name awac_antmaze_umaze --use-wandb

modal run --detach modal_train_para.py --algo awac \
  --env-name antmaze-medium-diverse-v0 --exp-name awac_antmaze_medium_diverse --use-wandb

# IQL（先调试单 seed）
modal run --detach modal_train.py --algo iql \
  --env-name antmaze-umaze-v0 --iql-expectile 0.9 \
  --exp-name iql_zeta_0.9_umaze --use-wandb --seed 1

# PointMass stitching（IQL vs Filtered BC）
modal run --detach modal_train_para.py --algo iql --env-name PointmassMedium-v0 \
  --exp-name iql_zeta_0.9_stitching --iql-expectile 0.9 \
  --offline-dataset offline_datasets/pointmass_stitching_dataset.npz --use-wandb

modal run --detach modal_train_para.py --algo bc --env-name PointmassMedium-v0 \
  --exp-name filtered_bc_stitching \
  --offline-dataset offline_datasets/pointmass_stitching_dataset.npz \
  --filter-top-percent 10 --use-wandb
```

结果下载：

```bash
mkdir -p data
modal volume get cs224r-hw3-results / ./data/
```

## Autograder CSV 结构（重要）

```text
P1/1/awac_umaze_seed{1,2,3}.csv
P1/2/awac_medium_maze_seed{1,2,3}.csv
P2/1/iql_umaze_seed{1,2,3}.csv
P2/2/iql_medium_maze_seed{1,2,3}.csv
```

- 每个文件夹恰好 3 个 CSV（每 seed 一个），**不要合并 seeds**；
- 从 W&B 名为 `Eval_AverageReturn` 的图导出；
- IQL 的 P2 只提交最优 expectile 的 CSV。

## 调试清单

- [ ] expectile 权重方向：`ζ>0.5` 时对 `µ>0`（预测偏低）加权更多；
- [ ] `V` 的目标 stop-gradient；Q 的目标用 `V_ψ(s')` 而非 `max`；
- [ ] critic 只在数据动作上更新（IQL），不要采样策略动作；
- [ ] AWAC 的策略更新里 advantage 要 detach，指数权重做截断/归一化防爆；
- [ ] 数据下载到 `~/.d4rl/datasets/`（容器退出即清空，正常）；
- [ ] 评估视频在 Modal volume 的对应实验目录下，可用于定性分析；
- [ ] 3 seeds 并行时注意 Modal 资源配额。

## 验收标准

- AWAC/IQL 在 umaze 与 medium-diverse 上给出 3-seed 均值±标准差；
- IQL 通常优于 AWAC，尤其在 medium（更难的 maze）；
- stitching：IQL 回报应超过数据最优 −46，Filtered BC 一般不能。

## 相关笔记

[L7 离线 RL（AWR/IQL/CQL/stitching）](../../../notes/lecture-07-offline-rl.md)
