# 学习进度

更新时链接到对应笔记、实验或实现，不只勾选状态。

## 准备

- [x] 同步官方材料（sl/assignments/materials，固定 commit 见 UPSTREAM.md）
- [x] 建立 18 讲中文笔记初稿
- [x] 笔记深度优化（速览卡、深入推导、面试追问链、自测题；`python notes/check_notes.py` 通过）
- [x] 建立官方/社区资源索引与实验卡片
- [x] 建立模板、检查脚本与 `solutions/` 工作区
- [ ] 跑通 PyTorch + MuJoCo/Gymnasium 本地环境
- [ ] 配置 Modal / W&B（HW2、HW3 需要）

## 笔记（初稿已生成，复习后升级为「已验证」）

- [x] L01 Intro & MDPs — `notes/lecture-01-intro-mdps.md`
- [x] L02 Imitation Learning
- [x] L03 Policy Gradients
- [x] L04 Actor-Critic
- [x] L05 Off-Policy Actor-Critic
- [x] L06 Q-Learning
- [x] L07 Offline RL
- [x] L08 Reward Learning
- [x] L09 RLHF & DPO
- [x] L10 RL for LLM Reasoning
- [x] L11 Model-Based RL
- [x] L12 Multi-Task & Goal-Conditioned RL
- [x] L13 Meta-RL
- [x] L14 Exploration（2025 版补充）
- [x] L15 Hierarchical RL & IL
- [x] L16 Sim-to-Real
- [x] L17 RL for VLAs
- [x] L18 Frontiers & Research
- [ ] 期中自测（覆盖 L01–L12）

## Assignment 1 — Imitation Learning（Flappy Bird）

- [ ] `networks.py::BCPolicy` + `losses.py::bc_loss`
- [ ] `FlowMatchingSchedule.interpolate/sample` + `losses.py::flow_matching_loss`
- [ ] `dagger.py::DeterministicExpert.act / rollout_episode / rollout_and_relabel`
- [ ] 跑通 `bc_reg easy/hard`、`bc_flow hard`、`dagger hard` 四条命令
- [ ] 完成报告（表格 + 学习曲线 + 3 个解释题）

## Assignment 2 — Online RL（Gridworld + Sawyer hammer）

- [ ] `gridworld_q_learning.py` 三个 reward 场景
- [ ] `on_policy.py::compute_gae` + PPO clipped loss
- [ ] `off_policy.py::bc`、`update_critic`、`update_actor`
- [ ] PPO 达到 ≥25% success；SAC 式 AC 达到 ≥90%
- [ ] UTD=5 / 10 critics 消融对比

## Assignment 3 — Offline RL（AWAC + IQL）

- [ ] `awac_critic.py` / `awac_agent.py`，跑通 antmaze-umaze + medium-diverse
- [ ] `iql_critic.py` / `iql_agent.py`，扫 expectile 0.2 vs 0.9
- [ ] PointMass stitching 实验（IQL vs Filtered BC）
- [ ] 导出 autograder CSV（P1/P2 各 3 seeds）

## Default Project — Countdown

- [ ] `sft_trainer/sft.py::train`
- [ ] `ipo_trainer/ipo.py::train`
- [ ] `rloo_trainer/rloo_update_worker.py::update`
- [ ] pass@k 分析 + 新扩展
- [ ] 项目报告与 poster

## 旧版本迁移记录

- [ ] 2025 → 2026 作业接口差异整理（见 `resources/versions.md`）
