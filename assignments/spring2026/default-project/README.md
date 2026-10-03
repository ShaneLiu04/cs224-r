# Default Project 学习指南 — Countdown 推理 RL

> 官方指南：[CS224R_Default_Project_Guidelines.pdf](CS224R_Default_Project_Guidelines.pdf) ·
> [CS224R_Custom_Project_Guidelines.pdf](CS224R_Custom_Project_Guidelines.pdf) ·
> starter code：[starter-code/](starter-code/)（内含官方详细 README）

## 项目目标

在 **Countdown** 算术推理任务上完成一条现代 LLM 后训练管线：

```text
SFT（监督微调）→ IPO（偏好优化）→ RLOO（可验证奖励 RL）→ pass@k 评估
```

Countdown 样例：给定目标 `24` 与数字 `[3,4,6,8]`，模型需输出
`<answer>(8 - 4) * 6</answer>`。

奖励（`evaluation/countdown.py`）：

| 得分 | 条件 |
| --- | --- |
| 0.0 | 没有 `<answer>` 标签 |
| 0.1 | 提取到答案但等式无效/结果错误 |
| 1.0 | 用恰好给定的数字构成正确等式 |

## TODO 地图

| 阶段 | 文件 | TODO | 要点 |
| --- | --- | --- | --- |
| SFT | `sft_trainer/sft.py` | `train(...)` | 前向 + 只在 response token 上算 LM loss + 反传/调度/日志/存档 |
| IPO | `ipo_trainer/ipo.py` | `train(...)` | chosen/rejected 对数比 − 参考模型对数比 + 偏好损失 |
| RLOO | `rloo_trainer/rloo_update_worker.py` | `update(...)` | 当前策略 log-prob、LOO 基线、优势、熵/KL 正则、梯度累积 |
| 评估 | `evaluation/view_passk.ipynb` | 分析 | 对比 SFT / IPO / RLOO 的 pass@k |

已实现（不需要动）：数据加载、RLOO 编排/采样（Ray + vLLM）、Countdown 评分、评估脚本。
默认模型 `Qwen/Qwen2.5-0.5B`。

## 运行命令（在 `starter-code/` 下）

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -e ".[all]"
export WANDB_API_KEY=... HF_TOKEN=...
modal setup   # 可选，远端 H100

bash sft_trainer/train_sft.sh          # 或 *_modal.sh
bash ipo_trainer/train_ipo.sh
export MODEL_NAME=... DATASET_NAME=...
bash rloo_trainer/train_rloo.sh

python evaluation/countdown_eval.py \
  --model_path path-or-hf-repo --eval_dataset asingh15/countdown_tasks_3to4 \
  --output_dir evaluation/eval_results --output_name my_eval_run
```

## 关键公式

**SFT**：只在 response token 上做因果 LM 损失。

**IPO**：`L = (Δ_w − Δ_l − 1/(2β))²`，其中 `Δ = log π_θ(y|x) − log π_ref(y|x)`。

**RLOO**：

```text
Â_i = r_i − (1/(K−1)) Σ_{j≠i} r_j
L = −E[ Σ_i log π_θ(y_i|x) · Â_i ] + 熵项 + KL(π_θ || π_ref)
```

## 里程碑与提交

1. SFT 跑通并验证；
2. IPO 对比 SFT 基线；
3. RLOO 调稳定（batch/group/entropy/KL）；
4. `view_passk.ipynb` 做 pass@k 分析；
5. 扩展实验（部分问题/已回答问题的探索，见指南 Section 2）；
6. 报告 + poster。

## 调试清单

- [ ] SFT mask 只对 response token 计 loss（`is_response_token`），不要把 prompt 计入；
- [ ] IPO 需要冻结参考模型并缓存其 log-prob；`β` 不要过小；
- [ ] RLOO 的 `lr_schedule='constant'` 要求 `warmup_ratio=0.0`；
- [ ] 组内奖励全同 → 优势为 0，需要难度混合或过滤；
- [ ] OOM：减小 `batch_size / max_tokens / max_model_len / group_size`，增大梯度累积；
- [ ] vLLM/Ray 启动失败：检查 `CUDA_VISIBLE_DEVICES` 与依赖；
- [ ] W&B 卡住：检查 `WANDB_API_KEY`，本地调试可 `WANDB_MODE=offline`。

## 相关笔记

[L9 RLHF/DPO/IPO](../../../notes/lecture-09-rlhf-preference-optimization.md) ·
[L10 LLM 推理 RL / GRPO / RLOO / pass@k](../../../notes/lecture-10-rl-llm-reasoning.md) ·
[L18 研究方法](../../../notes/lecture-18-frontiers.md)
