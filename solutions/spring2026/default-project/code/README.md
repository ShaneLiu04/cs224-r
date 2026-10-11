# Default Project

This folder is the main class-project workspace for training and evaluating language models on the Countdown arithmetic reasoning task.

The goal of this README is to be readable even if you are new to reinforcement learning (RL).

## 1. What You Are Building

You are implementing a standard post-training pipeline used in modern LLM work:

1. `SFT` (Supervised Fine-Tuning): learn from correct examples.
2. `IPO` (A General Theoretical Paradigm to Understand Learning from Human Preferences): learn from preference pairs.
3. `RLOO` (Reinforce Leave-One-Out): RL-style optimization directly from rewards.
4. `Evaluation`: measure how often the model solves Countdown problems.

Important: this is starter code for a class project. Some training logic is intentionally left as TODOs.

## 2. Current Project Status (What Is and Is Not Implemented)

Implemented:
- Data loading and batching for SFT, IPO, and RLOO.
- RLOO high-level orchestration (sampling, reward computation, checkpoint handoff).
- Countdown reward/scoring and vLLM-based evaluation pipeline.

Not implemented (you need to write these for the project):
- `sft_trainer/sft.py`: `train(...)` raises `NotImplementedError`.
- `ipo_trainer/ipo.py`: `train(...)` raises `NotImplementedError`.
- `rloo_trainer/rloo_update_worker.py`: `update(...)` raises `NotImplementedError`.

**Milestones for submission**

1. Implement and validate SFT training loop.
2. Implement IPO loop and compare against SFT baseline.
3. Implement RLOO update step and tune stability (batch/group/entropy/KL).
4. Run pass@k analysis in `evaluation/view_passk.ipynb`.
5. Implement novel extension entailing exploration of an answered/partially answered problem (See Section 2 of the Project Report)

## 3. Countdown Task Overview

Each example provides:
- Target number (for example, `24`)
- Allowed numbers (for example, `[3, 4, 6, 8]`)

The model should output an equation inside tags:

```text
<answer>(8 - 4) * 6</answer>
```

Scoring in `evaluation/countdown.py`:
- `0.0`: no `<answer>...</answer>` found.
- `0.1` (format score): answer extracted but invalid equation or wrong result.
- `1.0`: valid equation using exactly the provided numbers and equals target.

## 4. Folder Map

```text
student_code/
├── README.md
├── modal_train.py             # Shared Modal app/entrypoint for remote training
├── sft_trainer/
│   ├── sft.py                  # SFT training entrypoint (train loop TODO)
│   ├── sft_dataset.py          # SFT dataset + collate logic
│   ├── train_sft.sh            # Example SFT launcher
│   ├── train_sft_modal.sh      # Modal SFT launcher
│   ├── upload_sft.py           # Push trained checkpoint to Hugging Face Hub
│   └── upload_model_hf_sft.sh  # Batch upload helper script
├── ipo_trainer/
│   ├── ipo.py                  # IPO training entrypoint (train loop TODO)
│   ├── ipo_dataset.py          # IPO pairwise dataset + collate logic
│   ├── train_ipo.sh            # Example IPO launcher
│   └── train_ipo_modal.sh      # Modal IPO launcher
├── rloo_trainer/
│   ├── rloo.py                 # RLOO training orchestration loop
│   ├── rloo_update_worker.py   # Gradient update worker (update TODO)
│   ├── sampling_worker.py      # vLLM sampling worker
│   ├── rloo_dataset.py         # RLOO dataset loader
│   ├── train_rloo.sh           # Example RLOO launcher
│   └── train_rloo_modal.sh     # Modal RLOO launcher
└── evaluation/
    ├── countdown.py            # Answer extraction + validation + reward
    ├── countdown_eval.py       # Batch evaluation with vLLM
    ├── sample.sh               # Multi-model eval launcher
    ├── sample_modal.sh         # Multi-model Modal eval launcher
    └── view_passk.ipynb        # Notebook for pass@k analysis
```

## 6. Setup (First Time)

Use the repository root for installation, then switch into `student_code` for all training/eval commands. You can run training either locally on your own GPU or remotely on Modal.

### 6.1 Python Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[all]"
```

Notes:
- You need a CUDA-capable GPU for practical training/sampling.
- The code loads many models in `bfloat16`; ensure your GPU/runtime supports it.
- `modal` is included in `pip install -e ".[all]"`, so the same environment can be used for either local or Modal launches.

### 6.2 Authentication / Environment Variables

We will be using Weights and Biases for experiment tracking. For more details, you can check out their documentation [here](https://docs.wandb.ai/models/quickstart).

Set these before launching scripts:

```bash
export WANDB_API_KEY=...
export HF_TOKEN=...
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HOME/.cache/huggingface}
export WANDB__SERVICE_WAIT=300
```

Optional:
- `WANDB_ENTITY`
- `CUDA_VISIBLE_DEVICES`

### 6.3 Optional Modal Setup

If you want to run training or evaluation remotely on Modal instead of locally:

```bash
modal setup
```

The Modal launchers added in this repo are:
- `sft_trainer/train_sft_modal.sh`
- `ipo_trainer/train_ipo_modal.sh`
- `rloo_trainer/train_rloo_modal.sh`
- `evaluation/sample_modal.sh`

These scripts call `modal_train.py`, which packages this project and runs the existing trainer entrypoints remotely. By default they request:
- `H100!` GPU
- `86400` second timeout (24 hours)
- Modal volume `default-proj-training` for checkpoints and Hugging Face cache

Useful optional overrides before launching:

```bash
export MODAL_GPU=H100!
export MODAL_TIMEOUT_SECONDS=86400
export MODAL_VOLUME_NAME=default-proj-training
```

Modal artifacts are written into the mounted volume under `/vol`, for example:
- `/vol/checkpoints/sft_model`
- `/vol/checkpoints/dpo_checkpoints`
- `/vol/checkpoints/rloo_checkpoints`
- `/vol/evaluation/eval_results`

To inspect or download artifacts later:

```bash
modal volume ls default-proj-training /
modal volume get default-proj-training checkpoints ./downloaded_checkpoints
modal volume get default-proj-training evaluation/eval_results ./downloaded_eval_results
```

## 7. Expected Dataset Schemas

The loaders assume specific column names.

### 7.1 SFT (`sft_trainer/sft_dataset.py`)

Required columns:
- `query` (string prompt)
- `completion` (string target response)

### 7.2 IPO (`ipo_trainer/ipo_dataset.py`)

Required columns:
- `query`
- `response_ws` (preferred/chosen response)
- `response_ls` (non-preferred/rejected response)

### 7.3 RLOO (`rloo_trainer/rloo_dataset.py`)

Required columns:
- `prompt`
- `ground_truth` (dict containing at least `target` and `numbers`)

### 7.4 Evaluation (`evaluation/countdown_eval.py`)

Expected test split columns include:
- `prompt`
- `target`
- `nums` (mapped into `ground_truth["numbers"]`)

If your dataset uses different field names, update the loader code or preprocess columns before training/eval.

## 8. Stage-by-Stage Workflow

All commands in this section assume your current directory is `student_code`.

### 8.1 SFT Stage

Entry files:
- `sft_trainer/sft.py`
- `sft_trainer/sft_dataset.py`
- `sft_trainer/train_sft.sh`
- `sft_trainer/train_sft_modal.sh`

Typical local command:

```bash
bash sft_trainer/train_sft.sh
```

Typical Modal command:

```bash
bash sft_trainer/train_sft_modal.sh
```

Direct Python command (custom args):

```bash
python sft_trainer/sft.py \
  --model_name Qwen/Qwen2.5-0.5B \
  --dataset_name Asap7772/cog_behav_all_strategies \
  --output_dir checkpoints/sft \
  --batch_size 64 \
  --gradient_accumulation_steps 8 \
  --num_epochs 1
```

What you need to implement:
- Inside `sft_trainer/sft.py`, write the `train(...)` loop:
  - forward pass
  - masked LM loss on response tokens (`is_response_token`)
  - backward pass + optimizer/scheduler step
  - logging and checkpointing

### 8.2 IPO Stage

Entry files:
- `ipo_trainer/ipo.py`
- `ipo_trainer/ipo_dataset.py`
- `ipo_trainer/train_ipo.sh`
- `ipo_trainer/train_ipo_modal.sh`

Typical local command:

```bash
bash ipo_trainer/train_ipo.sh
```

Typical Modal command:

```bash
bash ipo_trainer/train_ipo_modal.sh
```

What you need to implement:
- Inside `ipo_trainer/ipo.py`, write `train(...)` for IPO objective:
  - compute policy log-probs on chosen/rejected
  - compute reference log-probs
  - compute preference loss
  - optimize + log metrics

### 8.3 RLOO Stage

Entry files:
- `rloo_trainer/rloo.py`
- `rloo_trainer/rloo_update_worker.py`
- `rloo_trainer/sampling_worker.py`
- `rloo_trainer/train_rloo.sh`
- `rloo_trainer/train_rloo_modal.sh`

Typical local command:

```bash
export MODEL_NAME=path-or-hf-repo-of-initial-policy
export DATASET_NAME=your-rloo-dataset
bash rloo_trainer/train_rloo.sh
```

Typical Modal command:

```bash
export MODEL_NAME=path-or-hf-repo-of-initial-policy
export DATASET_NAME=your-rloo-dataset
bash rloo_trainer/train_rloo_modal.sh
```

What already happens in `rloo.py`:
- Sample `group_size` responses per prompt via Ray + vLLM worker.
- Score each response with `compute_score`.
- Tokenize prompt/response pairs for policy update.
- Call update worker, then save latest checkpoint.

What you need to implement:
- `RLOOUpdateWorker.update(...)` in `rloo_trainer/rloo_update_worker.py`:
  - compute per-sample log-probs under current policy
  - compute leave-one-out baseline and advantages
  - add entropy and optional KL penalties
  - backprop with gradient accumulation support

Important config detail:
- `lr_schedule='constant'` currently requires `warmup_ratio=0.0` in update worker.

## 9. Evaluation Workflow

Evaluation entry files:
- `evaluation/countdown_eval.py`
- `evaluation/sample.sh`
- `evaluation/sample_modal.sh`

Single model evaluation:

```bash
python evaluation/countdown_eval.py \
  --model_path path-to-checkpoint-or-hf-repo \
  --eval_dataset asingh15/countdown_tasks_3to4 \
  --output_dir evaluation/eval_results \
  --output_name my_eval_run
```

Single model evaluation on Modal:

```bash
modal run modal_train.py eval \
  --model_path path-to-checkpoint-or-hf-repo \
  --eval_dataset asingh15/countdown_tasks_3to4 \
  --output_dir /vol/evaluation/eval_results \
  --output_name my_eval_run
```

Multi-model script:

1. Edit arrays in `evaluation/sample.sh`:
   - `gpus`
   - `model_paths`
   - `output_names`
2. Run:

```bash
bash evaluation/sample.sh
```

Multi-model Modal script:

1. Edit arrays in `evaluation/sample_modal.sh`:
   - `model_paths`
   - `output_names`
2. Run:

```bash
bash evaluation/sample_modal.sh
```

Outputs:
- JSON written to `output_dir/output_name.json`
- logs from sample script in `logs/`
- Modal evaluation outputs are typically written to `/vol/evaluation/eval_results` inside the Modal volume.

## 10. Checkpoint Upload to Hugging Face

Files:
- `sft_trainer/upload_sft.py`
- `sft_trainer/upload_model_hf_sft.sh`

Single upload example:

```bash
python sft_trainer/upload_sft.py \
  --model_path path/to/checkpoint/model \
  --base_model Qwen/Qwen2.5-0.5B \
  --output_name your-hf-username/your-model-name
```

Batch upload:
- Fill arrays in `upload_model_hf_sft.sh`, then run:

```bash
bash sft_trainer/upload_model_hf_sft.sh
```

## 11. Troubleshooting

`NotImplementedError` during training:
- Expected until you complete TODO functions listed in Section 2.

Out-of-memory (OOM):
- Reduce `batch_size`, `max_tokens`, `max_model_len`, `group_size`.
- Increase gradient accumulation instead of per-step batch.

vLLM / Ray startup issues:
- Confirm CUDA visibility (`CUDA_VISIBLE_DEVICES`).
- Ensure `vllm` and `ray` are installed in the active environment.

Dataset key errors:
- Verify columns match Section 7.

W&B hangs or auth failures:
- Verify `WANDB_API_KEY`.
- Try running with `WANDB_MODE=offline` for local debugging.

Modal launch issues:
- Run `modal setup` if you have not authenticated yet.
- Check the configured volume with `modal volume ls default-proj-training /`.
- If your workspace has multiple Modal environments, pass `--env` to `modal run` or set `MODAL_ENVIRONMENT`.

Missing evaluation results after Modal run:
- Check `modal volume ls default-proj-training /evaluation/eval_results`.
- Download results with `modal volume get default-proj-training evaluation/eval_results ./downloaded_eval_results`.

## 12. Quick Command Cheat Sheet

```bash
# SFT local (after implementing train loop)
bash sft_trainer/train_sft.sh

# SFT on Modal
bash sft_trainer/train_sft_modal.sh

# IPO local (after implementing train loop)
bash ipo_trainer/train_ipo.sh

# IPO on Modal
bash ipo_trainer/train_ipo_modal.sh

# RLOO local (after implementing update())
export MODEL_NAME=...
export DATASET_NAME=...
bash rloo_trainer/train_rloo.sh

# RLOO on Modal
export MODEL_NAME=...
export DATASET_NAME=...
bash rloo_trainer/train_rloo_modal.sh

# Evaluate a checkpoint
python evaluation/countdown_eval.py --model_path ... --output_name ...

# Evaluate on Modal
bash evaluation/sample_modal.sh
```

## 13. Final Notes

This starter code is intentionally structured so you can focus on algorithm implementation, not project plumbing.

If you are unsure where to begin, implement in this order:
1. `sft_trainer/sft.py::train`
2. `ipo_trainer/ipo.py::train`
3. `rloo_trainer/rloo_update_worker.py::update`

That order gives you a stable baseline, then preference tuning, then RL fine-tuning.
