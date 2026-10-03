#!/bin/bash

# Launch the existing RLOO trainer on Modal.
# Required before running:
#   export WANDB_API_KEY=...
#   export HF_TOKEN=...
#   export MODEL_NAME=path-or-hf-repo-of-initial-policy
#   export DATASET_NAME=your-rloo-dataset
# Useful optional Modal overrides:
#   export MODAL_GPU=H100!
#   export MODAL_VOLUME_NAME=default-proj-training
#   export MODAL_TIMEOUT_SECONDS=86400  # 24 hours

set -euo pipefail

export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export MODAL_GPU="${MODAL_GPU:-H100!}"
export MODAL_VOLUME_NAME="${MODAL_VOLUME_NAME:-default-proj-training}"
export MODAL_TIMEOUT_SECONDS="${MODAL_TIMEOUT_SECONDS:-86400}"
export MODAL_STARTUP_TIMEOUT_SECONDS="${MODAL_STARTUP_TIMEOUT_SECONDS:-1800}"

read -r -a lrs <<< "${LRS:-1e-5}"
num_lrs=${#lrs[@]}

which_exp=${1:-0}
if (( which_exp < 0 || which_exp >= num_lrs )); then
    echo "Error: which_exp must be between 0 and $((num_lrs - 1))"
    exit 1
fi

curr_lr="${lrs[$which_exp]}"

batch_size="${BATCH_SIZE:-128}"
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-128}"
gradient_clipping="${GRADIENT_CLIPPING:-0.0}"
group_size="${GROUP_SIZE:-8}"
num_training_steps="${NUM_TRAINING_STEPS:-100}"
kl_divergence_coefficient="${KL_DIVERGENCE_COEFFICIENT:-0.001}"
entropy_coefficient="${ENTROPY_COEFFICIENT:-0.001}"
save_every_n_steps="${SAVE_EVERY_N_STEPS:-10}"
lr_schedule="${LR_SCHEDULE:-constant}"
warmup_ratio="${WARMUP_RATIO:-0.0}"
weight_decay="${WEIGHT_DECAY:-1e-4}"
temperature="${TEMPERATURE:-1.0}"
top_k="${TOP_K:--1}"
top_p="${TOP_P:-1.0}"
min_p="${MIN_P:-0.0}"

tokenizer_name="${TOKENIZER_NAME:-Qwen/Qwen2.5-0.5B}"
model_name="${MODEL_NAME:-your-org/your-model}"
dataset_name="${DATASET_NAME:-your-org/your-dataset}"
wandb_project="${WANDB_PROJECT:-rloo_training}"
save_dir="${SAVE_DIR:-/vol/checkpoints/rloo_checkpoints}"

wandb_name="${WANDB_NAME:-rloo_neb_lr${curr_lr}_bs${batch_size}_gs${group_size}_ent${entropy_coefficient}_kl${kl_divergence_coefficient}_lr${lr_schedule}_warmup${warmup_ratio}_temp${temperature}_topp${top_p}_topk${top_k}}"

command=(
    modal run "$PROJECT_ROOT/modal_train.py"
    rloo
    --model_name "$model_name"
    --ref_model_name "$model_name"
    --tokenizer_name "$tokenizer_name"
    --dataset_name "$dataset_name"
    --wandb_project "$wandb_project"
    --wandb_name "$wandb_name"
    --learning_rate "$curr_lr"
    --batch_size "$batch_size"
    --gradient_accumulation_steps "$gradient_accumulation_steps"
    --gradient_clipping "$gradient_clipping"
    --group_size "$group_size"
    --entropy_coefficient "$entropy_coefficient"
    --kl_divergence_coefficient "$kl_divergence_coefficient"
    --num_training_steps "$num_training_steps"
    --lr_schedule "$lr_schedule"
    --save_every_n_steps "$save_every_n_steps"
    --save_dir "$save_dir"
    --warmup_ratio "$warmup_ratio"
    --weight_decay "$weight_decay"
    --temperature "$temperature"
    --top_p "$top_p"
    --top_k "$top_k"
    --min_p "$min_p"
)

printf 'Executing command: '
printf '%q ' "${command[@]}"
printf '\n'
"${command[@]}"
