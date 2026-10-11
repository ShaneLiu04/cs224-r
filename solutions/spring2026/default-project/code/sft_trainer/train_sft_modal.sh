#!/bin/bash

# Launch the existing SFT trainer on Modal.
# Required before running:
#   export WANDB_API_KEY=...
#   export HF_TOKEN=...
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

batch_size="${BATCH_SIZE:-64}"
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-8}"
gradient_clipping="${GRADIENT_CLIPPING:-1.0}"
weight_decay="${WEIGHT_DECAY:-0.01}"
warmup_ratio="${WARMUP_RATIO:-0.05}"
max_prompt_length="${MAX_PROMPT_LENGTH:-512}"
max_response_length="${MAX_RESPONSE_LENGTH:-1024}"
output_dir="${OUTPUT_DIR:-/vol/checkpoints/sft_model}"
model_name="${MODEL_NAME:-Qwen/Qwen2.5-0.5B}"
dataset_name="${DATASET_NAME:-Asap7772/cog_behav_all_strategies}"
wandb_project="${WANDB_PROJECT:-sft_default_project_0128}"
save_model="${SAVE_MODEL:-1}"
gradient_checkpointing="${GRADIENT_CHECKPOINTING:-1}"

read -r -a lrs <<< "${LRS:-5e-5}"
read -r -a epochs <<< "${EPOCHS:-6}"

num_lrs=${#lrs[@]}
num_epochs=${#epochs[@]}

if [[ $num_lrs -ne $num_epochs ]]; then
    echo "Error: LRS and EPOCHS must have the same number of values"
    exit 1
fi

for i in "${!lrs[@]}"; do
    curr_lr="${lrs[$i]}"
    curr_epochs="${epochs[$i]}"
    wandb_name="${WANDB_NAME_PREFIX:-lr_${curr_lr}_epochs_${curr_epochs}}"

    command=(
        modal run "$PROJECT_ROOT/modal_train.py"
        sft
        --model_name "$model_name"
        --dataset_name "$dataset_name"
        --output_dir "$output_dir"
        --max_prompt_length "$max_prompt_length"
        --max_response_length "$max_response_length"
        --batch_size "$batch_size"
        --gradient_accumulation_steps "$gradient_accumulation_steps"
        --gradient_clipping "$gradient_clipping"
        --num_epochs "$curr_epochs"
        --learning_rate "$curr_lr"
        --weight_decay "$weight_decay"
        --warmup_ratio "$warmup_ratio"
        --wandb_project "$wandb_project"
        --wandb_name "$wandb_name"
        --save_model "$save_model"
        --gradient_checkpointing "$gradient_checkpointing"
    )

    printf 'Executing command: '
    printf '%q ' "${command[@]}"
    printf '\n'
    "${command[@]}"
done
