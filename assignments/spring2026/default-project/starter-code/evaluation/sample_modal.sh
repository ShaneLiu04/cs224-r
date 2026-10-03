#!/bin/bash

# Launch countdown evaluation on Modal.
# Required before running:
#   export HF_TOKEN=...
# Useful optional Modal overrides:
#   export MODAL_GPU=H100!
#   export MODAL_VOLUME_NAME=default-proj-training
#   export MODAL_TIMEOUT_SECONDS=86400

set -euo pipefail

export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export MODAL_GPU="${MODAL_GPU:-H100!}"
export MODAL_VOLUME_NAME="${MODAL_VOLUME_NAME:-default-proj-training}"
export MODAL_TIMEOUT_SECONDS="${MODAL_TIMEOUT_SECONDS:-86400}"
export MODAL_STARTUP_TIMEOUT_SECONDS="${MODAL_STARTUP_TIMEOUT_SECONDS:-1800}"

model_paths=(
    'MODEL PATH GOES HERE'
)
num_model_paths=${#model_paths[@]}

output_names=(
    'EVAL JSON NAME GOES HERE'
)
num_output_names=${#output_names[@]}

if [[ $num_model_paths -ne $num_output_names ]]; then
    echo "Number of model paths and output names do not match"
    exit 1
fi

eval_dataset="${EVAL_DATASET:-asingh15/countdown_tasks_3to4}"
output_dir="${OUTPUT_DIR:-/vol/evaluation/eval_results}"
max_model_len="${MAX_MODEL_LEN:-2048}"
gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.9}"
max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-4096}"
enable_chunked_prefill="${ENABLE_CHUNKED_PREFILL:-True}"
temperature="${TEMPERATURE:-0.6}"
top_p="${TOP_P:-0.95}"
top_k="${TOP_K:-20}"
min_p="${MIN_P:-0.0}"
max_tokens="${MAX_TOKENS:-1024}"
max_num_seqs="${MAX_NUM_SEQS:-16}"
num_responses="${NUM_RESPONSES:-16}"

mkdir -p "$PROJECT_ROOT/logs"

for i in "${!model_paths[@]}"; do
    model_path="${model_paths[$i]}"
    output_name="${output_names[$i]}"
    log_file="$PROJECT_ROOT/logs/${output_name}.modal.log"

    command=(
        modal run "$PROJECT_ROOT/modal_train.py"
        eval
        --model_path "$model_path"
        --eval_dataset "$eval_dataset"
        --output_dir "$output_dir"
        --output_name "$output_name"
        --max_model_len "$max_model_len"
        --gpu_memory_utilization "$gpu_memory_utilization"
        --max_num_batched_tokens "$max_num_batched_tokens"
        --enable_chunked_prefill "$enable_chunked_prefill"
        --temperature "$temperature"
        --top_p "$top_p"
        --top_k "$top_k"
        --min_p "$min_p"
        --max_tokens "$max_tokens"
        --max_num_seqs "$max_num_seqs"
        --num_responses "$num_responses"
    )

    printf 'Running command: '
    printf '%q ' "${command[@]}"
    printf '\n'
    echo "Logging to $log_file"
    "${command[@]}" > "$log_file" 2>&1 &
done

wait
