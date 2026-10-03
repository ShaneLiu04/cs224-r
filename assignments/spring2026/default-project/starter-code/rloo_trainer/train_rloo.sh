#!/bin/bash
# Ensure conda is available (optional - remove if not using conda)
# eval "$(conda shell.bash hook)"
# conda activate default_proj

# Set environment variables (set these in your shell or .env before running)
# export WANDB_API_KEY=your_wandb_api_key
# export WANDB_USERNAME=your_username
# export WANDB_USER_EMAIL=your_email
export WANDB__SERVICE_WAIT=300
# export WANDB_ENTITY=your_entity
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HOME/.cache/huggingface}
# export HF_TOKEN=your_hf_token  # Required for gated datasets

set -e  # Exit immediately if a command exits with a non-zero status

lrs=(1e-5)
num_lrs=${#lrs[@]}

gpus=(0)
num_gpus=${#gpus[@]}

if [[ $num_lrs -ne $num_gpus ]]; then
    echo "Error: num_lrs and num_gpus must be the same"
    exit 1
fi

which_exp=${1:-0}
if [[ $which_exp -lt 0 || $which_exp -ge $num_lrs ]]; then
    echo "Error: which_exp must be between 0 and $num_lrs - 1"
    exit 1
fi

curr_lr=${lrs[$which_exp]}
curr_gpu=${gpus[$which_exp]}

batch_size=128
gradient_accumulation_steps=128
gradient_clipping=0.0
group_size=8
num_training_steps=100
kl_divergence_coefficient=0.001
entropy_coefficient=0.001
save_every_n_steps=10
lr_schedule='constant'
warmup_ratio=0.0
weight_decay=1e-4
temperature=1.0
top_k=-1
top_p=1.0
min_p=0.0

tokenizer_name='Qwen/Qwen2.5-0.5B'
model_name="${MODEL_NAME:-your-org/your-model}"
dataset_name="${DATASET_NAME:-your-org/your-dataset}"
wandb_project="${WANDB_PROJECT:-rloo_training}"
save_dir="${SAVE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/checkpoints/rloo_checkpoints}"


wandb_name="rloo_neb_lr${curr_lr}_bs${batch_size}_gs${group_size}_ent${entropy_coefficient}_kl${kl_divergence_coefficient}_lr${lr_schedule}_warmup${warmup_ratio}_temp${temperature}_topp${top_p}_topk${top_k}"
log_file="$PWD/logs/${wandb_project}/${wandb_name}.log"
log_dir=$(dirname $log_file)
mkdir -p $log_dir

command="CUDA_VISIBLE_DEVICES=$curr_gpu python rloo_trainer/rloo.py \
    --model_name $model_name \
    --ref_model_name $model_name \
    --tokenizer_name $tokenizer_name \
    --dataset_name $dataset_name \
    --wandb_project $wandb_project \
    --wandb_name $wandb_name \
    --learning_rate $curr_lr \
    --batch_size $batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --gradient_clipping $gradient_clipping \
    --group_size $group_size \
    --entropy_coefficient $entropy_coefficient \
    --kl_divergence_coefficient $kl_divergence_coefficient \
    --num_training_steps $num_training_steps \
    --lr_schedule $lr_schedule \
    --save_every_n_steps $save_every_n_steps \
    --save_dir $save_dir \
    --warmup_ratio $warmup_ratio \
    --weight_decay $weight_decay \
    --temperature $temperature \
    --top_p $top_p \
    --top_k $top_k \
    --min_p $min_p \
"

echo "Executing command: $command"
echo "Logging to $log_file"
eval "$command" > $log_file 2>&1