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

batch_size=64
gradient_accumulation_steps=8
gradient_clipping=1.0
weight_decay=0.01
warmup_ratio=0.05
wandb_project="sft_default_project"
max_prompt_length=512
max_response_length=1024
output_dir="sft_model"
model_name="Qwen/Qwen2.5-0.5B"
dataset_name="Asap7772/cog_behav_all_strategies"
wandb_project="sft_default_project_0128"

lrs=(
    5e-5 
)
num_lrs=${#lrs[@]}

epochs=(
    6
)
num_epochs=${#epochs[@]}

if [ $num_lrs -ne $num_epochs ]; then
    echo "Error: num_lrs and num_epochs must be the same"
    exit 1
fi

for i in $(seq 0 $((num_lrs - 1))); do
    curr_lr=${lrs[$i]}
    curr_epochs=${epochs[$i]}
    wandb_name="lr_${curr_lr}_epochs_${curr_epochs}"

    command="python sft_trainer/sft.py \
        --model_name $model_name \
        --dataset_name $dataset_name \
        --output_dir $output_dir \
        --max_prompt_length $max_prompt_length \
        --max_response_length $max_response_length \
        --batch_size $batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --gradient_clipping $gradient_clipping \
        --num_epochs $curr_epochs \
        --learning_rate $curr_lr \
        --weight_decay $weight_decay \
        --warmup_ratio $warmup_ratio \
        --wandb_project $wandb_project \
        --wandb_name $wandb_name \
    "

    echo "Executing command: $command"
    eval "$command"
done