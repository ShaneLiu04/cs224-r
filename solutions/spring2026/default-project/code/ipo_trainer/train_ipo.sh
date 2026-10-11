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
gradient_accumulation_steps=16
gradient_clipping=0.0
weight_decay=0.01
warmup_ratio=0.05
wandb_project="dpo_default_project"
max_prompt_length=512
max_response_length=1024
output_dir="checkpoints/dpo_checkpoints/"
model_name='asingh15/qwen-sft-countdown-defaultproj'
dataset_name="asingh15/countdown_tasks_3to4-dpo"
wandb_project="dpo_default_project_0128"
average_logps=0

lrs=(
    5e-6
)
num_lrs=${#lrs[@]}

epochs=(
    1
)
num_epochs=${#epochs[@]}

betas=(
    0.1
)
num_betas=${#betas[@]}

loss_types=(
    'ipo'
)
num_loss_types=${#loss_types[@]}

if [[ $num_lrs -ne $num_epochs || $num_lrs -ne $num_betas || $num_lrs -ne $num_loss_types ]]; then
    echo "Error: num_lrs and num_epochs and num_betas and num_loss_types must be the same"
    exit 1
fi

for i in $(seq 0 $((num_lrs - 1))); do
    curr_lr=${lrs[$i]}
    curr_epochs=${epochs[$i]}
    curr_beta=${betas[$i]}
    curr_loss_type=${loss_types[$i]}
    wandb_name="loss${curr_loss_type}_lr${curr_lr}_beta${curr_beta}_ep${curr_epochs}"

    command="python ipo_trainer/ipo.py \
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
        --loss_type $curr_loss_type \
        --beta $curr_beta \
        --average_logps $average_logps \
    "

    echo "Executing command: $command"
    eval "$command"
done