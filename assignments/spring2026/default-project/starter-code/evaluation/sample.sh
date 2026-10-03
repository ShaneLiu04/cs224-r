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

gpus=(
    0
)
num_gpus=${#gpus[@]}

model_paths=(
    'MODEL PATH GOES HERE'
)
num_model_paths=${#model_paths[@]}

output_names=(
    'EVAL JSON NAME GOES HERE'
)
num_output_names=${#output_names[@]}

if [ $num_model_paths -ne $num_output_names ]; then
    echo "Number of model paths and output names do not match"
    exit 1
fi 

if [ $num_model_paths -gt $num_gpus ]; then
    echo "Number of model paths is greater than number of GPUs"
    exit 1
fi 

for i in $(seq 0 $((num_model_paths - 1))); do
    model_path=${model_paths[$i]}
    output_name=${output_names[$i]}
    gpu=${gpus[$i]}
    command="CUDA_VISIBLE_DEVICES=$gpu python evaluation/countdown_eval.py --model_path $model_path --output_name $output_name"
    echo "Running command: $command"
    mkdir -p logs
    eval $command > logs/$output_name.log 2>&1 &
done
wait