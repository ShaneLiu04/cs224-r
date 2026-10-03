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
# export HF_TOKEN=your_hf_token  # Required for uploading to Hugging Face

base_model='Qwen/Qwen2.5-0.5B'

all_checkpoint_paths=(
    'PATH_TO_CHECKPOINT'
)
num_checkpoint_paths=${#all_checkpoint_paths[@]}

output_names=(
    'OUTPUT_NAME'
)
num_output_names=${#output_names[@]}

if [ $num_checkpoint_paths -ne $num_output_names ]; then
    echo "Number of checkpoint paths and output names do not match"
    exit 1
fi

for i in $(seq 0 $((num_checkpoint_paths - 1))); do
    checkpoint_path=${all_checkpoint_paths[$i]}
    output_name=${output_names[$i]}
    echo "Uploading model to $checkpoint_path to $output_name"
    command="HF_HUB_ENABLE_HF_TRANSFER=1 python sft_trainer/upload_sft.py --model_path $checkpoint_path --base_model $base_model --output_name $output_name"
    echo "Running command: $command"
    eval $command &
done
wait
echo "All models uploaded"