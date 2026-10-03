"""Upload a trained checkpoint to the Hugging Face Hub."""

from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default='PATH_TO_CHECKPOINT')
parser.add_argument('--base_model', type=str, default='Qwen/Qwen2.5-0.5B')
parser.add_argument('--output_name', type=str, default='OUTPUT_NAME')
 

args = parser.parse_args()

# We load tokenizer from base model to ensure expected tokenizer config.
tokenizer = AutoTokenizer.from_pretrained(args.base_model)
model = AutoModelForCausalLM.from_pretrained(args.model_path)

# Requires valid HF auth token in environment or local HF CLI login.
model.push_to_hub(args.output_name)
tokenizer.push_to_hub(args.output_name)
print(f"Model and tokenizer uploaded to {args.output_name}")
