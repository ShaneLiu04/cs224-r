"""Batch evaluation script for Countdown checkpoints using vLLM."""

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential
import argparse
import os
from datasets import load_dataset, Dataset
import pandas as pd
from evaluation.countdown import compute_score

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
def load_checkpoint(model_path, max_model_len=2048, gpu_memory_utilization=0.9, max_num_batched_tokens=4096, enable_chunked_prefill=True, max_num_seqs=16):
    """Load tokenizer + vLLM engine with simple retry for transient failures."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path, max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization, max_num_batched_tokens=max_num_batched_tokens, enable_chunked_prefill=enable_chunked_prefill, max_num_seqs=max_num_seqs)
    return tokenizer, llm

def parse_args():
    """Parse command line arguments for evaluation run."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_model_len', type=int, default=2048)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9)
    parser.add_argument('--max_num_batched_tokens', type=int, default=4096)
    parser.add_argument('--enable_chunked_prefill', type=bool, default=True)
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--top_k', type=int, default=20)
    parser.add_argument('--min_p', type=float, default=0.0)
    parser.add_argument('--max_tokens', type=int, default=1024)
    parser.add_argument('--max_num_seqs', type=int, default=16)
    parser.add_argument('--model_path', type=str, default='PATH_TO_CHECKPOINT')
    parser.add_argument('--eval_dataset', type=str, default='asingh15/countdown_tasks_3to4')
    parser.add_argument('--output_dir', type=str, default='evaluation/eval_results')
    parser.add_argument('--output_name', type=str, default='OUTPUT_NAME')
    parser.add_argument('--num_responses', type=int, default=16)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # 1) Load model checkpoint into vLLM runtime.
    tokenizer, llm = load_checkpoint(args.model_path, args.max_model_len, args.gpu_memory_utilization, args.max_num_batched_tokens, args.enable_chunked_prefill, args.max_num_seqs)

    # 2) Configure sampling with pass@k style multiple responses.
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, min_p=args.min_p, max_tokens=args.max_tokens, n=args.num_responses)

    # 3) Load evaluation prompts and generate responses in batch.
    loaded_dataset = load_dataset(args.eval_dataset, split='test')
    prompt_df = loaded_dataset.to_pandas()
    outputs = llm.generate(prompt_df['prompt'], sampling_params)

    response = []
    scores = []
    for i, output in enumerate(outputs):
        row = prompt_df.iloc[i]
        prompt = output.prompt
        curr_response = []
        curr_scores = []
        for j, o in enumerate(output.outputs):
            generated_text = o.text

            # Compute task reward per sampled response.
            ground_truth = {
                'target': row['target'],
                'numbers': row['nums']
            }
            score = compute_score(generated_text, ground_truth)
            curr_response.append(generated_text)
            curr_scores.append(score)
        response.append(curr_response)
        scores.append(curr_scores)

    prompt_df['response'] = response
    prompt_df['scores'] = scores
    # Persist per-prompt sampled responses and scores as JSON for analysis.
    os.makedirs(args.output_dir, exist_ok=True)
    output_ds = Dataset.from_pandas(prompt_df)
    output_ds.to_json(f'{args.output_dir}/{args.output_name}.json')
