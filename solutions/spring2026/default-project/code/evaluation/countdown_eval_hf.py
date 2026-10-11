"""Countdown pass@k with the training policy, no separate sampling engine.

The unbiased pass@k estimator is the one used by Chen et al. (Codex):
1 - C(n-c, k) / C(n, k), where c of n samples solve the prompt.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from countdown_objectives import load_named_split
from evaluation.countdown import compute_score


def pass_at_k(n, c, k):
    """Unbiased pass@k for one prompt."""
    if k > n:
        raise ValueError("k cannot exceed the number of samples")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--eval_dataset", default="asingh15/countdown_tasks_3to4")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output_dir", default="evaluation/eval_results")
    parser.add_argument("--output_name", default="eval")
    parser.add_argument("--num_responses", type=int, default=16)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--batch_prompts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    data = load_named_split(args.eval_dataset, args.split)
    if args.limit:
        data = data.select(range(min(args.limit, len(data))))

    records = []
    ks = [k for k in (1, 4, 8, 16) if k <= args.num_responses]
    totals = {k: 0.0 for k in ks}
    for start in range(0, len(data), args.batch_prompts):
        chunk = data.select(range(start, min(len(data), start + args.batch_prompts)))
        prompts = list(chunk["prompt"])
        repeated = [prompt for prompt in prompts for _ in range(args.num_responses)]
        encoded = tokenizer(
            repeated, return_tensors="pt", padding=True, truncation=True, max_length=512,
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        prompt_width = encoded["input_ids"].size(1)
        with torch.no_grad():
            sequences = model.generate(
                **encoded,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        texts = [
            tokenizer.decode(row[prompt_width:], skip_special_tokens=True) for row in sequences
        ]
        for local, row in enumerate(chunk):
            numbers = row["nums"] if "nums" in row else row["numbers"]
            truth = {"target": row["target"], "numbers": list(numbers)}
            group = texts[local * args.num_responses:(local + 1) * args.num_responses]
            scores = [float(compute_score(text, truth)) for text in group]
            correct = sum(score >= 1.0 for score in scores)
            item = {
                "prompt": row["prompt"],
                "target": row["target"],
                "nums": list(numbers),
                "scores": scores,
                "n_correct": correct,
            }
            for k in ks:
                value = pass_at_k(args.num_responses, correct, k)
                item[f"pass@{k}"] = value
                totals[k] += value
            records.append(item)
        done = len(records)
        summary = " ".join(f"pass@{k}={totals[k] / done:.3f}" for k in ks)
        print(f"evaluated {done}/{len(data)} {summary}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    mean = {f"pass@{k}": totals[k] / max(len(records), 1) for k in ks}
    mean["n"] = len(records)
    mean["num_responses"] = args.num_responses
    payload = {"summary": mean, "records": records}
    path = os.path.join(args.output_dir, args.output_name + ".json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    print(json.dumps(mean), flush=True)


if __name__ == "__main__":
    main()
