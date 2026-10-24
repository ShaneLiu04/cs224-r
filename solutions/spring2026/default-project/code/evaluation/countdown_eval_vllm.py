"""Countdown pass@k, answer selection, and length/format statistics with vLLM.

Writes <output_name>.json (summary + per-prompt statistics) and
<output_name>.samples.jsonl.gz (raw texts) so selection can be re-run offline.
"""
import argparse
import gzip
import json
import os
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from countdown_objectives import load_named_split
from evaluation.selection import SELECTORS, featurise, pass_at_k


def score_prompt(samples, ks, rng, draws):
    """pass@k and selector accuracy at every budget k for one prompt."""
    n = len(samples)
    c = sum(s.verified for s in samples)
    item = {"n_correct": c}
    for k in ks:
        item[f"pass@{k}"] = pass_at_k(n, c, k)
        rounds = 1 if k == n else draws
        hits = dict.fromkeys(SELECTORS, 0)
        for _ in range(rounds):
            idx = list(range(n)) if k == n else rng.sample(range(n), k)
            subset = [samples[i] for i in idx]
            for name, select in SELECTORS.items():
                hits[name] += subset[select(subset)].verified
        for name in SELECTORS:
            item[f"{name}@{k}"] = hits[name] / rounds
    return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--eval_dataset", default="/root/autodl-tmp/data/countdown")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_name", required=True)
    parser.add_argument("--num_responses", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.35)
    parser.add_argument("--subset_draws", type=int, default=64)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    data = load_named_split(args.eval_dataset, args.split)
    if args.limit:
        data = data.select(range(min(args.limit, len(data))))
    rows = list(data)

    llm = LLM(
        model=args.model_path, dtype="bfloat16", seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=2048,
    )
    params = SamplingParams(
        n=args.num_responses, temperature=args.temperature, top_p=args.top_p,
        max_tokens=args.max_tokens, stop=["</answer>"],
        include_stop_str_in_output=True, logprobs=0, seed=args.seed,
    )
    start = time.time()
    outputs = llm.generate([row["prompt"] for row in rows], params)
    gen_seconds = time.time() - start

    n = args.num_responses
    ks = [k for k in (1, 2, 4, 8, 16, 32, 64) if k <= n]
    rng = random.Random(args.seed)
    records = []
    agg = dict.fromkeys(["samples", "tokens", "format", "numbers_ok", "truncated",
                         "correct", "tokens_correct"], 0)
    os.makedirs(args.output_dir, exist_ok=True)
    sample_path = os.path.join(args.output_dir, args.output_name + ".samples.jsonl.gz")
    with gzip.open(sample_path, "wt", encoding="utf-8") as sample_file:
        for row, out in zip(rows, outputs):
            numbers = [int(x) for x in (row["nums"] if "nums" in row else row["numbers"])]
            target = int(row["target"])
            texts = [c.text for c in out.outputs]
            lengths = [len(c.token_ids) for c in out.outputs]
            mean_logps = [(c.cumulative_logprob or 0.0) / max(len(c.token_ids), 1)
                          for c in out.outputs]
            samples = [featurise(t, lp, numbers, target) for t, lp in zip(texts, mean_logps)]
            item = score_prompt(samples, ks, rng, args.subset_draws)
            item.update({"target": target, "nums": numbers, "mean_len": sum(lengths) / n})
            records.append(item)
            for s, length, c in zip(samples, lengths, out.outputs):
                agg["samples"] += 1
                agg["tokens"] += length
                agg["format"] += bool(s.key)
                agg["numbers_ok"] += s.numbers_ok
                agg["truncated"] += c.finish_reason == "length"
                if s.verified:
                    agg["correct"] += 1
                    agg["tokens_correct"] += length
            sample_file.write(json.dumps({
                "target": target, "nums": numbers, "texts": texts, "lengths": lengths,
                "mean_logps": mean_logps, "correct": [s.verified for s in samples],
            }) + "\n")

    m = max(len(records), 1)
    metric_keys = [key for key in records[0] if "@" in key] if records else []
    summary = {key: sum(r[key] for r in records) / m for key in metric_keys}
    s = max(agg["samples"], 1)
    summary.update({
        "n_prompts": len(records), "num_responses": n,
        "mean_tokens": agg["tokens"] / s,
        "mean_tokens_correct": agg["tokens_correct"] / max(agg["correct"], 1),
        "format_rate": agg["format"] / s, "numbers_ok_rate": agg["numbers_ok"] / s,
        "truncation_rate": agg["truncated"] / s, "sample_accuracy": agg["correct"] / s,
        "gen_seconds": gen_seconds, "model_path": args.model_path,
        "split": args.split, "eval_dataset": args.eval_dataset,
        "temperature": args.temperature, "top_p": args.top_p, "max_tokens": args.max_tokens,
    })
    with open(os.path.join(args.output_dir, args.output_name + ".json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
