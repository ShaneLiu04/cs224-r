"""On-policy RLOO / GRPO for Countdown with a colocated vLLM sampler.

Sampling uses vLLM in the same process; after every optimizer step the
policy weights are copied into the vLLM model and its prefix cache is reset,
so every batch is sampled from the current policy (temperature 1, top-p 1).
The loss is the same as RLOOUpdateWorker: -(A * sum_t log pi) averaged over
samples, plus the entropy bonus and token-level KL to the frozen start model.
Only the advantage estimator changes between --algo values.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from countdown_objectives import (
    grpo_advantages, load_named_split, response_mask, rloo_advantages,
)
from evaluation.countdown import compute_score
from evaluation.selection import featurise, pass_at_k


def advantages_for(algo, rewards, group_size):
    if algo == "rloo":
        return rloo_advantages(rewards, group_size)
    if algo == "grpo":
        return grpo_advantages(rewards, group_size, scale_by_std=True)
    if algo == "drgrpo":
        return grpo_advantages(rewards, group_size, scale_by_std=False)
    raise ValueError(algo)


def read_rows(path, split, start=0, limit=0):
    data = load_named_split(path, split)
    rows = []
    for index in range(start, len(data)):
        row = data[index]
        numbers = [int(x) for x in (row["nums"] if "nums" in row else row["numbers"])]
        rows.append({"prompt": row["prompt"], "target": int(row["target"]), "nums": numbers})
        if limit and len(rows) >= limit:
            break
    return rows


def _chunk_terms(logits, targets, ref_logits):
    """Token log-prob, entropy, and KL(pi || ref) for one time chunk (float32)."""
    block = logits.float()
    logp_all = block - torch.logsumexp(block, dim=-1, keepdim=True)
    token_logp = logp_all.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    probs = logp_all.exp()
    entropy = -(probs * logp_all).sum(dim=-1)
    if ref_logits is None:
        return token_logp, entropy, torch.zeros_like(entropy)
    ref = ref_logits.float()
    ref_logp = ref - torch.logsumexp(ref, dim=-1, keepdim=True)
    kl = (probs * (logp_all - ref_logp)).sum(dim=-1)
    return token_logp, entropy, kl


def token_terms(logits, input_ids, ref_logits=None, chunk=64):
    """Checkpointed per-chunk terms: only the bf16 logits are kept for backward."""
    from torch.utils.checkpoint import checkpoint
    targets = input_ids[:, 1:]
    out = ([], [], [])
    for start in range(0, logits.size(1), chunk):
        ref = None if ref_logits is None else ref_logits[:, start:start + chunk]
        terms = checkpoint(_chunk_terms, logits[:, start:start + chunk],
                           targets[:, start:start + chunk], ref, use_reentrant=False)
        for bucket, value in zip(out, terms):
            bucket.append(value)
    return tuple(torch.cat(bucket, dim=1) for bucket in out)


def vllm_model(llm):
    engine = llm.llm_engine
    executor = getattr(engine, "model_executor", None)
    if executor is None:
        executor = engine.engine_core.engine_core.model_executor
    worker = executor.driver_worker
    worker = getattr(worker, "worker", worker)
    return worker.model_runner.model


def sync_weights(llm, model):
    target = vllm_model(llm)
    with torch.no_grad():
        target.load_weights(((name, param.data) for name, param in model.named_parameters()))
    llm.reset_prefix_cache()


def pad_batch(seqs, pad_id, device):
    width = max(len(ids) for ids, _ in seqs)
    input_ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(seqs), width), dtype=torch.long)
    is_response = torch.zeros((len(seqs), width), dtype=torch.long)
    for row, (ids, prompt_len) in enumerate(seqs):
        input_ids[row, :len(ids)] = torch.tensor(ids)
        attention[row, :len(ids)] = 1
        is_response[row, prompt_len:len(ids)] = 1
    return input_ids.to(device), attention.to(device), is_response.to(device)


def evaluate(llm, rows, n, max_tokens, seed):
    from vllm import SamplingParams
    params = SamplingParams(n=n, temperature=0.6, top_p=0.95, max_tokens=max_tokens,
                            stop=["</answer>"], include_stop_str_in_output=True, seed=seed)
    outputs = llm.generate([r["prompt"] for r in rows], params, use_tqdm=False)
    p1 = pn = tokens = fmt = 0.0
    for row, out in zip(rows, outputs):
        samples = [featurise(c.text, 0.0, row["nums"], row["target"]) for c in out.outputs]
        c = sum(s.verified for s in samples)
        p1 += pass_at_k(n, c, 1)
        pn += pass_at_k(n, c, n)
        tokens += sum(len(o.token_ids) for o in out.outputs) / n
        fmt += sum(bool(s.key) for s in samples) / n
    m = len(rows)
    return {"eval_pass@1": p1 / m, f"eval_pass@{n}": pn / m,
            "eval_mean_tokens": tokens / m, "eval_format_rate": fmt / m}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--train_dataset", default="/root/autodl-tmp/data/countdown")
    parser.add_argument("--eval_dataset", default="/root/autodl-tmp/data/countdown_heldout")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--algo", choices=["rloo", "grpo", "drgrpo"], default="rloo")
    parser.add_argument("--curriculum", choices=["none", "nums"], default="none")
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--prompts_per_step", type=int, default=32)
    parser.add_argument("--num_training_steps", type=int, default=100)
    parser.add_argument("--micro_batch", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--entropy_coefficient", type=float, default=0.001)
    parser.add_argument("--kl_divergence_coefficient", type=float, default=0.001)
    parser.add_argument("--gradient_clipping", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--eval_limit", type=int, default=200)
    parser.add_argument("--eval_n", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_model", type=int, default=1)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=1)

    llm = LLM(model=args.model_name, dtype="bfloat16", seed=args.seed,
              gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=2048)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    ref_model = None
    if args.kl_divergence_coefficient > 0:
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
        ref_model.eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)

    total_prompts = args.prompts_per_step * args.num_training_steps
    pool = read_rows(args.train_dataset, "train", 0, total_prompts * 3)
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    if args.curriculum == "nums":
        easy = [r for r in pool if len(r["nums"]) == 3]
        hard = [r for r in pool if len(r["nums"]) == 4]
        half = total_prompts // 2
        schedule = easy[:half] + hard[:total_prompts - half]
    else:
        schedule = pool[:total_prompts]
    eval_rows = read_rows(args.eval_dataset, "test", 0, args.eval_limit)

    sample_params = SamplingParams(
        n=args.group_size, temperature=1.0, top_p=1.0, max_tokens=args.max_new_tokens,
        stop=["</answer>"], include_stop_str_in_output=True)
    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    open(metrics_path, "w").close()

    def log(row):
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    log({"step": 0, **evaluate(llm, eval_rows, args.eval_n, args.max_new_tokens, args.seed)})
    for step in range(1, args.num_training_steps + 1):
        batch = schedule[(step - 1) * args.prompts_per_step: step * args.prompts_per_step]
        t0 = time.time()
        outputs = llm.generate([r["prompt"] for r in batch], sample_params, use_tqdm=False)
        t_gen = time.time() - t0

        seqs, rewards, lengths, truncated = [], [], [], 0
        for row, out in zip(batch, outputs):
            prompt_ids = list(out.prompt_token_ids)
            for comp in out.outputs:
                ids = prompt_ids + list(comp.token_ids)
                seqs.append((ids, len(prompt_ids)))
                rewards.append(float(compute_score(comp.text, {"target": row["target"],
                                                               "numbers": row["nums"]})))
                lengths.append(len(comp.token_ids))
                truncated += comp.finish_reason == "length"
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        adv = advantages_for(args.algo, rewards_t, args.group_size)
        grouped = rewards_t.view(-1, args.group_size)
        zero_var = float((grouped.max(dim=1).values == grouped.min(dim=1).values).float().mean())

        t1 = time.time()
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]))
        n_total = len(seqs)
        stats = {"pg": 0.0, "ent": 0.0, "kl": 0.0}
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, n_total, args.micro_batch):
            idx = order[start:start + args.micro_batch]
            input_ids, attention, is_response = pad_batch([seqs[i] for i in idx], pad_id, "cuda")
            a = adv[idx].cuda()
            ref_logits = None
            if ref_model is not None:
                with torch.no_grad():
                    ref_logits = ref_model(input_ids=input_ids, attention_mask=attention,
                                           use_cache=False).logits[:, :-1, :]
            logits = model(input_ids=input_ids, attention_mask=attention,
                           use_cache=False).logits[:, :-1, :]
            token_logp, ent_tok, kl_tok = token_terms(logits, input_ids, ref_logits)
            mask = response_mask(is_response, attention)
            seq_logp = (token_logp * mask).sum(dim=-1)
            frac = len(idx) / n_total
            pg = -(a * seq_logp).sum() / n_total
            denom = mask.sum().clamp(min=1.0)
            ent = (ent_tok * mask).sum() / denom
            kl = (kl_tok * mask).sum() / denom
            loss = (pg - args.entropy_coefficient * ent * frac
                    + args.kl_divergence_coefficient * kl * frac)
            loss.backward()
            stats["pg"] += float(pg.detach())
            stats["ent"] += float(ent.detach()) * frac
            stats["kl"] += float(kl.detach()) * frac
            del logits, ref_logits, token_logp, ent_tok, kl_tok
        grad_norm = 0.0
        if args.gradient_clipping > 0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                             args.gradient_clipping))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        sync_weights(llm, model)
        t_train = time.time() - t1

        row = {
            "step": step, "reward_mean": float(rewards_t.mean()),
            "reward_exact": float((rewards_t >= 1.0).float().mean()),
            "reward_format": float(((rewards_t > 0) & (rewards_t < 1)).float().mean()),
            "reward_none": float((rewards_t == 0).float().mean()),
            "zero_variance_groups": zero_var,
            "adv_abs_mean": float(adv.abs().mean()),
            "mean_tokens": sum(lengths) / len(lengths),
            "truncation_rate": truncated / len(lengths),
            "pg_loss": stats["pg"], "entropy": stats["ent"], "kl": stats["kl"],
            "grad_norm": grad_norm, "frac_3nums": sum(len(r["nums"]) == 3 for r in batch) / len(batch),
            "t_gen": t_gen, "t_train": t_train,
        }
        if step % args.eval_every == 0 or step == args.num_training_steps:
            row.update(evaluate(llm, eval_rows, args.eval_n, args.max_new_tokens, args.seed))
        log(row)
        print(json.dumps(row), flush=True)

    if args.save_model:
        model.save_pretrained(os.path.join(args.output_dir, "model"))
        tokenizer.save_pretrained(os.path.join(args.output_dir, "model"))


if __name__ == "__main__":
    main()
