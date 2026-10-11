"""In-process RLOO for Countdown.

train_rloo.sh samples at temperature 1 with top-p 1, so the behavior policy
is the current model. This driver generates with that policy, scores the
Countdown reward, and applies the same leave-one-out update as
RLOOUpdateWorker.update. It does not reload a second engine between steps.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from countdown_objectives import (
    causal_token_logprobs,
    response_mask,
    rloo_advantages,
    sequence_logprobs,
    token_entropy,
    token_kl,
)
from evaluation.countdown import compute_score


class StopOnTokens(StoppingCriteria):
    """Stop a row when its tail matches any answer-closing token sequence."""

    def __init__(self, sequences):
        self.sequences = [list(item) for item in sequences if item]

    def __call__(self, input_ids, scores, **kwargs):
        hits = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)
        for stop_ids in self.sequences:
            width = len(stop_ids)
            if input_ids.size(1) < width:
                continue
            tail = input_ids[:, -width:]
            target = torch.tensor(stop_ids, device=input_ids.device).view(1, -1)
            hits = hits | (tail == target).all(dim=-1)
        return hits


def load_policy(model_name, checkpointing):
    """Load a bf16 policy. An optional frozen copy is created by the caller."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.train()
    return model, tokenizer


def read_rows(dataset_name, split, limit):
    """Load prompts and Countdown ground truth."""
    from countdown_objectives import load_named_split
    data = load_named_split(dataset_name, split)
    rows = []
    for row in data:
        if "ground_truth" in row and isinstance(row["ground_truth"], dict):
            truth = row["ground_truth"]
        else:
            numbers = row.get("nums", row.get("numbers"))
            truth = {"target": row["target"], "numbers": list(numbers)}
        rows.append((row["prompt"], truth))
        if limit and len(rows) >= limit:
            break
    return rows


def generate_group(model, tokenizer, prompts, group_size, max_new_tokens, stop_ids):
    """Sample group_size continuations for each prompt."""
    was_training = model.training
    checkpointing = getattr(model, "is_gradient_checkpointing", False)
    if checkpointing:
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()
    repeated = [prompt for prompt in prompts for _ in range(group_size)]
    encoded = tokenizer(
        repeated, return_tensors="pt", padding=True, truncation=True, max_length=512,
    )
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    prompt_width = encoded["input_ids"].size(1)
    stopper = StoppingCriteriaList([StopOnTokens(stop_ids)]) if stop_ids else None
    with torch.no_grad():
        sequences = model.generate(
            **encoded,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=stopper,
            use_cache=True,
        )
    if checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    if was_training:
        model.train()
    return sequences, encoded["attention_mask"], prompt_width


def response_text(tokenizer, sequences, prompt_width):
    """Decode only the newly generated tokens."""
    texts = []
    for row in sequences:
        texts.append(tokenizer.decode(row[prompt_width:], skip_special_tokens=True))
    return texts


def _first_cut(generated, eos_id, stop_sequences):
    """Index just after the first EOS or answer-closing sequence."""
    hits = (generated == eos_id).nonzero(as_tuple=False)
    cut = int(hits[0]) + 1 if hits.numel() else None
    for stop_ids in stop_sequences:
        width = len(stop_ids)
        if generated.numel() < width:
            continue
        stop = torch.tensor(stop_ids, device=generated.device)
        for start in range(int(generated.numel()) - width + 1):
            if torch.equal(generated[start:start + width], stop):
                end = start + width
                cut = end if cut is None else min(cut, end)
                break
    return cut


def build_batch(sequences, prompt_attention, prompt_width, eos_id, stop_sequences):
    """Mask padding, tokens after EOS, and tokens after the closing answer tag."""
    attention = torch.ones_like(sequences)
    attention[:, :prompt_width] = prompt_attention
    is_response = torch.zeros_like(sequences)
    is_response[:, prompt_width:] = 1
    for row in range(sequences.size(0)):
        cut = _first_cut(sequences[row, prompt_width:], eos_id, stop_sequences)
        if cut is None:
            continue
        attention[row, prompt_width + cut:] = 0
        is_response[row, prompt_width + cut:] = 0
    return sequences, attention, is_response


def rloo_backward(
    model, ref_model, input_ids, attention, is_response, rewards, group_size,
    entropy_coefficient, kl_coefficient, accum_steps,
):
    """One RLOO backward. Returns detached metric floats."""
    advantages = rloo_advantages(rewards, group_size)
    logits, token_logp = causal_token_logprobs(model, input_ids, attention)
    mask = response_mask(is_response, attention)
    seq_logp = sequence_logprobs(token_logp, mask, average=False)
    weights = torch.ones_like(seq_logp)
    pg_loss = -(weights * advantages.detach() * seq_logp).mean()
    entropy_value = torch.zeros((), device=input_ids.device)
    entropy_bonus = 0.0
    if entropy_coefficient:
        ent = token_entropy(logits)
        entropy_value = (ent * mask).sum() / mask.sum().clamp(min=1.0)
        entropy_bonus = -entropy_coefficient * entropy_value
    kl_value = torch.zeros((), device=input_ids.device)
    if kl_coefficient > 0:
        with torch.no_grad():
            ref_logits, _ = causal_token_logprobs(ref_model, input_ids, attention)
        kl_tokens = token_kl(logits, ref_logits.detach())
        del ref_logits
        kl_value = (kl_tokens * mask).sum() / mask.sum().clamp(min=1.0)
    loss = pg_loss + entropy_bonus + kl_coefficient * kl_value
    (loss / accum_steps).backward()
    return {
        "loss": float(loss.detach().item()),
        "pg_loss": float(pg_loss.detach().item()),
        "entropy": float(entropy_value.detach().item()),
        "kl_loss": float(kl_value.detach().item()),
        "advantage_mean": float(advantages.detach().mean().item()),
        "advantage_abs_mean": float(advantages.detach().abs().mean().item()),
        "reward_mean": float(rewards.mean().item()),
        "ratio_mean": 1.0,
        "reward_exact": float((rewards >= 1.0).float().mean().item()),
        "reward_format": float(((rewards > 0) & (rewards < 1)).float().mean().item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--dataset_name", default="asingh15/countdown_tasks_3to4")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", default="rloo_local")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--prompts_per_step", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_training_steps", type=int, default=40)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--entropy_coefficient", type=float, default=0.001)
    parser.add_argument("--kl_divergence_coefficient", type=float, default=0.001)
    parser.add_argument("--gradient_clipping", type=float, default=0.0)
    parser.add_argument("--gradient_checkpointing", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    model, tokenizer = load_policy(args.model_name, args.gradient_checkpointing)
    ref_model = None
    if args.kl_divergence_coefficient > 0:
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    rows = read_rows(args.dataset_name, args.split, args.limit)
    stop_sequences = [
        tokenizer.encode("</answer>", add_special_tokens=False),
        tokenizer.encode(" </answer>", add_special_tokens=False),
    ]
    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    cursor = 0
    micro = 0
    optimizer.zero_grad(set_to_none=True)

    for step in range(args.num_training_steps):
        batch_rows = []
        for _ in range(args.prompts_per_step):
            batch_rows.append(rows[cursor % len(rows)])
            cursor += 1
        prompts = [item[0] for item in batch_rows]
        truths = [item[1] for item in batch_rows]
        sequences, prompt_attention, prompt_width = generate_group(
            model, tokenizer, prompts, args.group_size, args.max_new_tokens, stop_sequences,
        )
        texts = response_text(tokenizer, sequences, prompt_width)
        rewards = []
        for index, text in enumerate(texts):
            rewards.append(compute_score(text, truths[index // args.group_size]))
        rewards_t = torch.tensor(rewards, device=model.device, dtype=torch.float32)
        input_ids, attention, is_response = build_batch(
            sequences, prompt_attention, prompt_width, tokenizer.eos_token_id, stop_sequences,
        )
        stats = rloo_backward(
            model, ref_model, input_ids, attention, is_response, rewards_t,
            args.group_size, args.entropy_coefficient, args.kl_divergence_coefficient,
            args.gradient_accumulation_steps,
        )
        micro += 1
        grad_norm = 0.0
        if micro % args.gradient_accumulation_steps == 0:
            if args.gradient_clipping > 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.gradient_clipping,
                ))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        stats.update({
            "step": step + 1,
            "lr": args.learning_rate,
            "grad_norm": grad_norm,
            "group_size": args.group_size,
            "entropy_coefficient": args.entropy_coefficient,
            "kl_divergence_coefficient": args.kl_divergence_coefficient,
            "seed": args.seed,
        })
        with open(metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(stats) + "\n")
        print(
            f"step {step + 1} reward {stats['reward_mean']:.3f} "
            f"exact {stats['reward_exact']:.3f} loss {stats['loss']:.4f}",
            flush=True,
        )
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
