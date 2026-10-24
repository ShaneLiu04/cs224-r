"""Shared Countdown objectives for SFT, IPO, and RLOO.

The token mask treats a position as trainable only when it is both a response
token and not padding. Causal LM predicts token t from logits at t-1, so the
mask is shifted by one.
"""
import os

import torch
import torch.nn.functional as F


def load_named_split(dataset_name, split, num_proc=None):
    """Load a Hub split, or ``{split}.parquet`` from a local directory."""
    from datasets import load_dataset
    if os.path.isdir(dataset_name):
        path = os.path.join(dataset_name, f"{split}.parquet")
        import pyarrow.parquet as pq
        from datasets import Dataset
        table = pq.read_table(path).replace_schema_metadata(None)
        return Dataset(table)
    kwargs = {"split": split}
    if num_proc:
        kwargs["num_proc"] = num_proc
    return load_dataset(dataset_name, **kwargs)


def response_mask(is_response_token, attention_mask):
    """Return a float mask aligned with next-token targets."""
    response = is_response_token[:, 1:].to(dtype=torch.float32)
    visible = attention_mask[:, 1:].to(dtype=torch.float32)
    return response * visible


def causal_token_logprobs(model, input_ids, attention_mask, chunk=32):
    """Per-target log π(x_t | x_<t).

    The log-sum-exp is float32 and runs over time chunks, so the vocabulary
    axis is not stored as a full float32 tensor.
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    logits = outputs.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    pieces = []
    for start in range(0, logits.size(1), chunk):
        block = logits[:, start:start + chunk, :].float()
        chosen = targets[:, start:start + chunk].unsqueeze(-1)
        log_z = torch.logsumexp(block, dim=-1)
        pieces.append(block.gather(-1, chosen).squeeze(-1) - log_z)
    return logits, torch.cat(pieces, dim=1)


def masked_nll(token_logp, mask):
    """Mean negative log-likelihood over masked target tokens."""
    denom = mask.sum().clamp(min=1.0)
    return -(token_logp * mask).sum() / denom


def sequence_logprobs(token_logp, mask, average=False):
    """Sum (or mean) of token log-probs inside the response mask."""
    summed = (token_logp * mask).sum(dim=-1)
    if average:
        return summed / mask.sum(dim=-1).clamp(min=1.0)
    return summed


def token_entropy(logits, chunk=32):
    """Shannon entropy of the next-token distribution, chunked over time."""
    pieces = []
    for start in range(0, logits.size(1), chunk):
        block = logits[:, start:start + chunk, :].float()
        log_z = torch.logsumexp(block, dim=-1, keepdim=True)
        logp = block - log_z
        pieces.append(-(logp.exp() * logp).sum(dim=-1))
    return torch.cat(pieces, dim=1)


def token_kl(policy_logits, ref_logits, chunk=32):
    """Token-level KL(π || π_ref), chunked over time."""
    pieces = []
    width = policy_logits.size(1)
    for start in range(0, width, chunk):
        policy = policy_logits[:, start:start + chunk, :].float()
        ref = ref_logits[:, start:start + chunk, :].float()
        logp = policy - torch.logsumexp(policy, dim=-1, keepdim=True)
        logq = ref - torch.logsumexp(ref, dim=-1, keepdim=True)
        pieces.append((logp.exp() * (logp - logq)).sum(dim=-1))
    return torch.cat(pieces, dim=1)


def rloo_advantages(rewards, group_size):
    """Leave-one-out advantage. rewards is flat, groups are contiguous."""
    if group_size < 2:
        raise ValueError("group_size must be >= 2")
    if rewards.numel() % group_size != 0:
        raise ValueError("reward count must be divisible by group_size")
    grouped = rewards.view(-1, group_size)
    baseline = (grouped.sum(dim=1, keepdim=True) - grouped) / (group_size - 1)
    return (grouped - baseline).reshape(-1)


def grpo_advantages(rewards, group_size, eps=1e-4, scale_by_std=True):
    """GRPO advantage (r - mean) / (std + eps) within each contiguous group.

    scale_by_std=False gives the mean-centred variant (Dr. GRPO), which equals
    RLOO up to the factor (G - 1) / G.
    """
    if group_size < 2:
        raise ValueError("group_size must be >= 2")
    if rewards.numel() % group_size != 0:
        raise ValueError("reward count must be divisible by group_size")
    grouped = rewards.view(-1, group_size)
    centred = grouped - grouped.mean(dim=1, keepdim=True)
    if scale_by_std:
        centred = centred / (grouped.std(dim=1, keepdim=True) + eps)
    return centred.reshape(-1)


def preference_loss(delta_w, delta_l, beta, loss_type):
    """IPO square loss or DPO logistic loss on the pairwise log-ratio."""
    margin = delta_w - delta_l
    if loss_type == "ipo":
        target = 1.0 / (2.0 * beta)
        return ((margin - target) ** 2).mean()
    if loss_type == "dpo":
        return -F.logsigmoid(beta * margin).mean()
    raise ValueError("loss_type must be ipo or dpo")
