"""Starter IPO training entrypoint for the class project.

This script wires model loading, data loading, and optimizer setup.
Students are expected to implement `train(...)` for the IPO objective.
"""

import sys
from pathlib import Path

# Allow `python ipo_trainer/ipo.py` to resolve imports from project root.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
import gc
import argparse
import os
from ipo_trainer.ipo_dataset import get_dataloaders
try:
    import wandb
except ImportError:
    class wandb:
        """JSONL is the record of record when wandb is not installed."""
        @staticmethod
        def init(*args, **kwargs):
            return wandb
        @staticmethod
        def log(*args, **kwargs):
            return None
        class config:
            @staticmethod
            def update(*args, **kwargs):
                return None
import json
import tqdm.auto as tqdm
import copy
from countdown_objectives import (
    causal_token_logprobs,
    preference_loss,
    response_mask,
    sequence_logprobs,
)
# os.environ['WANDB_MODE'] = 'offline'

def get_model(model_name, device, use_gradient_checkpointing=True):
    """Load trainable policy model and frozen reference model."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Enable gradient checkpointing to reduce memory (trades compute for memory)
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    
    model.train()

    # A second load stays frozen. deepcopy breaks on sharded device maps.
    reference_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad_(False)
    return model, tokenizer, reference_model

def clear_cache(model):
    """Best-effort GPU/CPU cache cleanup between heavy steps."""
    torch.cuda.empty_cache()
    gc.collect()

def save_checkpoint(model, tokenizer, optimizer, scheduler, output_dir):
    """Save model/tokenizer plus optimizer/scheduler states."""
    os.makedirs(output_dir, exist_ok=True)

    model_dir = os.path.join(output_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"Model and tokenizer saved to {model_dir}")

    torch.save({
        'scheduler': scheduler.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, os.path.join(output_dir, 'train_states.pth'))
    print(f"Model saved to {output_dir}")

def train(
    model, 
    tokenizer, 
    reference_model,
    train_dataloader, 
    test_dataloader, 
    optimizer, 
    scheduler, 
    num_epochs, 
    device='cuda', 
    save_model=1, 
    output_dir='sft_model', 
    gradient_accumulation_steps=1, 
    gradient_clipping=1.0,
    beta=0.1,
    average_logps=False,
    loss_type='ipo',
    max_steps=0,
):
    device = next(model.parameters()).device
    metrics_path = os.path.join(output_dir, "metrics.jsonl")
    os.makedirs(output_dir, exist_ok=True)
    average = bool(average_logps)
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    micro = 0

    def pair_logprobs(active_model, input_ids, attention_mask, is_response):
        _, token_logp = causal_token_logprobs(active_model, input_ids, attention_mask)
        mask = response_mask(is_response, attention_mask)
        return sequence_logprobs(token_logp, mask, average=average), mask

    def batch_loss(batch):
        chosen_ids = batch["input_ids_w"].to(device)
        chosen_mask = batch["attention_mask_w"].to(device)
        chosen_resp = batch["is_response_token_w"].to(device)
        rejected_ids = batch["input_ids_l"].to(device)
        rejected_mask = batch["attention_mask_l"].to(device)
        rejected_resp = batch["is_response_token_l"].to(device)
        policy_w, _ = pair_logprobs(model, chosen_ids, chosen_mask, chosen_resp)
        policy_l, _ = pair_logprobs(model, rejected_ids, rejected_mask, rejected_resp)
        with torch.no_grad():
            ref_w, _ = pair_logprobs(reference_model, chosen_ids, chosen_mask, chosen_resp)
            ref_l, _ = pair_logprobs(reference_model, rejected_ids, rejected_mask, rejected_resp)
        delta_w = policy_w - ref_w
        delta_l = policy_l - ref_l
        loss = preference_loss(delta_w, delta_l, beta, loss_type)
        stats = {
            "loss": float(loss.detach().item()),
            "margin": float((delta_w - delta_l).detach().mean().item()),
            "delta_w": float(delta_w.detach().mean().item()),
            "delta_l": float(delta_l.detach().mean().item()),
            "acc": float((delta_w > delta_l).float().mean().item()),
        }
        return loss, stats

    def write_row(row):
        with open(metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        wandb.log(row)

    for epoch in range(num_epochs):
        model.train()
        reference_model.eval()
        progress = tqdm.tqdm(train_dataloader, desc=f"ipo epoch {epoch}")
        for batch in progress:
            loss, stats = batch_loss(batch)
            (loss / gradient_accumulation_steps).backward()
            micro += 1
            if micro % gradient_accumulation_steps != 0:
                continue
            grad_norm = None
            if gradient_clipping and gradient_clipping > 0:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clipping
                ))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            row = {
                "stage": loss_type,
                "epoch": epoch,
                "step": global_step,
                "lr": float(scheduler.get_last_lr()[0]),
                "grad_norm": grad_norm,
                "beta": beta,
                **stats,
            }
            write_row(row)
            progress.set_postfix(loss=stats["loss"], acc=stats["acc"])
            if max_steps and global_step >= max_steps:
                break
        if max_steps and global_step >= max_steps:
            eval_stats = evaluate_preference(
                model, reference_model, test_dataloader, batch_loss
            )
            write_row({
                "stage": loss_type,
                "epoch": epoch,
                "step": global_step,
                "split": "eval",
                **eval_stats,
            })
            break
        eval_stats = evaluate_preference(
            model, reference_model, test_dataloader, batch_loss
        )
        write_row({
            "stage": loss_type,
            "epoch": epoch,
            "step": global_step,
            "split": "eval",
            **eval_stats,
        })
        if save_model:
            epoch_dir = os.path.join(output_dir, f"epoch_{epoch}")
            save_checkpoint(model, tokenizer, optimizer, scheduler, epoch_dir)
    if save_model:
        save_checkpoint(model, tokenizer, optimizer, scheduler, output_dir)


def evaluate_preference(model, reference_model, dataloader, batch_loss, max_batches=32):
    """Mean preference loss and accuracy on up to max_batches test batches."""
    model.eval()
    reference_model.eval()
    totals = {}
    seen = 0
    with torch.no_grad():
        for batch in dataloader:
            _, stats = batch_loss(batch)
            for key, value in stats.items():
                totals[key] = totals.get(key, 0.0) + value
            seen += 1
            if seen >= max_batches:
                break
    model.train()
    if seen == 0:
        return {"eval_loss": None, "eval_batches": 0}
    averaged = {f"eval_{key}": value / seen for key, value in totals.items()}
    averaged["eval_batches"] = seen
    return averaged

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B')
    parser.add_argument('--dataset_name', type=str, default='asingh15/countdown_tasks_3to4-dpo')
    parser.add_argument('--output_dir', type=str, default='sft_model')
    parser.add_argument('--max_prompt_length', type=int, default=512)
    parser.add_argument('--max_response_length', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=5e-6)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--wandb_project', type=str, default='sft_default_project')
    parser.add_argument('--wandb_name', type=str, default='test')
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--gradient_checkpointing', type=int, default=1)
    parser.add_argument('--gradient_clipping', type=float, default=1.0)
    parser.add_argument('--max_steps', type=int, default=0)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--average_logps', type=int, default=0)
    parser.add_argument('--loss_type', type=str, default='dpo')
    args = parser.parse_args()

    wandb.init(project=args.wandb_project, name=args.wandb_name)
    wandb.config.update(vars(args))

    model, tokenizer, reference_model = get_model(args.model_name, args.device, use_gradient_checkpointing=args.gradient_checkpointing)

    dataloaders = get_dataloaders(
        dataset_name=args.dataset_name, 
        tokenizer=tokenizer, 
        max_prompt_length=args.max_prompt_length, 
        max_response_length=args.max_response_length, 
        batch_size=args.batch_size, 
        splits=['train', 'test'],
        pin_memory=True,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    train_dataloader, test_dataloader = dataloaders['train'], dataloaders['test']
    # Scheduler steps happen only after an optimizer step, so account for
    # gradient accumulation when estimating total training steps.
    num_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps
    if args.max_steps:
        num_steps = args.max_steps
    warmup_steps = int(num_steps * args.warmup_ratio)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps)

    full_output_dir = os.path.join(args.output_dir, args.wandb_project, args.wandb_name)
    os.makedirs(full_output_dir, exist_ok=True)

    train(
        model, 
        tokenizer, 
        reference_model,
        train_dataloader, 
        test_dataloader, 
        optimizer, 
        scheduler, 
        args.num_epochs, 
        args.device, 
        args.save_model, 
        full_output_dir, 
        args.gradient_accumulation_steps, 
        args.gradient_clipping,
        args.beta,
        args.average_logps,
        args.loss_type,
        args.max_steps,
    )

if __name__ == "__main__":
    main()
