"""High-level RLOO training orchestration.

This script alternates between:
1) sampling responses with vLLM (SamplingWorker), and
2) updating policy weights with PyTorch (RLOOUpdateWorker).
"""

import os
import warnings
import ray
import torch
from transformers import AutoTokenizer
import random
import sys
from pathlib import Path
warnings.filterwarnings("ignore")
import tempfile

# Make sibling packages (e.g., evaluation/) importable when this file is run as
# `python rloo_trainer/rloo.py`.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.countdown import compute_score
import numpy as np
from rloo_trainer.sampling_worker import SamplingWorker
from rloo_trainer.rloo_update_worker import RLOOUpdateWorker
from rloo_trainer.rloo_dataset import get_dataloaders
import wandb
from argparse import ArgumentParser
import uuid
import shutil
# os.environ['WANDB_MODE'] = 'offline'

class RLOOTrainer:
    """Coordinates online sampling, reward computation, and policy updates."""
    def __init__(
        self, 
        model_name='asingh15/qwen-sft-countdown-defaultproj',
        ref_model_name=None,
        tokenizer_name=None,
        dataset_name='asingh15/countdown_tasks_3to4',
        wandb_project='rloo_default_project',
        wandb_name='test',
        lr_schedule='constant',
        learning_rate=1e-5,
        warmup_ratio=0.05,
        weight_decay=0.01,
        batch_size=4,
        group_size=2, 
        entropy_coefficient=0.01, 
        kl_divergence_coefficient=0.0,
        num_epochs=10,
        gradient_accumulation_steps=1,
        gradient_clipping=1.0,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        max_tokens=1024,
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        max_num_batched_tokens=8192,
        enable_chunked_prefill=True,
        max_num_seqs=64,
        num_training_steps=250,
        max_table_rows=20,
        save_every_n_steps=-1,
        save_dir='checkpoints/rloo_checkpoints',
    ):
        self.model_name = model_name
        self.ref_model_name = self.model_name if ref_model_name is None else ref_model_name
        self.tokenizer_name = tokenizer_name if tokenizer_name is not None else model_name
        self.dataset_name = dataset_name
        self.wandb_project = wandb_project
        self.wandb_name = wandb_name
        self.lr_schedule = lr_schedule
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.num_training_steps = num_training_steps
        self.group_size = group_size
        self.entropy_coefficient = entropy_coefficient
        self.kl_divergence_coefficient = kl_divergence_coefficient
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clipping = gradient_clipping
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.max_tokens = max_tokens
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_batched_tokens = max_num_batched_tokens
        self.enable_chunked_prefill = enable_chunked_prefill
        self.max_num_seqs = max_num_seqs
        self.batch_size = batch_size
        self.max_prompt_length = 512
        self.max_response_length = max_tokens
        self.max_table_rows = max_table_rows
        self.save_every_n_steps = save_every_n_steps
        self.save_dir = save_dir
        
        # DataLoader yields prompts + ground-truth metadata only.
        dataloaders = get_dataloaders(
            self.dataset_name, 
            splits=['train', 'test'], 
            batch_size=self.batch_size, 
            num_proc=4,
        )
        self.train_dataloader, self.test_dataloader = dataloaders['train'], dataloaders['test']
        
        # Initialize actor references as None - will be created on demand
        self.sampling_worker = None
        self.update_worker = None
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        self.wandb = wandb.init(project=self.wandb_project, name=self.wandb_name)
        self.wandb.config.update(vars(self))

    def _create_sampling_worker(self, model_path):
        """Create a new sampling worker, killing any existing update worker first."""
        if self.update_worker is not None:
            ray.kill(self.update_worker)
            self.update_worker = None
        
        self.sampling_worker = SamplingWorker.remote(
            model_path=model_path,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_num_batched_tokens=self.max_num_batched_tokens,
            enable_chunked_prefill=self.enable_chunked_prefill,
            max_num_seqs=self.max_num_seqs,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            max_tokens=self.max_tokens,
            group_size=self.group_size
        )
        ray.get(self.sampling_worker.load_checkpoint.remote())
        return self.sampling_worker

    def _create_update_worker(self, model_path, optimizer_path, scheduler_path):
        """Create a new update worker, killing any existing sampling worker first."""
        if self.sampling_worker is not None:
            ray.kill(self.sampling_worker)
            self.sampling_worker = None
        
        self.update_worker = RLOOUpdateWorker.remote(
            model_path=model_path,
            ref_model_path=self.ref_model_name,
            optimizer_path=optimizer_path,
            scheduler_path=scheduler_path,
            batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            gradient_clipping=self.gradient_clipping,
            group_size=self.group_size,
            entropy_coefficient=self.entropy_coefficient,
            kl_divergence_coefficient=self.kl_divergence_coefficient,
            lr_schedule=self.lr_schedule,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            num_training_steps=self.num_training_steps,
        )
        ray.get(self.update_worker.load_checkpoint.remote())
        return self.update_worker

    def _build_generation_table(self, prompts, responses, rewards):
        """Create a lightweight W&B table of sampled generations."""
        if self.max_table_rows <= 0:
            return None

        flat_rows = []
        for prompt, prompt_responses, prompt_rewards in zip(prompts, responses, rewards):
            for response, reward in zip(prompt_responses, prompt_rewards):
                flat_rows.append((prompt, response, float(np.array(reward).item())))

        if not flat_rows:
            return None

        random.shuffle(flat_rows)
        flat_rows = flat_rows[: self.max_table_rows]

        table = wandb.Table(columns=["prompt", "response", "reward"])
        for prompt, response, reward in flat_rows:
            table.add_data(prompt, response, reward)
        return table

    def tokenize_batch(self, batch):
        """Tokenize prompt/response rollouts into arrays for policy update worker."""
        all_prompts = batch['prompt'] # batch
        all_responses = batch['response'] # batch x group_size
        all_rewards = batch['rewards'] # batch x group_size
        all_sample_log_probs = batch['sample_log_probs'] # batch x group_size

        # Flatten prompt-major grouped outputs into row-aligned training examples.
        all_prompts_repeated = [item for item in all_prompts for _ in range(self.group_size)]
        all_responses_flattened = [item for sublist in all_responses for item in sublist]
        all_rewards_flattened = [item for sublist in all_rewards for item in sublist]
        all_sample_log_probs_flattened = [item for sublist in all_sample_log_probs for item in sublist]
        assert (
            len(all_prompts_repeated)
            == len(all_responses_flattened)
            == len(all_rewards_flattened)
            == len(all_sample_log_probs_flattened)
        ), (
            f"len(all_prompts_repeated) = {len(all_prompts_repeated)}, "
            f"len(all_responses_flattened) = {len(all_responses_flattened)}, "
            f"len(all_rewards_flattened) = {len(all_rewards_flattened)}, "
            f"len(all_sample_log_probs_flattened) = {len(all_sample_log_probs_flattened)}"
        )

        # Left-pad prompts and right-pad responses before concatenation.
        self.tokenizer.padding_side = "left"
        tokenized_prompts = self.tokenizer(all_prompts_repeated, add_special_tokens=False, padding=True, truncation=True, max_length=self.max_prompt_length, return_tensors="np")
        self.tokenizer.padding_side = "right"
        tokenized_responses = self.tokenizer(all_responses_flattened, add_special_tokens=False, padding=True, truncation=True, max_length=self.max_response_length, return_tensors="np")

        prompt_input_ids, prompt_attention_mask = tokenized_prompts["input_ids"], tokenized_prompts["attention_mask"]
        response_input_ids, response_attention_mask = tokenized_responses["input_ids"], tokenized_responses["attention_mask"]
        is_response_token = np.concatenate([np.zeros_like(prompt_input_ids), np.ones_like(response_input_ids)], axis=1) # 0 for prompt tokens, 1 for response tokens
        input_ids = np.concatenate([prompt_input_ids, response_input_ids], axis=1)
        attention_mask = np.concatenate([prompt_attention_mask, response_attention_mask], axis=1)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "is_response_token": is_response_token,
            "rewards": np.array(all_rewards_flattened, dtype=np.float32),
            "sample_log_probs": np.array(all_sample_log_probs_flattened, dtype=np.float32),
        }

    def train(self):
        """Run online RLOO training for `num_training_steps` updates."""
        last_checkpoint_dir = None
        global_step = 0
        for epoch in range(self.num_epochs):
            if global_step > 0 and global_step == self.num_training_steps: break
            for train_iter, batch in enumerate(self.train_dataloader):
                if global_step > 0 and global_step == self.num_training_steps: break
                # 1) Sample `group_size` responses per prompt with current policy.
                ### SAMPLE ###
                print(f"Sampling from model, Epoch {epoch}, Global Step {global_step}")
                model_path = self.model_name if last_checkpoint_dir is None else os.path.join(last_checkpoint_dir, "model")
                
                self._create_sampling_worker(model_path)

                all_prompts = batch['prompt']
                all_ground_truth = batch['ground_truth']
                assert len(all_prompts) == len(all_ground_truth) == self.batch_size, f"len(all_prompts) = {len(all_prompts)}, len(all_ground_truth) = {len(all_ground_truth)}, self.batch_size = {self.batch_size}"
                all_responses, all_sample_log_probs = ray.get(self.sampling_worker.generate.remote(all_prompts))

                assert len(all_responses) == self.batch_size, f"len(all_responses) = {len(all_responses)}, self.batch_size = {self.batch_size}"
                assert len(all_sample_log_probs) == self.batch_size, f"len(all_sample_log_probs) = {len(all_sample_log_probs)}, self.batch_size = {self.batch_size}"
                for i in range(self.batch_size):
                    assert len(all_responses[i]) == self.group_size, f"len(all_responses[i]) = {len(all_responses[i])}, self.group_size = {self.group_size}"
                    assert len(all_sample_log_probs[i]) == self.group_size, f"len(all_sample_log_probs[i]) = {len(all_sample_log_probs[i])}, self.group_size = {self.group_size}"
                    for j in range(self.group_size):
                        assert isinstance(all_responses[i][j], str), f"all_responses[i][j] = {all_responses[i][j]}"
                        assert isinstance(all_sample_log_probs[i][j], (float, np.floating, int, np.integer)), (
                            f"all_sample_log_probs[i][j] = {all_sample_log_probs[i][j]}"
                        )

                print(f"Computing rewards, Epoch {epoch}, Global Step {global_step}")
                
                # 2) Score sampled responses against task ground truth.
                ### COMPUTE REWARDS ###
                all_rewards = []
                for curr_responses, curr_ground_truth in zip(all_responses, all_ground_truth):
                    curr_rewards = []
                    for x in curr_responses:
                        curr_rewards.append(compute_score(x, curr_ground_truth))
                    all_rewards.append(curr_rewards)
                reward_mean = np.mean(all_rewards).item()
                print('Reward Mean: ', reward_mean)

                generation_table = self._build_generation_table(all_prompts, all_responses, all_rewards)

                # 3) Convert sampled text/rewards into tokenized training arrays.
                ### TOKENIZE BATCH ###
                print(f"Tokenizing batch, Epoch {epoch}, Global Step {global_step}")

                batch_to_tokenize = {
                    'prompt': all_prompts,
                    'response': all_responses,
                    'rewards': all_rewards,
                    'sample_log_probs': all_sample_log_probs,
                }

                tokenized_batch = self.tokenize_batch(batch_to_tokenize)

                # 4) Spin up update worker with latest checkpoint state.
                ### LOAD MODEL FOR TRAINING ###
                print(f"Loading model for Training, Epoch {epoch}, Global Step {global_step}")
                model_path = self.model_name if last_checkpoint_dir is None else os.path.join(last_checkpoint_dir, "model")
                optimizer_path = None if last_checkpoint_dir is None else os.path.join(last_checkpoint_dir, "optimizer.pt")
                scheduler_path = None if last_checkpoint_dir is None else os.path.join(last_checkpoint_dir, "scheduler.pt")
                
                self._create_update_worker(model_path, optimizer_path, scheduler_path)

                # 5) Apply one policy update (with optional grad accumulation).
                ### UPDATE ###
                print(f"Updating model, Epoch {epoch}, Global Step {global_step}")
                all_metrics = ray.get(self.update_worker.update_gradient_accumulation.remote(
                    input_ids=tokenized_batch["input_ids"],
                    attention_mask=tokenized_batch["attention_mask"],
                    is_response_token=tokenized_batch["is_response_token"],
                    rewards=tokenized_batch["rewards"],
                    sample_log_probs=tokenized_batch["sample_log_probs"],
                ))

                print(f"Saving checkpoint, Epoch {epoch}, Global Step {global_step}")
                if self.save_every_n_steps > 0 and global_step % self.save_every_n_steps == 0:
                    # save checkpoint to save_dir to load for evaluation + sampling
                    save_dir = os.path.join(self.save_dir, self.wandb_project, self.wandb_name, f"epoch_{epoch}_step_{global_step}")
                else:
                    # save checkpoint to tmp_dir to load for sampling
                    save_dir = os.path.join(self.save_dir, self.wandb_project, self.wandb_name, f"latest_checkpoint")
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
                os.makedirs(save_dir, exist_ok=True)
                    
                save_model_path = os.path.join(save_dir, "model")
                save_optimizer_path = os.path.join(save_dir, "optimizer.pt")
                save_scheduler_path = os.path.join(save_dir, "scheduler.pt")
                ray.get(self.update_worker.update_checkpoint_paths.remote(
                    model_path=save_model_path,
                    optimizer_path=save_optimizer_path,
                    scheduler_path=save_scheduler_path,
                    load_checkpoint=False
                ))
                ray.get(self.update_worker.save_checkpoint.remote())
                last_checkpoint_dir = save_dir

                print("-" * 100)
                print(f"Epoch {epoch}, Global Step {global_step}")
                scientific_metric_names = {
                    "lr",
                    "kl_loss",
                    "weight_mse",
                    "weight_max_abs_diff",
                    "weight_nonzero_diff_ratio",
                    "weight_changed_tensor_ratio",
                }
                for metric_name, metric_value in all_metrics.items():
                    if isinstance(metric_value, (float, np.floating)):
                        metric_value_float = float(metric_value)
                        if (
                            metric_name in scientific_metric_names
                            or (0 < abs(metric_value_float) < 1e-4)
                        ):
                            print(f"{metric_name}: {metric_value_float:.6e}")
                        else:
                            print(f"{metric_name}: {metric_value_float:.4f}")
                    else:
                        print(f"{metric_name}: {metric_value}")
                print("-" * 100)

                metrics_logged = {'train/'+ k: v for k, v in all_metrics.items()}

                log_dict = {
                    "train/epoch": epoch,
                    "train/train_iter": train_iter,
                    "train/global_step": global_step,
                    "sampling/reward_mean": reward_mean,
                    **metrics_logged,
                }
                if generation_table is not None:
                    log_dict["samples/generations"] = generation_table

                self.wandb.log(log_dict, step=global_step)
                global_step += 1
        
        if self.sampling_worker is not None:
            ray.kill(self.sampling_worker)
            self.sampling_worker = None
        if self.update_worker is not None:
            ray.kill(self.update_worker)
            self.update_worker = None
        
        ray.shutdown()
        self.wandb.finish()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='asingh15/qwen-sft-countdown-defaultproj')
    parser.add_argument('--ref_model_name', type=str, default=None)
    parser.add_argument('--tokenizer_name', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='asingh15/countdown_tasks_3to4')
    parser.add_argument('--wandb_project', type=str, default='rloo_default_project')
    parser.add_argument('--wandb_name', type=str, default='test')
    parser.add_argument('--lr_schedule', type=str, default='constant')
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--group_size', type=int, default=2)
    parser.add_argument('--entropy_coefficient', type=float, default=0.01)
    parser.add_argument('--kl_divergence_coefficient', type=float, default=0.0)
    parser.add_argument('--num_training_steps', type=int, default=250)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--gradient_clipping', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=-1)
    parser.add_argument('--min_p', type=float, default=0.0)
    parser.add_argument('--max_tokens', type=int, default=1024)
    parser.add_argument('--max_model_len', type=int, default=2048)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9)
    parser.add_argument('--max_num_batched_tokens', type=int, default=8192)
    parser.add_argument('--enable_chunked_prefill', action='store_true')
    parser.add_argument('--disable_chunked_prefill', action='store_true')
    parser.add_argument('--max_num_seqs', type=int, default=64)
    parser.add_argument('--max_table_rows', type=int, default=20)
    parser.add_argument('--save_every_n_steps', type=int, default=-1) # -1 means don't save every n steps
    parser.add_argument('--save_dir', type=str, default='checkpoints/rloo_checkpoints')
    args = parser.parse_args()
    if args.enable_chunked_prefill and args.disable_chunked_prefill:
        raise ValueError("Cannot set both --enable_chunked_prefill and --disable_chunked_prefill.")
    if args.enable_chunked_prefill:
        args.enable_chunked_prefill = True
    elif args.disable_chunked_prefill:
        args.enable_chunked_prefill = False
    else:
        args.enable_chunked_prefill = True
    del args.disable_chunked_prefill

    ray.init()
    
    trainer = RLOOTrainer(
        **vars(args)
    )
    trainer.train()
