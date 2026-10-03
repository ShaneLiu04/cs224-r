"""Dataset + collation utilities for IPO/DPO-style preference training."""

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import os
from transformers import AutoTokenizer
import torch

def get_map_fn(tokenizer, prompt_key, response_w_key, response_l_key):
    """Build a mapping function that applies chat formatting to all fields."""
    def map_dataset(examples):
        prompt, response_w, response_l = examples[prompt_key], examples[response_w_key], examples[response_l_key]
        tok_string_input = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        tok_string_output_w = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response_w}
            ],
            add_generation_prompt=False,
            tokenize=False,
        )
        tok_string_output_l = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response_l}
            ],
            add_generation_prompt=False,
            tokenize=False,
        )

        # We slice using prompt length below, so prompt must be a strict prefix.
        assert tok_string_output_w.startswith(tok_string_input), (
            f"input: {tok_string_input} is not a prefix of output: {tok_string_output_w}"
        )
        assert tok_string_output_l.startswith(tok_string_input), (
            f"input: {tok_string_input} is not a prefix of output: {tok_string_output_l}"
        )
        examples[prompt_key] = tok_string_input
        examples[response_w_key] = tok_string_output_w[len(tok_string_input):]
        examples[response_l_key] = tok_string_output_l[len(tok_string_input):]
        return examples
    return map_dataset

class IPODataset(Dataset):
    """Loads pairwise preference examples: (prompt, chosen, rejected)."""
    def __init__(
        self, 
        dataset_name, 
        tokenizer, 
        prompt_key='query', 
        response_w_key='response_ws',
        response_l_key='response_ls',
        max_prompt_length=512,
        max_response_length=1024,
        padding=True,
        truncation=True,
        num_proc=os.cpu_count(),
        split='train'
    ):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.prompt_key = prompt_key
        self.response_w_key = response_w_key
        self.response_l_key = response_l_key
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.padding = padding
        self.truncation = truncation
        self.num_proc = num_proc
        self.split = split

        self.dataset = load_dataset(dataset_name, num_proc=self.num_proc, split=split)
        curr_map_fn = get_map_fn(self.tokenizer, self.prompt_key, self.response_w_key, self.response_l_key)
        self.dataset = self.dataset.map(curr_map_fn, num_proc=self.num_proc, desc="Applying chat template")

        self.all_prompts = self.dataset[self.prompt_key]
        self.all_responses_w = self.dataset[self.response_w_key]
        self.all_responses_l = self.dataset[self.response_l_key]

    def __len__(self):
        return len(self.all_prompts)

    def __getitem__(self, idx):
        prompt = self.all_prompts[idx]
        response_w = self.all_responses_w[idx]
        response_l = self.all_responses_l[idx]
        return {'prompt': prompt, 'response_w': response_w, 'response_l': response_l}

    def collate_fn(self, batch):
        prompts = [item['prompt'] for item in batch]
        responses_w = [item['response_w'] for item in batch]
        responses_l = [item['response_l'] for item in batch]
        # Prompt left padding keeps final prompt position aligned across examples.
        prompt_toks = self.tokenizer(prompts, add_special_tokens=False, padding=self.padding, truncation=self.truncation, max_length=self.max_prompt_length, padding_side="left", return_tensors="pt")
        # Chosen/rejected continuations are right-padded for autoregressive loss.
        response_toks_w = self.tokenizer(responses_w, add_special_tokens=False, padding=self.padding, truncation=self.truncation, max_length=self.max_response_length, padding_side="right", return_tensors="pt")
        response_toks_l = self.tokenizer(responses_l, add_special_tokens=False, padding=self.padding, truncation=self.truncation, max_length=self.max_response_length, padding_side="right", return_tensors="pt")

        prompt_input_ids, prompt_attention_mask = prompt_toks["input_ids"], prompt_toks["attention_mask"]
        response_input_ids_w, response_attention_mask_w = response_toks_w["input_ids"], response_toks_w["attention_mask"]
        response_input_ids_l, response_attention_mask_l = response_toks_l["input_ids"], response_toks_l["attention_mask"]

        input_ids_w = torch.cat([prompt_input_ids, response_input_ids_w], dim=1)
        attention_mask_w = torch.cat([prompt_attention_mask, response_attention_mask_w], dim=1)
        # Response masks mark which token positions belong to continuations.
        is_response_token_w = torch.cat([torch.zeros_like(prompt_input_ids), torch.ones_like(response_input_ids_w)], dim=1)
        input_ids_l = torch.cat([prompt_input_ids, response_input_ids_l], dim=1)
        attention_mask_l = torch.cat([prompt_attention_mask, response_attention_mask_l], dim=1)
        is_response_token_l = torch.cat([torch.zeros_like(prompt_input_ids), torch.ones_like(response_input_ids_l)], dim=1)

        return {
            "input_ids_w": input_ids_w,
            "attention_mask_w": attention_mask_w,
            "is_response_token_w": is_response_token_w,
            "input_ids_l": input_ids_l,
            "attention_mask_l": attention_mask_l,
            "is_response_token_l": is_response_token_l,
        }

def get_dataloaders(
    dataset_name, 
    tokenizer, 
    max_prompt_length=512, 
    max_response_length=1024, 
    padding=True, 
    truncation=True, 
    num_proc=os.cpu_count(), 
    batch_size=16, 
    splits=['train', 'test'],
    num_workers=4,
    pin_memory=True,
    drop_last=True,
    prompt_key='query',
    response_w_key='response_ws',
    response_l_key='response_ls',
    shuffle=True,
    gradient_accumulation_steps=1
):
    """Create split->DataLoader dict with per-microbatch sizing.

    `batch_size` here represents effective batch size. We divide by
    gradient accumulation steps to obtain DataLoader microbatch size.
    """
    assert batch_size % gradient_accumulation_steps == 0, "batch_size must be divisible by gradient_accumulation_steps"
    bs_for_dataloader = batch_size // gradient_accumulation_steps
    dataloaders = {}
    for split in splits:
        dataset = IPODataset(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            prompt_key=prompt_key,
            response_w_key=response_w_key,
            response_l_key=response_l_key,
            max_prompt_length=max_prompt_length,
            max_response_length=max_response_length,
            padding=padding,
            truncation=truncation,
            num_proc=num_proc,
            split=split
        )
        dataloaders[split] = DataLoader(
            dataset, 
            batch_size=bs_for_dataloader, 
            shuffle=shuffle, 
            collate_fn=dataset.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last
        )
    return dataloaders

    
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    dataloaders = get_dataloaders(dataset_name="asingh15/countdown_tasks_3to4-dpo", tokenizer=tokenizer, splits=['train'], batch_size=2)

    for batch in iter(dataloaders['train']):
        input_ids_w = batch["input_ids_w"]
        attention_mask_w = batch["attention_mask_w"]
        is_response_token_w = batch["is_response_token_w"]
        input_ids_l = batch["input_ids_l"]
        attention_mask_l = batch["attention_mask_l"]
        is_response_token_l = batch["is_response_token_l"]
        break

    print(input_ids_w.shape, attention_mask_w.shape, is_response_token_w.shape)
    print(input_ids_l.shape, attention_mask_l.shape, is_response_token_l.shape)
    print(tokenizer.decode(input_ids_w[0]))
    print(tokenizer.decode(input_ids_l[0]))
