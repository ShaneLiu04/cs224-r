"""Dataset + collation utilities for SFT stage.

The main design choice here is to keep prompt and response tokenization
separate, then concatenate them and return a response-token mask so training
can compute loss only on assistant tokens.
"""

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import os
from transformers import AutoTokenizer
import torch

def get_map_fn(tokenizer, prompt_key, response_key):
    """Build a mapping function that applies chat template formatting."""
    def map_dataset(examples):
        prompt, response = examples[prompt_key], examples[response_key]
        tok_string_input = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        tok_string_output = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ],
            add_generation_prompt=False,
            tokenize=False,
        )

        # The assistant-formatted sequence should contain the prompt-only prefix
        # so we can safely slice out the response portion below.
        assert tok_string_input in tok_string_output, f"input: {tok_string_input} not in output: {tok_string_output}"
        examples[prompt_key] = tok_string_input
        examples[response_key] = tok_string_output[len(tok_string_input):]
        return examples
    return map_dataset

class SFTDataset(Dataset):
    """Loads an SFT dataset and exposes prompt/response string pairs."""
    def __init__(
        self, 
        dataset_name, 
        tokenizer, 
        prompt_key='query', 
        response_key='completion',
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
        self.response_key = response_key
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.padding = padding
        self.truncation = truncation
        self.num_proc = num_proc
        self.split = split

        self.dataset = load_dataset(dataset_name, num_proc=self.num_proc, split=split)
        curr_map_fn = get_map_fn(self.tokenizer, self.prompt_key, self.response_key)
        self.dataset = self.dataset.map(curr_map_fn, num_proc=self.num_proc, desc="Applying chat template")

        self.all_prompts = self.dataset[self.prompt_key]
        self.all_responses = self.dataset[self.response_key]

    def __len__(self):
        return len(self.all_prompts)

    def __getitem__(self, idx):
        prompt = self.all_prompts[idx]
        response = self.all_responses[idx]
        return {'prompt': prompt, 'response': response}

    def collate_fn(self, batch):
        # Tokenize prompts with left padding so right edge stays aligned with
        # the final prompt token across the batch.
        prompts = [item['prompt'] for item in batch]
        responses = [item['response'] for item in batch]
        prompt_toks = self.tokenizer(prompts, add_special_tokens=False, padding=self.padding, truncation=self.truncation, max_length=self.max_prompt_length, padding_side="left", return_tensors="pt")
        # Tokenize responses with right padding since they are predicted left->right.
        response_toks = self.tokenizer(responses, add_special_tokens=False, padding=self.padding, truncation=self.truncation, max_length=self.max_response_length, padding_side="right", return_tensors="pt")

        prompt_input_ids, prompt_attention_mask = prompt_toks["input_ids"].squeeze(), prompt_toks["attention_mask"].squeeze()
        response_input_ids, response_attention_mask = response_toks["input_ids"].squeeze(), response_toks["attention_mask"].squeeze()

        input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1)
        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
        # 0 marks prompt tokens (ignored for loss), 1 marks response tokens
        # (included for loss/metrics).
        is_response_token = torch.cat([torch.zeros_like(prompt_input_ids), torch.ones_like(response_input_ids)], dim=1)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "is_response_token": is_response_token
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
    response_key='completion',
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
        dataset = SFTDataset(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            prompt_key=prompt_key,
            response_key=response_key,
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

    dataloaders = get_dataloaders(dataset_name="Asap7772/cog_behav_all_strategies", tokenizer=tokenizer, splits=['train', 'test'])

    for batch in iter(dataloaders['train']):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        is_response_token = batch["is_response_token"]
        break

    print(input_ids.shape, attention_mask.shape, is_response_token.shape)
    print(tokenizer.decode(input_ids[0]))
