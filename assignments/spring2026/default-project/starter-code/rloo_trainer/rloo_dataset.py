"""Minimal dataset wrapper for RLOO sampling.

Unlike SFT/IPO, this loader returns raw prompt strings and structured
ground-truth metadata because responses are generated online.
"""

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import os
from transformers import AutoTokenizer
import torch

class RLOODataset(Dataset):
    """Loads prompts plus per-example reward metadata."""
    def __init__(self, dataset_name, split='train', batch_size=16, num_proc=os.cpu_count()):
        self.dataset_name = dataset_name
        self.split = split
        self.batch_size = batch_size
        self.num_proc = num_proc

        self.dataset = load_dataset(dataset_name, split=split, num_proc=num_proc)
        self.all_prompts = self.dataset['prompt']
        self.all_ground_truth = self.dataset['ground_truth']

    def __len__(self):
        return len(self.all_prompts)

    def __getitem__(self, idx):
        prompt = self.all_prompts[idx]
        ground_truth = self.all_ground_truth[idx]
        return {'prompt': prompt, 'ground_truth': ground_truth}

    def collate_fn(self, batch):
        # Keep items as Python objects; tokenization happens after sampling.
        prompts = [item['prompt'] for item in batch]
        ground_truths = [item['ground_truth'] for item in batch]
        return {'prompt': prompts, 'ground_truth': ground_truths}


def get_dataloaders(dataset_name, splits=['train', 'test'], batch_size=16, num_proc=4):
    """Create split->DataLoader dict for online RL sampling."""
    dataloaders = {}
    for split in splits:
        dataset = RLOODataset(
            dataset_name=dataset_name,
            split=split,
            batch_size=batch_size,
            num_proc=num_proc
        )
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=dataset.collate_fn,
            num_workers=num_proc,
            pin_memory=True,
            drop_last=True
        )
    return dataloaders
