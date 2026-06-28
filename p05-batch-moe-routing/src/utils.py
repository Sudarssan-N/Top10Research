import os
import random
import numpy as np
import torch
from typing import Dict, Any

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def hello():
    return "Hello from P05 scaffold — ready to research!"

def save_jsonl(path, records):
    import json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

def load_jsonl(path):
    import json
    with open(path) as f:
        return [json.loads(line) for line in f]

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# P5-specific synthetic for batch MoE router (hidden summary + ideal expert distribution)
def make_synthetic_batch(batch_size: int = 4, hidden_size: int = 2048, num_experts: int = 8, seed: int = 42):
    import torch
    torch.manual_seed(seed)
    hidden = torch.randn(batch_size, hidden_size)
    # Simulate preferred experts per item in batch (one-hot-ish)
    target_dist = torch.zeros(batch_size, num_experts)
    for i in range(batch_size):
        pref = torch.randint(0, num_experts, (2,))
        target_dist[i, pref] = 1.0 / len(pref)
    return {"hidden": hidden, "target_dist": target_dist}


def get_dummy_loader(num_batches: int = 5, **kwargs):
    class DummyLoader(list):
        pass
    batches = [make_synthetic_batch(**kwargs) for _ in range(num_batches)]
    return DummyLoader(batches)
