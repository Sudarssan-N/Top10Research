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
    return "Hello from P07 scaffold — ready to research!"

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


# P7 synthetic for bias probe (pair embs + bias label)
def make_synthetic_batch(batch_size: int = 4, hidden_size: int = 2048, seed: int = 42):
    import torch
    torch.manual_seed(seed)
    # For pair models, we store two "hiddens"
    emb_a = torch.randn(batch_size, hidden_size)
    emb_b = torch.randn(batch_size, hidden_size)
    target = (torch.rand(batch_size) > 0.5).float()
    return {"hidden_a": emb_a, "hidden_b": emb_b, "target": target}


def get_dummy_loader(num_batches: int = 5, **kwargs):
    class DummyLoader(list):
        pass
    batches = [make_synthetic_batch(**kwargs) for _ in range(num_batches)]
    return DummyLoader(batches)
