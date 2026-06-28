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
    return "Hello from P01 scaffold — ready to research!"

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


# P1-specific helpers (synthetic data for controller training demo)
def make_synthetic_batch(batch_size: int = 4, hidden_size: int = 2048, seed: int = 42):
    """Create fake hidden states + regression targets (value) + binary trust targets.
    In real work: replace with actual last-hidden from frozen LM + labels from verifier correctness.
    """
    import torch
    torch.manual_seed(seed)
    hidden = torch.randn(batch_size, hidden_size)
    # Simulate: some queries 'easy' (high value from extra compute? use proxy)
    true_value = torch.randn(batch_size) * 0.5 + 0.3   # centered around moderate value
    # Trust higher when 'value' higher in this fake (correlated in real via analysis)
    true_trust = torch.sigmoid(true_value * 1.5 + torch.randn(batch_size) * 0.3)
    return {
        "hidden": hidden,
        "target": true_value,           # used as value regression target
        "target_trust": true_trust
    }


def get_dummy_loader(num_batches: int = 5, **kwargs):
    class DummyLoader(list):
        pass
    batches = []
    for i in range(num_batches):
        batches.append(make_synthetic_batch(**kwargs))
    return DummyLoader(batches)
