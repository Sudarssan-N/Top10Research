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
    return "Hello from P03 scaffold — ready to research!"

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


# P3-specific synthetic data for overthinking controller
def make_synthetic_batch(batch_size: int = 4, hidden_size: int = 2048, seed: int = 42):
    """Hidden + binary target: 1 = 'converged / safe to stop', 0 = 'needs more reasoning'.
    In real: derived from hidden states during long chains or from final answer quality vs length.
    """
    import torch
    torch.manual_seed(seed)
    hidden = torch.randn(batch_size, hidden_size)
    # Fake signal: some "easy" examples converge early (higher chance of stop target)
    difficulty = torch.randn(batch_size)
    stop_target = (torch.sigmoid(difficulty * -0.8) > torch.rand(batch_size)).float()
    return {
        "hidden": hidden,
        "target": stop_target,
    }


def get_dummy_loader(num_batches: int = 5, **kwargs):
    class DummyLoader(list):
        pass
    batches = [make_synthetic_batch(**kwargs) for _ in range(num_batches)]
    return DummyLoader(batches)
