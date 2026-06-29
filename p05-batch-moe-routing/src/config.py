import os
import yaml
import torch
from dataclasses import dataclass, asdict
from typing import List, Optional

@dataclass
class ExperimentConfig:
    # Open MoE target. Mixtral (~90GB fp16) / Qwen3-30B (~60GB) need quantization or
    # CPU-offload on a single 40GB A100 — see research.md "Critical assessment".
    base_model: str = "Qwen/Qwen1.5-MoE-A2.7B"
    seed: int = 42
    device: str = "auto"  # auto, cpu, cuda

    # MoE / routing (override num_experts/top_k per model)
    num_experts: int = 60          # Qwen1.5-MoE-A2.7B = 60 experts
    top_k_experts: int = 4         # model's native per-token top-k
    routing_strategy: str = "baseline"  # baseline | opportunistic | similarity | budget
    expert_budget: int = 0         # batch-wide cap on unique active experts (0 = uncapped)

    # Inference workload (decode is memory-bound on |union of experts| across the batch)
    max_new_tokens: int = 256
    batch_sizes: List[int] = None  # swept for union-of-experts profiling
    serving_baseline: str = "vllm" # wall-clock comparison target (the load-bearing metric)

    # Only if a lightweight router head is calibrated post-hoc (NOT a trained router)
    learning_rate: float = 1e-4
    num_epochs: int = 2

    output_dir: str = "results/default"
    log_interval: int = 10

    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 4, 16, 64]

    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        # Coerce common numeric fields (yaml sometimes loads 1e-4 etc as str)
        numeric_keys = {"learning_rate", "num_epochs", "log_interval", "seed",
                        "num_experts", "top_k_experts", "expert_budget", "max_new_tokens"}
        for k in list(data.keys()):
            if k in numeric_keys and isinstance(data[k], str):
                try:
                    if "." in data[k] or "e" in data[k].lower():
                        data[k] = float(data[k])
                    else:
                        data[k] = int(data[k])
                except ValueError:
                    pass
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


    def to_yaml(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, sort_keys=False)

    def get_device(self):
        # Never hand back "cuda" when it isn't available (guards CPU smoke runs).
        if self.device in ("auto", "cuda"):
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device
