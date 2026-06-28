import os
import yaml
import torch
from dataclasses import dataclass, asdict
from typing import List, Optional

@dataclass
class ExperimentConfig:
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    seed: int = 42
    device: str = "auto"  # auto, cpu, cuda
    max_new_tokens: int = 512
    temperature: float = 0.7
    num_beams: int = 1

    # Problem-specific
    controller_hidden_dim: int = 128
    num_controller_layers: int = 2
    train_batch_size: int = 8
    eval_batch_size: int = 4
    learning_rate: float = 1e-4
    num_epochs: int = 2
    budget_max_tokens: int = 2048

    # Eval
    benchmarks: List[str] = None
    n_samples_best_of_n: int = 16

    # P2-specific
    target_fpr: float = 0.05
    fp_loss_weight: float = 2.0
    loss_type: str = "asymmetric_bce"  # asymmetric_bce | focal | standard_bce
    data_dir: str = "data"
    val_fraction: float = 0.15

    output_dir: str = "results/default"
    log_interval: int = 10

    def __post_init__(self):
        if self.benchmarks is None:
            self.benchmarks = ["math500", "gsm8k"]

    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        # Coerce common numeric fields (yaml sometimes loads 1e-4 etc as str)
        numeric_keys = {"learning_rate", "temperature", "max_new_tokens", "num_beams",
                        "controller_hidden_dim", "num_controller_layers",
                        "train_batch_size", "eval_batch_size", "num_epochs",
                        "budget_max_tokens", "n_samples_best_of_n", "log_interval", "seed",
                        "target_fpr", "fp_loss_weight", "val_fraction"}
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
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device
