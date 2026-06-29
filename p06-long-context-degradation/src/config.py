import os
import yaml
import torch
from dataclasses import dataclass, asdict
from typing import List, Optional

@dataclass
class ExperimentConfig:
    # Long-context model with a real long window (override per experiment).
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    seed: int = 42
    device: str = "auto"  # auto, cpu, cuda

    # Controlled length-vs-position design (the core contribution):
    # orthogonalize {token count} x {absolute position of the evidence}.
    context_lengths: List[int] = None   # total context sizes to sweep (tokens)
    evidence_positions: List[float] = None  # relative placement of the gold evidence (0=start,1=end)
    num_distractors: int = 0            # filler items; retrieval kept PERFECT (gold always present)
    eval_task: str = "niah_qa"          # certified-perfect-retrieval QA probe

    # Training-free mitigations to test
    mitigation: str = "none"            # none | reorder | attn_calibrate | recitation
    max_new_tokens: int = 64

    output_dir: str = "results/default"
    log_interval: int = 10

    def __post_init__(self):
        if self.context_lengths is None:
            self.context_lengths = [1000, 4000, 16000, 64000]
        if self.evidence_positions is None:
            self.evidence_positions = [0.0, 0.25, 0.5, 0.75, 1.0]

    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        # Coerce common numeric fields (yaml sometimes loads 1e-4 etc as str)
        numeric_keys = {"max_new_tokens", "num_distractors", "log_interval", "seed"}
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
