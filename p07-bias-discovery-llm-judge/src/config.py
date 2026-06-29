import os
import yaml
import torch
from dataclasses import dataclass, asdict
from typing import List, Optional

@dataclass
class ExperimentConfig:
    # The JUDGE under audit (API or local). Candidate answers come from the testbed.
    judge_model: str = "gpt-4o-mini"
    seed: int = 42
    device: str = "auto"  # auto, cpu, cuda

    # Bias-discovery setup (causal, code-judge domain)
    testbed: str = "codejudgebench"     # execution-verifiable: unit tests fix "quality"
    candidate_bias_factors: List[str] = None  # factors to perturb while holding quality fixed
    perturbation_model: str = "gpt-4o"  # generates quality-preserving counterfactuals
    n_bootstrap: int = 1000             # bootstrap CIs for the flip-rate ATE
    n_perturbations_per_pair: int = 8

    # Judge call settings
    max_new_tokens: int = 512
    temperature: float = 0.0            # judges run greedy for reproducibility

    output_dir: str = "results/default"
    log_interval: int = 10

    def __post_init__(self):
        if self.candidate_bias_factors is None:
            # seeds for discovery, not an exhaustive list — the point is to find NEW ones
            self.candidate_bias_factors = [
                "position", "verbosity", "self_preference", "authority", "formatting",
            ]

    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        # Coerce common numeric fields (yaml sometimes loads 1e-4 etc as str)
        numeric_keys = {"temperature", "max_new_tokens", "n_bootstrap",
                        "n_perturbations_per_pair", "log_interval", "seed"}
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
