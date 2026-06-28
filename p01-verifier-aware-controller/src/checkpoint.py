"""Checkpoint load/save for controller models."""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import torch

from .config import ExperimentConfig
from .core import DifficultyOnlyController, VerifierAwareController, build_controller


def save_controller_checkpoint(
    path: str,
    model: torch.nn.Module,
    meta: Dict,
    variant: str,
    config: ExperimentConfig,
    metrics: Optional[Dict] = None,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "variant": variant,
            "meta": meta,
            "hidden_size": meta.get("hidden_size"),
            "metrics": metrics or {},
            "config": {
                "base_model": config.base_model,
                "controller_hidden_dim": config.controller_hidden_dim,
                "num_controller_layers": config.num_controller_layers,
                "n_samples_best_of_n": config.n_samples_best_of_n,
            },
        },
        path,
    )


def load_controller_checkpoint(
    path: str,
    device: str = "cpu",
) -> Tuple[torch.nn.Module, Dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    variant = payload.get("variant", "verifier_aware")
    hidden_size = int(payload.get("hidden_size") or payload.get("meta", {}).get("hidden_size", 2048))
    cfg = payload.get("config", {})
    model = build_controller(
        variant=variant,
        hidden_size=hidden_size,
        controller_dim=int(cfg.get("controller_hidden_dim", 128)),
        num_layers=int(cfg.get("num_controller_layers", 2)),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload