import json
import os
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ExperimentConfig
from .utils import set_seed, AverageMeter


def _forward_loss(model, batch, device, config: ExperimentConfig, difficulty_only: bool = False):
    value_crit = torch.nn.MSELoss()
    trust_crit = torch.nn.BCELoss()

    hidden = batch["hidden"].to(device)
    target_val = batch["target"].to(device).float()
    target_trust = batch["target_trust"].to(device).float()

    out = model(hidden)
    loss_val = value_crit(out["value"].view(-1), target_val.view(-1))

    if difficulty_only or getattr(model, "variant", "") == "difficulty_only":
        return loss_val, {
            "value_loss": float(loss_val.item()),
            "total_loss": float(loss_val.item()),
        }

    loss_trust = trust_crit(out["verifier_trust"].view(-1), target_trust.view(-1))
    total = loss_val + config.trust_loss_weight * loss_trust
    return total, {
        "value_loss": float(loss_val.item()),
        "trust_loss": float(loss_trust.item()),
        "total_loss": float(total.item()),
    }


@torch.inference_mode()
def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: str,
    config: ExperimentConfig,
    difficulty_only: bool = False,
) -> dict:
    model.eval().to(device)
    total_loss = 0.0
    value_loss = 0.0
    trust_loss = 0.0
    n = 0
    for batch in val_loader:
        loss, parts = _forward_loss(model, batch, device, config, difficulty_only=difficulty_only)
        bs = batch["hidden"].size(0)
        total_loss += parts["total_loss"] * bs
        value_loss += parts["value_loss"] * bs
        if "trust_loss" in parts:
            trust_loss += parts["trust_loss"] * bs
        n += bs
    denom = max(n, 1)
    out = {
        "val_loss": total_loss / denom,
        "val_value_loss": value_loss / denom,
        "val_samples": n,
    }
    if not difficulty_only:
        out["val_trust_loss"] = trust_loss / denom
    return out


def train_with_validation(
    config: ExperimentConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: torch.nn.Module,
    device: str,
    variant: str = "verifier_aware",
    run_dir: Optional[str] = None,
    feature_meta: Optional[dict] = None,
) -> Tuple[torch.nn.Module, dict]:
    """Phase 3 training: per-epoch validation + best checkpoint."""
    from .checkpoint import save_controller_checkpoint

    set_seed(config.seed)
    model = model.to(device)
    difficulty_only = variant == "difficulty_only"
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    run_dir = run_dir or config.output_dir
    ckpt_dir = os.path.join(run_dir, "checkpoints", variant)
    os.makedirs(ckpt_dir, exist_ok=True)

    history = []
    best_val = float("inf")
    best_path = os.path.join(ckpt_dir, "best.pt")

    for epoch in range(config.num_epochs):
        model.train()
        meter = AverageMeter()
        pbar = tqdm(train_loader, desc=f"{variant} epoch {epoch+1}/{config.num_epochs}")
        for step, batch in enumerate(pbar):
            loss, _ = _forward_loss(model, batch, device, config, difficulty_only=difficulty_only)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            meter.update(loss.item())
            if step % config.log_interval == 0:
                pbar.set_postfix(loss=f"{meter.avg:.4f}")

        val_metrics = validate(model, val_loader, device, config, difficulty_only=difficulty_only)
        row = {
            "epoch": epoch + 1,
            "train_loss": meter.avg,
            **val_metrics,
        }
        history.append(row)
        print(f"  {variant} epoch {epoch+1}: train={meter.avg:.4f} val={val_metrics['val_loss']:.4f}")

        epoch_path = os.path.join(ckpt_dir, f"epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), epoch_path)

        if val_metrics["val_loss"] < best_val:
            best_val = val_metrics["val_loss"]
            torch.save(model.state_dict(), best_path)

    meta = dict(feature_meta or {})
    meta.update({"variant": variant, "best_val_loss": best_val})
    save_controller_checkpoint(
        os.path.join(ckpt_dir, "best_controller.pt"),
        model,
        meta,
        variant,
        config,
        metrics={"history": history, "best_val_loss": best_val},
    )

    hist_path = os.path.join(run_dir, f"train_history_{variant}.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    return model, {"history": history, "best_val_loss": best_val, "checkpoint": best_path}


def train(config: ExperimentConfig, train_loader: DataLoader, model: torch.nn.Module, device: str):
    """Simple train loop (Phase 2 compat) — no validation."""
    set_seed(config.seed)
    model = model.to(device)
    difficulty_only = getattr(model, "variant", "") == "difficulty_only"
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    model.train()
    for epoch in range(config.num_epochs):
        meter = AverageMeter()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            loss, _ = _forward_loss(model, batch, device, config, difficulty_only=difficulty_only)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            meter.update(loss.item())
            if pbar.n % config.log_interval == 0:
                pbar.set_postfix(loss=f"{meter.avg:.4f}")
        print(f"Epoch {epoch+1} avg loss: {meter.avg:.4f}")

        ckpt_dir = os.path.join(config.output_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"epoch_{epoch}.pt"))

    return model