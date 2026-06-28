import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .config import ExperimentConfig
from .utils import set_seed, AverageMeter
from .core import FPRBoundedVerifier  # P2 specific
import os

def train(config: ExperimentConfig, train_loader: DataLoader, model: torch.nn.Module, device: str):
    set_seed(config.seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    base_crit = torch.nn.BCEWithLogitsLoss(reduction='none')
    fp_weight = getattr(config, "fp_loss_weight", 2.0)
    loss_type = getattr(config, "loss_type", "asymmetric_bce")

    model.train()
    for epoch in range(config.num_epochs):
        meter = AverageMeter()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for step, batch in enumerate(pbar):
            hidden = batch["hidden"].to(device)
            target = batch["target"].to(device).float()  # 1 = correct/good

            out = model(hidden)
            logit = out["logit"]
            logit_flat = logit.squeeze()
            per_sample_loss = base_crit(logit_flat, target)

            if loss_type == "standard_bce":
                loss = per_sample_loss.mean()
            elif loss_type == "focal":
                pred_prob = torch.sigmoid(logit_flat)
                pt = torch.where(target > 0.5, pred_prob, 1 - pred_prob)
                focal = (1 - pt) ** 2 * per_sample_loss
                is_fp = (pred_prob > 0.5) & (target < 0.5)
                weights = torch.where(is_fp, torch.tensor(fp_weight, device=device), torch.tensor(1.0, device=device))
                loss = (focal * weights).mean()
            else:
                pred_prob = torch.sigmoid(logit_flat)
                is_fp = (pred_prob > 0.5) & (target < 0.5)
                weights = torch.where(is_fp, torch.tensor(fp_weight, device=device), torch.tensor(1.0, device=device))
                loss = (per_sample_loss * weights).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            meter.update(loss.item())
            if step % config.log_interval == 0:
                pbar.set_postfix(loss=f"{meter.avg:.4f}")

        print(f"Epoch {epoch+1} avg loss: {meter.avg:.4f}")

        # TODO: save checkpoint
        ckpt_dir = os.path.join(config.output_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"epoch_{epoch}.pt"))

    return model
