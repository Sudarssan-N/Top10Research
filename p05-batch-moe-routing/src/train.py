import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .config import ExperimentConfig
from .utils import set_seed, AverageMeter
from .core import BatchAwareMoERouter  # P5 specific
import os

def train(config: ExperimentConfig, train_loader: DataLoader, model: torch.nn.Module, device: str):
    set_seed(config.seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    # P5: simple regression or CE on expert distribution for load-aware routing
    criterion = torch.nn.MSELoss()

    model.train()
    for epoch in range(config.num_epochs):
        meter = AverageMeter()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for step, batch in enumerate(pbar):
            hidden = batch["hidden"].to(device)
            # target can be a simulated ideal expert distribution (one-hot or soft)
            target = batch.get("target_dist", batch.get("target")).to(device).float()
            if target.dim() == 1:
                target = torch.nn.functional.one_hot(target.long(), num_classes=8).float()

            out = model(hidden)
            pred = out["expert_probs"]
            # match shapes
            if pred.shape != target.shape:
                target = target[:, :pred.shape[-1]] if target.shape[-1] > pred.shape[-1] else target
            loss = criterion(pred, target)

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
