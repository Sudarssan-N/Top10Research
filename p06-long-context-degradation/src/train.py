import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .config import ExperimentConfig
from .utils import set_seed, AverageMeter
from .core import ContextDegradationProbe  # P6 specific
import os

def train(config: ExperimentConfig, train_loader: DataLoader, model: torch.nn.Module, device: str):
    set_seed(config.seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()  # or MSE / custom for the problem

    model.train()
    for epoch in range(config.num_epochs):
        meter = AverageMeter()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for step, batch in enumerate(pbar):
            # TODO: batch contains hidden states + labels (or rewards + verifier scores)
            hidden = batch["hidden"].to(device)
            target = batch["target"].to(device).float()

            out = model(hidden)
            # Adapt loss head access per problem
            pred = out.get("logit", out.get("value", out.get("prob")))
            loss = criterion(pred.squeeze(), target)

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
