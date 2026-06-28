from typing import Dict, Optional

import torch

from .calibration import compute_ece, fit_temperature, precision_at_target_fpr
from .config import ExperimentConfig
from .utils import get_dummy_loader


def evaluate_model(
    model,
    config: ExperimentConfig,
    split: str = "test",
    data_loader=None,
) -> Dict:
    """Evaluate FPRBoundedVerifier on held-out hidden states + calibration metrics."""
    device = config.get_device()
    model.eval().to(device)
    loader = data_loader or get_dummy_loader(num_batches=8, hidden_size=2048)

    all_logits, all_labels, all_scores = [], [], []
    with torch.no_grad():
        for batch in loader:
            hidden = batch["hidden"].to(device)
            target = batch["target"].to(device).float()
            out = model(hidden)
            all_logits.append(out["logit"].cpu())
            all_scores.append(out["score"].cpu())
            all_labels.append(target.cpu())

    logits = torch.cat(all_logits)
    scores = torch.cat(all_scores).numpy().tolist()
    labels = torch.cat(all_labels).numpy().astype(int).tolist()

    temperature = fit_temperature(logits, torch.tensor(labels, dtype=torch.float32))
    calibrated_logits = logits / max(temperature, 1e-3)
    cal_scores = torch.sigmoid(calibrated_logits).numpy().tolist()

    target_fpr = getattr(config, "target_fpr", 0.05)
    raw_op = precision_at_target_fpr(scores, labels, target_fpr=target_fpr)
    cal_op = precision_at_target_fpr(cal_scores, labels, target_fpr=target_fpr)
    thresh = model.find_threshold_for_target_fpr(
        torch.tensor(cal_scores), torch.tensor(labels, dtype=torch.float32), target_fpr=target_fpr
    )
    metrics_at_tau = model.compute_fpr_precision(
        torch.tensor(cal_scores), torch.tensor(labels, dtype=torch.float32), thresh
    )

    results = {
        "ece_raw": compute_ece(scores, labels),
        "ece_calibrated": compute_ece(cal_scores, labels),
        "temperature": temperature,
        "operating_point_raw": raw_op,
        "operating_point_calibrated": cal_op,
        "metrics_at_tau": metrics_at_tau,
        "num_samples": len(labels),
    }
    print(
        f"[evaluate] ECE {results['ece_raw']:.4f} -> {results['ece_calibrated']:.4f} | "
        f"precision@FPR≤{target_fpr}: {cal_op['precision']:.3f}"
    )
    return results


def run_ceiling_baselines(
    config: ExperimentConfig,
    max_examples: Optional[int] = None,
    mock: bool = False,
    output_dir: Optional[str] = None,
) -> Dict:
    from .phase1 import run_phase1_ceiling

    return run_phase1_ceiling(
        config=config,
        max_examples=max_examples,
        mock=mock,
        target_fpr=getattr(config, "target_fpr", 0.05),
        output_dir=output_dir,
    )
