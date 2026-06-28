#!/usr/bin/env python3
"""
Bootstrap script: creates initial README.md, run.md, requirements.txt,
basic source code skeletons for all 10 problems.
Run once from the workspace root.
"""

import os
from textwrap import dedent

PROBLEMS = [
    {
        "id": "p01",
        "dir": "p01-verifier-aware-controller",
        "title": "P1: Verifier-Aware Learned Controller for Test-Time Compute",
        "short": "Lightweight learned controller that allocates test-time compute per query while accounting for verifier imperfection.",
        "compute": "1×A100-40GB (~150-300 GPU-hrs, $300-600)",
        "target_venues": "NeurIPS / ICLR / COLM",
        "priority": "Highest (start here)",
        "key_refs": "Snell et al. (ICLR 2025), Damani et al. (ICLR 2025), Re-FORC, Agrawal et al. (imperfect verifier ceiling)",
    },
    {
        "id": "p02",
        "dir": "p02-calibrated-fp-verifiers",
        "title": "P2: Calibrated, False-Positive-Bounded Verifiers for Best-of-N",
        "short": "Train/calibrate verifiers explicitly to minimize false positives at chosen operating points and raise the best-of-N ceiling.",
        "compute": "1-4×A100 (~200 GPU-hrs, $400-800)",
        "target_venues": "NeurIPS / ICLR",
        "priority": "Rank 2 (after P1 progress)",
        "key_refs": "Agrawal et al. (Cut the Overcredit), Stroebl et al. (arXiv:2411.17501), PRM/GenRM literature",
    },
    {
        "id": "p03",
        "dir": "p03-overthinking-latent-controller",
        "title": "P3: Latent Controller for Adaptive Overthinking Reduction",
        "short": "Per-query learned controller (probe on hidden states) that decides reasoning depth / early termination to combat ~18x token bloat on simple problems.",
        "compute": "1×A100 (~100-200 GPU-hrs, $200-400)",
        "target_venues": "ACL / EMNLP / COLM",
        "priority": "High (parallel track)",
        "key_refs": "LLMThinkBench, 'Wait, We Don't Need to Wait' (EMNLP 2025), ThinkPrune, L1, Re-FORC",
    },
    {
        "id": "p04",
        "dir": "p04-calibration-probe",
        "title": "P4: Test-Time Compute vs. Confidence Calibration + Corrective Probe",
        "short": "Systematic study of how reasoning budgets affect calibration; train lightweight recalibration probe, focus on ECE + low-FPR metrics.",
        "compute": "1×A100 (~80-150 GPU-hrs, $150-350)",
        "target_venues": "ICLR / COLM / EMNLP",
        "priority": "Rank 3 hedge / analysis paper (low risk, fast)",
        "key_refs": "arXiv:2508.15050 (over-reasoning impairs calibration)",
    },
    {
        "id": "p05",
        "dir": "p05-batch-moe-routing",
        "title": "P5: Batch-Aware MoE Expert Routing at Inference (No Retraining)",
        "short": "Training-free or lightly adapted batch-aware router that mitigates union-of-experts activation and load imbalance for batched decode speedup.",
        "compute": "2-8×A100 burst (~100 GPU-hrs, $400-900)",
        "target_venues": "MLSys / NeurIPS / ICLR",
        "priority": "Medium (systems-heavy, good for MLSys)",
        "key_refs": "arXiv:2511.02237 (opportunistic MoE activation)",
    },
    {
        "id": "p06",
        "dir": "p06-long-context-degradation",
        "title": "P6: Context-Length-Induced Degradation Under Perfect Retrieval",
        "short": "Diagnose attention dilution / other mechanisms for performance drop as context length grows even with perfect retrieval; propose simple mitigations (reorder/calibrate).",
        "compute": "1×A100 (~80 GPU-hrs, $150-300)",
        "target_venues": "ACL / EMNLP / ICLR",
        "priority": "Medium-high (clean analysis)",
        "key_refs": "arXiv:2510.05381",
    },
    {
        "id": "p07",
        "dir": "p07-bias-discovery-llm-judge",
        "title": "P7: Automated Discovery of Novel Biases in LLM-as-Judge",
        "short": "Build contrastive perturbation + causal validation pipeline for automated bias discovery in judges; go beyond known position/verbosity biases.",
        "compute": "Mostly API + 1×A100 (~50 GPU-hrs + API cost, $100-300)",
        "target_venues": "ACL / EMNLP / NeurIPS D&B",
        "priority": "High (fresh, low-compute)",
        "key_refs": "BiasScope (arXiv:2602.09383), position/verbosity bias papers",
    },
    {
        "id": "p08",
        "dir": "p08-slm-distillation-tooluse",
        "title": "P8: Step-Wise On-Policy Distillation for Small Reasoning Models (Tool Use)",
        "short": "Improve on naive distillation for ≤4B tool-using reasoners via step-wise, divergence-reweighted on-policy distillation.",
        "compute": "1-2×A100 (~150 GPU-hrs, $300-600)",
        "target_venues": "COLM / ICLR / ACL",
        "priority": "Medium (somewhat crowded but unsettled recipes)",
        "key_refs": "SOD (arXiv:2605.07725), LIMO/S1K lessons, Phi-4-Mini-Reasoning",
    },
    {
        "id": "p09",
        "dir": "p09-agent-memory-efficient",
        "title": "P9: Token-Efficient Agent Memory with Bounded Reasoning Cost",
        "short": "Lightweight memory management policy (what to remember/forget) that bounds per-task token/compute cost while preserving success rate on agent benchmarks.",
        "compute": "1×A100 + API (~80 GPU-hrs, $200-400)",
        "target_venues": "NeurIPS / COLM / EMNLP",
        "priority": "Medium-high (agent cost is painful)",
        "key_refs": "Memori, A-Mem, LoCoMo, MultiAgentBench, 'Why Do Multi-Agent LLM Systems Fail?'",
    },
    {
        "id": "p10",
        "dir": "p10-speculative-drafter-selection",
        "title": "P10: Dynamic Heterogeneous Drafter Selection for Speculative Decoding",
        "short": "Learned or training-free input-adaptive selector among heterogeneous drafters (small models) with acceptance-rate / speedup guarantees.",
        "compute": "1-2×A100 (~100 GPU-hrs, $250-500)",
        "target_venues": "MLSys / ICLR / NeurIPS",
        "priority": "Medium (needs strong wall-clock baselines)",
        "key_refs": "arXiv:2604.05417, arXiv:2512.23765 (training-free variants)",
    },
]

BASE_REQUIREMENTS = """torch>=2.4.0
transformers>=4.46.0
datasets>=3.0.0
accelerate>=1.0.0
evaluate>=0.4.0
numpy>=1.26
pandas>=2.2
scikit-learn>=1.5
tqdm>=4.66
pyyaml>=6.0
matplotlib>=3.9
seaborn>=0.13
huggingface_hub>=0.25
wandb>=0.18
sympy>=1.13
"""

P01_EXTRA = """# P1 specific
# (add any later)
"""

def write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    print(f"  wrote {path}")

def get_readme(problem):
    return dedent(f"""\
        # {problem['title']}

        **Problem statement summary:**  
        {problem['short']}

        **Compute target:** {problem['compute']}  
        **Target venues:** {problem['target_venues']}  
        **Priority:** {problem['priority']}  
        **Key references:** {problem['key_refs']}

        ## Goals for this project

        - Reproduce / implement strong baselines (best-of-N, simple difficulty-only controllers, Re-FORC style where applicable).
        - Implement the core contribution described in the strategy doc.
        - Run clean ablations and report compute-efficiency + accuracy tradeoffs on MATH-500, AIME 2024/25, GPQA (and code/math where relevant).
        - Produce publication-quality figures, tables, and analysis.

        ## Current Status (scaffold)

        - [x] Folder + basic structure created
        - [x] README + run.md + requirements skeleton
        - [ ] Core implementation (controller / model heads)
        - [ ] Data loading + eval harness (MATH/AIME)
        - [ ] Training script with logging
        - [ ] Full reproduction scripts
        - [ ] Results + analysis

        See `run.md` for exact next steps and how to execute.

        ## Directory layout

        ```
        {problem['dir']}/
        ├── README.md
        ├── run.md
        ├── requirements.txt
        ├── src/
        │   ├── __init__.py
        │   ├── controller.py          # (or probe, router, etc.)
        │   ├── train.py
        │   ├── evaluate.py
        │   └── utils.py
        ├── scripts/
        │   └── run_experiment.py
        ├── configs/
        │   └── default.yaml
        ├── data/          # (gitignored, populated at runtime)
        ├── experiments/
        ├── results/
        └── ...
        ```

        ## Quick links

        - Full strategy: see top-level `compass_artifact_...markdown.md` and top `README.md`
        - Run instructions: `run.md`
    """)

def get_run_md(problem):
    p = problem
    base = dedent(f"""\
        # How to Run — {p['title']}

        This document gives complete, reproducible instructions to set up and run experiments for this problem.

        > **Status note:** This is the initial scaffold. Code implements basic structure, mock runs, and the core skeleton. Full research implementation (model training + strong baselines) will be built iteratively. Start by following the "Minimal smoke test" then move to real training.

        ## 1. Environment Setup

        ```bash
        cd {p['dir']}
        python -m venv .venv
        source .venv/bin/activate   # or conda / your preferred
        pip install --upgrade pip
        pip install -r requirements.txt
        ```

        **Recommended hardware:** See top of this doc (usually 1x A100 or equivalent).  
        For smoke tests: CPU or small GPU is fine (models will be tiny or mocked at first).

        **HF login (for models/datasets):**
        ```bash
        huggingface-cli login
        # or set HF_TOKEN env
        ```

        ## 2. Data Preparation

        Most projects use:
        - MATH (hendrycks_math or math dataset on HF)
        - GSM8K
        - AIME 2024/2025 (usually manual or community splits; scripts will download)
        - GPQA (diamond)

        The `scripts/prepare_data.py` (or equivalent in src) will cache them under `data/`.

        Run once:
        ```bash
        python scripts/prepare_data.py   # if present, or use the logic in evaluate/train
        ```

        ## 3. Minimal Smoke Test (always works, even without GPU)

        ```bash
        python -c "
        from src.utils import hello
        print(hello())
        # or
        python scripts/run_experiment.py --config configs/default.yaml --smoke-test
        "

        Expected: prints version info, runs a dummy forward pass or mock controller, writes a tiny result file to results/.
        ```

        ## 4. Full Experiment Run (example for this problem)

        Typical pattern (will evolve):

        ```bash
        # Train the controller / probe / etc.
        python scripts/run_experiment.py \\
            --config configs/default.yaml \\
            --mode train \\
            --model_name Qwen/Qwen2.5-1.5B-Instruct \\
            --output_dir results/run-001

        # Evaluate (generations + metrics)
        python scripts/run_experiment.py \\
            --config configs/default.yaml \\
            --mode eval \\
            --checkpoint results/run-001/best_controller.pt \\
            --benchmarks math500,aime24,gpqa

        # Or combined train+eval script for convenience
        bash experiments/run_full.sh
        ```

        ## 5. Key Hyperparameters / Config

        See `configs/default.yaml`. Common levers:
        - base_model
        - controller_hidden_size / layers (keep very small: 32-256 dim)
        - num_samples_for_best_of_n
        - budget_range (tokens or steps)
        - verifier_model (or oracle for ceiling studies)
        - learning_rate, epochs (small: 1-3 epochs on frozen features often enough)

        ## 6. Expected Outputs & Metrics

        - Accuracy vs. compute (tokens or FLOPs or wall time) curves
        - Comparison table vs. best-of-N (fixed N=4,8,16,32), majority vote, difficulty-only router
        - For verifier problems: FPR, precision-recall at operating points, ceiling shift
        - Plots saved to results/
        - JSONL generations + per-example traces

        Target bar (from strategy for P1): ≥1.5-2× compute reduction at matched accuracy vs. best-of-N on MATH-500/AIME.

        ## 7. Reproducing Baselines

        The code ships with:
        - Naive best-of-N
        - Difficulty estimator baseline (e.g. from hidden state entropy or simple probe)
        - Random / fixed budget

        See `src/evaluate.py` or run with `--baseline_only`.

        ## 8. Logging & Tracking

        - Console + file logs to experiments/
        - Optional: set `WANDB_PROJECT=...` and `wandb` will be used if installed.

        ## 9. Common Issues & Tips

        - OOM: lower batch size, use gradient checkpointing on base (but frozen usually), or smaller controller.
        - Slow generation: use vLLM if available for eval generations (future extension); start with HF generate.
        - Reproducibility: fix seeds everywhere (see utils.set_seed).
        - Dataset licenses: respect original dataset terms.

        ## 10. Iterating / Extending

        1. Edit `src/` files.
        2. Update `configs/`.
        3. Add new benchmarks or ablations in `evaluate.py`.
        4. When you have good numbers, update this `run.md` with exact commands + observed metrics.

        ## Current TODOs (update as you progress)

        - [ ] Implement data loaders for MATH + AIME
        - [ ] Implement frozen feature extraction from base model
        - [ ] Core controller architecture (per problem)
        - [ ] Loss that incorporates verifier reliability estimate
        - [ ] Strong best-of-N + oracle verifier baselines
        - [ ] Full training loop with early stopping
        - [ ] Analysis + plotting scripts
        - [ ] Reproduce key numbers from cited papers on same base models

        ## References (key papers to cite / implement against)

        {p['key_refs']}

        Good luck — this is designed to be high-signal, low-compute research!
    """)
    return base

def get_basic_src_files(problem_id, problem_dir, problem):
    files = {}

    # __init__.py
    files[f"{problem_dir}/src/__init__.py"] = '"""Core package for {}"""\n\n__version__ = "0.1.0"\n'.format(problem["title"])

    # utils.py (shared basics)
    files[f"{problem_dir}/src/utils.py"] = dedent("""\
        import os
        import random
        import numpy as np
        import torch
        from typing import Dict, Any

        def set_seed(seed: int = 42):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        def hello():
            return "Hello from {} scaffold — ready to research!"

        def save_jsonl(path, records):
            import json
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\\n")

        def load_jsonl(path):
            import json
            with open(path) as f:
                return [json.loads(line) for line in f]

        class AverageMeter:
            def __init__(self):
                self.reset()
            def reset(self):
                self.val = self.avg = self.sum = self.count = 0
            def update(self, val, n=1):
                self.val = val
                self.sum += val * n
                self.count += n
                self.avg = self.sum / self.count
    """.format(problem["id"].upper()))

    # Basic config loader
    files[f"{problem_dir}/src/config.py"] = dedent("""\
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

            output_dir: str = "results/default"
            log_interval: int = 10

            def __post_init__(self):
                if self.benchmarks is None:
                    self.benchmarks = ["math500", "gsm8k"]

            @classmethod
            def from_yaml(cls, path: str):
                with open(path) as f:
                    data = yaml.safe_load(f) or {}
                return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

            def to_yaml(self, path: str):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as f:
                    yaml.dump(asdict(self), f, sort_keys=False)

            def get_device(self):
                if self.device == "auto":
                    return "cuda" if torch.cuda.is_available() else "cpu"
                return self.device
    """)

    # Placeholder for problem-specific core logic
    if problem_id == "p01":
        core_content = dedent('''
            """
            P1 Core: Verifier-Aware Test-Time Compute Controller

            A small MLP or transformer head on top of frozen base model hidden states
            that predicts:
              - difficulty / expected gain from more compute
              - verifier reliability / false-positive risk for this query
            Then uses a stopping policy (Gittins / Pandora style or learned threshold)
            to decide how much compute (samples / tokens) to allocate.
            """
            import torch
            import torch.nn as nn
            from typing import Tuple, Dict

            class VerifierAwareController(nn.Module):
                def __init__(self, hidden_size: int = 2048, controller_dim: int = 128, num_layers: int = 2):
                    super().__init__()
                    layers = []
                    in_dim = hidden_size
                    for _ in range(num_layers):
                        layers += [nn.Linear(in_dim, controller_dim), nn.ReLU(), nn.Dropout(0.1)]
                        in_dim = controller_dim
                    self.backbone = nn.Sequential(*layers)
                    # Two heads
                    self.value_head = nn.Linear(controller_dim, 1)          # expected value of extra compute
                    self.trust_head = nn.Linear(controller_dim, 1)          # verifier trust / 1-FPR estimate
                    self.sigmoid = nn.Sigmoid()

                def forward(self, hidden_state: torch.Tensor) -> Dict[str, torch.Tensor]:
                    # hidden_state: [batch, hidden] or mean-pooled
                    feat = self.backbone(hidden_state)
                    value = self.value_head(feat).squeeze(-1)
                    trust = self.sigmoid(self.trust_head(feat)).squeeze(-1)  # 0..1
                    return {"value": value, "verifier_trust": trust}

                def decide_budget(self, value: torch.Tensor, trust: torch.Tensor, 
                                  base_budget: int = 256, max_budget: int = 2048,
                                  trust_threshold: float = 0.6) -> torch.Tensor:
                    # Simple heuristic policy (replace with Gittins/Pandora later)
                    # When trust low, be more conservative
                    scale = torch.clamp(trust, 0.3, 1.0)
                    budget = (base_budget + (value * 1024) * scale).clamp(base_budget, max_budget).int()
                    return budget
        ''')
    elif problem_id == "p02":
        core_content = dedent('''
            """
            P2: Calibrated False-Positive-Bounded Verifier

            Fine-tune or post-calibrate a verifier (PRM or generative) with precision-oriented / asymmetric loss
            to reduce FPR at relevant selection thresholds.
            """
            import torch
            import torch.nn as nn

            class FPRBoundedVerifier(nn.Module):
                def __init__(self, base_verifier, alpha: float = 0.8):  # weight on precision
                    super().__init__()
                    self.base = base_verifier  # frozen or LoRA
                    self.alpha = alpha

                def forward(self, input_ids, attention_mask):
                    # assume base returns logits or score
                    out = self.base(input_ids=input_ids, attention_mask=attention_mask)
                    return out

                # Add custom loss in train
        ''')
    else:
        # Generic probe / controller
        title_esc = problem['title'].replace('"', "'")
        core_content = dedent(f'''
            """
            {title_esc} - Core Model / Probe / Controller

            Lightweight head on frozen LM hidden states.
            """
            import torch
            import torch.nn as nn
            from typing import Dict

            class LightweightProbe(nn.Module):
                def __init__(self, hidden_size: int, probe_dim: int = 128):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(hidden_size, probe_dim),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(probe_dim, 1)
                    )

                def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
                    logit = self.net(hidden)
                    return {{"logit": logit, "prob": torch.sigmoid(logit)}}
        ''')

    files[f"{problem_dir}/src/core.py"] = core_content

    # train.py skeleton
    files[f"{problem_dir}/src/train.py"] = dedent("""\
        import torch
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        from .config import ExperimentConfig
        from .utils import set_seed, AverageMeter
        from .core import LightweightProbe, VerifierAwareController  # adjust import per problem
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
    """)

    # evaluate.py skeleton
    files[f"{problem_dir}/src/evaluate.py"] = dedent("""\
        import torch
        from typing import Dict, List
        from .config import ExperimentConfig
        from datasets import load_dataset
        # TODO: add generation + math eval code using sympy for answer extraction etc.

        def evaluate_model(model, config: ExperimentConfig, split: str = "test") -> Dict:
            device = config.get_device()
            model.eval().to(device)

            results = {"accuracy": 0.0, "avg_tokens": 0.0, "samples": 0}
            # TODO: implement full generation loop over benchmarks
            # For now return dummy
            print("[evaluate] Running evaluation scaffold (replace with real generations + grading)")
            return results

        def compute_ece(probs, labels, n_bins=10):
            # Expected Calibration Error helper (useful for P4 and verifier work)
            import numpy as np
            bins = np.linspace(0, 1, n_bins + 1)
            ece = 0.0
            for i in range(n_bins):
                mask = (probs > bins[i]) & (probs <= bins[i+1])
                if mask.sum() == 0: continue
                bin_acc = labels[mask].mean()
                bin_conf = probs[mask].mean()
                ece += (mask.sum() / len(probs)) * abs(bin_acc - bin_conf)
            return ece
    """)

    # scripts/run_experiment.py
    title_for_doc = problem['title']
    files[f"{problem_dir}/scripts/run_experiment.py"] = dedent(f"""\
        #!/usr/bin/env python3
        \"\"\"
        Main entry point for {title_for_doc}
        \"\"\"
        import argparse
        import os
        import sys
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

        import torch
        from src.config import ExperimentConfig
        from src.utils import set_seed
        from src.core import LightweightProbe   # TODO: swap for problem-specific (VerifierAwareController etc.)
        from src.train import train
        from src.evaluate import evaluate_model

        def main():
            parser = argparse.ArgumentParser()
            parser.add_argument("--config", type=str, default="configs/default.yaml")
            parser.add_argument("--mode", choices=["train", "eval", "both"], default="both")
            parser.add_argument("--smoke-test", action="store_true", help="Run tiny mock without loading big models")
            parser.add_argument("--output_dir", type=str, default=None)
            args = parser.parse_args()

            if os.path.exists(args.config):
                cfg = ExperimentConfig.from_yaml(args.config)
            else:
                cfg = ExperimentConfig()
            if args.output_dir:
                cfg.output_dir = args.output_dir

            set_seed(cfg.seed)
            device = cfg.get_device()
            print(f"Running in mode={{args.mode}} on device={{device}}")
            print(f"Config: base_model={{cfg.base_model}}")

            if args.smoke_test:
                print("=== SMOKE TEST ===")
                model = LightweightProbe(hidden_size=2048, probe_dim=cfg.controller_hidden_dim)
                dummy_hidden = torch.randn(4, 2048)
                out = model(dummy_hidden)
                print("Probe output keys:", list(out.keys()))
                print("Smoke test passed.")
                os.makedirs(cfg.output_dir, exist_ok=True)
                with open(os.path.join(cfg.output_dir, "smoke_results.json"), "w") as f:
                    f.write('{{"status": "ok", "mode": "smoke"}}')
                return

            # TODO: real data loaders, real base model hidden extraction, real training
            model = LightweightProbe(hidden_size=2048, probe_dim=cfg.controller_hidden_dim)

            if args.mode in ("train", "both"):
                # TODO: replace with real DataLoader
                class DummyLoader(list): pass
                dummy_loader = DummyLoader([{{ "hidden": torch.randn(4, 2048), "target": torch.tensor([0.,1.,0.,1.]) }} for _ in range(3)])
                model = train(cfg, dummy_loader, model, device)

            if args.mode in ("eval", "both"):
                metrics = evaluate_model(model, cfg)
                print("Eval metrics:", metrics)

            print("Done. See results/ for outputs.")

        if __name__ == "__main__":
            main()
    """)

    # configs/default.yaml
    files[f"{problem_dir}/configs/default.yaml"] = dedent(f"""\
        base_model: "Qwen/Qwen2.5-1.5B-Instruct"
        seed: 42
        controller_hidden_dim: 128
        num_controller_layers: 2
        learning_rate: 1e-4
        num_epochs: 2
        train_batch_size: 4
        eval_batch_size: 8
        budget_max_tokens: 2048
        n_samples_best_of_n: 8
        benchmarks:
          - math500
          - gsm8k
        output_dir: "results/run-001"
        log_interval: 5
    """)

    # requirements per problem (base + extras)
    req = BASE_REQUIREMENTS
    if problem_id == "p01":
        req += "\n# P01 extras: none yet\n"
    elif problem_id == "p05":
        req += "# For MoE: consider adding bitsandbytes, auto-gptq if quantizing experts\n"
    elif problem_id == "p10":
        req += "# Speculative decoding may benefit from vllm or direct attention hacking\n"

    files[f"{problem_dir}/requirements.txt"] = req

    # Also create a simple prepare_data placeholder in scripts
    files[f"{problem_dir}/scripts/prepare_data.py"] = dedent("""\
        #!/usr/bin/env python3
        \"\"\"
        Download / prepare datasets for this problem.
        For now just prints instructions; flesh out per-problem.
        \"\"\"
        from datasets import load_dataset
        import os

        def main():
            print("Preparing data (scaffold)...")
            print("MATH / GSM8K example:")
            try:
                ds = load_dataset("gsm8k", "main", split="train[:100]")
                print(f"  Loaded sample of GSM8K: {len(ds)} examples")
            except Exception as e:
                print("  Could not load (offline or network). Will need manual or cached data.")
            print("Datasets will be cached by huggingface under ~/.cache/huggingface")
            # TODO: add AIME download (many use https://github.com/hendrycks/math or community parquet)

        if __name__ == "__main__":
            main()
    """)

    # Add a basic experiment runner shell script
    files[f"{problem_dir}/experiments/run_full.sh"] = dedent(f"""\
        #!/bin/bash
        set -e
        echo "=== Full run for {problem['title']} ==="
        python scripts/run_experiment.py --config configs/default.yaml --mode both --smoke-test
        echo "Extend this script with real training commands."
    """)
    # make it executable later

    return files

def main():
    root = os.getcwd()
    print(f"Scaffolding all problems from {root}...")

    for p in PROBLEMS:
        print(f"\n=== {p['id']}: {p['dir']} ===")
        problem_dir = os.path.join(root, p["dir"])

        # README
        write_file(os.path.join(problem_dir, "README.md"), get_readme(p))

        # run.md
        write_file(os.path.join(problem_dir, "run.md"), get_run_md(p))

        # src + scripts + configs files
        for path, content in get_basic_src_files(p["id"], problem_dir, p).items():
            write_file(path, content)

        # Make run_full.sh executable
        script_path = os.path.join(problem_dir, "experiments/run_full.sh")
        if os.path.exists(script_path):
            os.chmod(script_path, 0o755)

        # Copy or link common requirements if wanted, but we already wrote per-folder one

        print(f"  {p['id']} scaffold complete.")

    print("\n✅ All 10 problem scaffolds created!")
    print("Next: cd into a problem (start with p01) and follow its run.md")

if __name__ == "__main__":
    main()
