"""Frozen LM hidden-state extraction for P1 Phase 2."""
from __future__ import annotations

import hashlib
import os
import re
from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn

from .generation import MATH_PROMPT_TEMPLATE

PoolingMode = Literal["last_token", "mean_prompt"]


def model_slug(model_name: str) -> str:
    s = model_name.replace("/", "__")
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", s)


def feature_cache_path(data_dir: str, model_name: str, benchmark: str) -> str:
    return os.path.join(data_dir, "features", model_slug(model_name), f"{benchmark}.pt")


class MockBackbone(nn.Module):
    """Deterministic fake hidden states for smoke tests."""

    def __init__(self, hidden_size: int = 2048, seed: int = 42):
        super().__init__()
        self.hidden_size = hidden_size
        self.seed = seed

    def format_prompt(self, question: str) -> str:
        return MATH_PROMPT_TEMPLATE.format(question=question)

    @torch.inference_mode()
    def encode(self, prompts: List[str], pooling: PoolingMode = "last_token", batch_size: int = 4) -> torch.Tensor:
        rows = []
        for p in prompts:
            h = hashlib.md5((str(self.seed) + p).encode()).hexdigest()
            gen = torch.Generator().manual_seed(int(h[:8], 16))
            vec = torch.randn(self.hidden_size, generator=gen)
            rows.append(vec)
        return torch.stack(rows, dim=0)


class FrozenBackbone:
    """Frozen HF causal LM — prompt forward pass only (no generation)."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: Optional[str] = "auto",
        trust_remote_code: bool = True,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if dtype == "auto":
            torch_dtype = torch.float16 if device == "cuda" else torch.float32
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = {"trust_remote_code": trust_remote_code}
        if device == "cuda":
            load_kwargs["torch_dtype"] = torch_dtype
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        if device != "cuda":
            self.model = self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        cfg = self.model.config
        self.hidden_size = int(
            getattr(cfg, "hidden_size", None)
            or getattr(cfg, "n_embd", None)
            or getattr(cfg, "d_model", 2048)
        )

    def format_prompt(self, question: str) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": MATH_PROMPT_TEMPLATE.format(question=question)}]
            try:
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        return MATH_PROMPT_TEMPLATE.format(question=question)

    @torch.inference_mode()
    def encode(self, prompts: List[str], pooling: PoolingMode = "last_token", batch_size: int = 4) -> torch.Tensor:
        all_hiddens: List[torch.Tensor] = []
        formatted = [self.format_prompt(p) if "\nProblem:" not in p else p for p in prompts]

        for i in range(0, len(formatted), batch_size):
            batch_text = formatted[i : i + batch_size]
            inputs = self.tokenizer(
                batch_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            outputs = self.model(**inputs, output_hidden_states=True)
            # last layer: [batch, seq, hidden]
            hidden = outputs.hidden_states[-1]

            if pooling == "last_token":
                # rightmost non-pad token per row
                if self.tokenizer.pad_token_id is not None:
                    attn = inputs["attention_mask"]
                    last_idx = attn.sum(dim=1) - 1
                    pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last_idx]
                else:
                    pooled = hidden[:, -1, :]
            else:
                attn = inputs["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1.0)

            all_hiddens.append(pooled.detach().cpu().float())

        return torch.cat(all_hiddens, dim=0)


def build_backbone(model_name: str, device: str = "auto", mock: bool = False, mock_hidden_size: int = 2048):
    if mock or model_name.lower() in ("mock", "none", "smoke"):
        return MockBackbone(hidden_size=mock_hidden_size)
    return FrozenBackbone(model_name=model_name, device=device)