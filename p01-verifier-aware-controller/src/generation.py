"""Model loading and text generation for Phase 1 baselines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


MATH_PROMPT_TEMPLATE = (
    "Solve the following problem step by step. "
    "Put your final answer in \\boxed{{}}.\n\nProblem:\n{question}\n\nSolution:"
)


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class MockGenerator:
    """Deterministic mock for smoke tests — no GPU / HF download."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7, n: int = 1) -> List[GenerationResult]:
        import re
        import random

        rng = random.Random(self.seed + hash(prompt) % 10_000)
        nums = [int(x) for x in re.findall(r"\d+", prompt)]
        if len(nums) >= 2:
            ans = nums[0] + nums[1]
        elif nums:
            ans = nums[0] * 2
        else:
            ans = rng.randint(1, 99)

        results = []
        for i in range(n):
            # ~30% wrong samples for N>1 diversity
            wrong = i > 0 and rng.random() < 0.35
            final = ans + (rng.randint(1, 5) if wrong else 0)
            text = f"Reasoning step {i+1}...\nFinal answer: \\boxed{{{final}}}"
            comp_tok = max(20, len(text.split()))
            prompt_tok = max(10, len(prompt.split()))
            results.append(
                GenerationResult(
                    text=text,
                    prompt_tokens=prompt_tok,
                    completion_tokens=comp_tok,
                    total_tokens=prompt_tok + comp_tok,
                )
            )
        return results


class HFGenerator:
    """HuggingFace causal LM generator."""

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

        torch_dtype = None
        if dtype == "auto":
            torch_dtype = torch.float16 if device == "cuda" else torch.float32
        elif dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16
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

    def _format_prompt(self, question: str) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": MATH_PROMPT_TEMPLATE.format(question=question)}]
            try:
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        return MATH_PROMPT_TEMPLATE.format(question=question)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        n: int = 1,
    ) -> List[GenerationResult]:
        text_prompt = self._format_prompt(prompt) if "\nProblem:" not in prompt else prompt
        inputs = self.tokenizer(text_prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": max(temperature, 1e-5),
            "pad_token_id": self.tokenizer.pad_token_id,
            "num_return_sequences": n,
        }

        outputs = self.model.generate(**inputs, **gen_kwargs)
        results: List[GenerationResult] = []
        for seq in outputs:
            completion_ids = seq[prompt_len:]
            completion = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
            comp_len = int(completion_ids.shape[0])
            results.append(
                GenerationResult(
                    text=completion,
                    prompt_tokens=prompt_len,
                    completion_tokens=comp_len,
                    total_tokens=prompt_len + comp_len,
                )
            )
        return results

    @torch.inference_mode()
    def prompt_token_count(self, prompt: str) -> int:
        text_prompt = self._format_prompt(prompt)
        ids = self.tokenizer(text_prompt, return_tensors="pt")["input_ids"]
        return int(ids.shape[1])


def build_generator(model_name: str, device: str = "auto", mock: bool = False):
    if mock or model_name.lower() in ("mock", "none", "smoke"):
        return MockGenerator()
    return HFGenerator(model_name=model_name, device=device)