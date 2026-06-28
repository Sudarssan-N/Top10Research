"""Answer extraction and grading for math benchmarks."""
from __future__ import annotations

import re
from fractions import Fraction
from typing import Optional, Tuple

import sympy
from sympy.parsing.sympy_parser import parse_expr


_HASH_ANSWER_RE = re.compile(r"####\s*([^\n]+)")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")


def extract_boxed(text: str) -> Optional[str]:
    """Extract the content of the last ``\\boxed{...}``, with proper brace matching.

    A regex like ``\\boxed\\{([^}]*)\\}`` breaks on nested braces — e.g.
    ``\\boxed{\\frac{1}{2}}`` would yield ``\\frac{1`` — which silently corrupts
    accuracy on MATH-500 (fractions, matrices, etc.). We scan for the matching
    closing brace instead.
    """
    results = []
    idx = 0
    needle = "\\boxed"
    while True:
        start = text.find(needle, idx)
        if start == -1:
            break
        brace = text.find("{", start)
        if brace == -1:
            idx = start + len(needle)
            continue
        depth = 0
        for j in range(brace, len(text)):
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    results.append(text[brace + 1 : j].strip())
                    break
        idx = brace + 1
    return results[-1] if results else None


def extract_gsm8k_answer(text: str) -> Optional[str]:
    m = _HASH_ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()
    boxed = extract_boxed(text)
    if boxed:
        return boxed
    nums = _NUMBER_RE.findall(text)
    return nums[-1] if nums else None


def extract_math_answer(text: str) -> Optional[str]:
    boxed = extract_boxed(text)
    if boxed:
        return boxed
    nums = _NUMBER_RE.findall(text)
    return nums[-1] if nums else None


_FRAC_RE = re.compile(r"\\[dt]?frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")


def normalize_answer(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    s = str(text).strip()
    # Drop wrapping \boxed{...} / \text{...} but keep contents.
    s = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mbox\s*\{([^{}]*)\}", r"\1", s)
    # LaTeX fractions -> (a)/(b) so Fraction/sympy can compare. Run twice for
    # simple nesting like \frac{1}{\frac{2}{3}} (one level is enough in practice).
    for _ in range(2):
        s, n = _FRAC_RE.subn(r"(\1)/(\2)", s)
        if n == 0:
            break
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\ ", "")
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("^\\circ", "").replace("^{\\circ}", "").replace("\\%", "").replace("%", "")
    s = s.replace("$", "").replace(",", "")
    s = s.replace("\\$", "").replace("\\", "")
    s = re.sub(r"\s+", "", s)
    s = s.strip().rstrip(".")
    return s if s else None


def extract_gold_math(gold: str) -> Optional[str]:
    """Gold answers are clean references — prefer boxed, else the whole string.

    Unlike model completions we must NOT fall back to "last number in text":
    for a clean gold like ``\\frac{1}{2}`` that would return ``2``.
    """
    boxed = extract_boxed(gold)
    if boxed is not None:
        return boxed
    return normalize_answer(gold)


def _try_parse(expr: str):
    expr = expr.replace("^", "**")
    try:
        return parse_expr(expr, evaluate=True)
    except Exception:
        return None


def answers_equal(pred: Optional[str], gold: Optional[str], tol: float = 1e-4) -> bool:
    pred_n = normalize_answer(pred)
    gold_n = normalize_answer(gold)
    if pred_n is None or gold_n is None:
        return False
    if pred_n == gold_n:
        return True

    # Numeric compare
    try:
        pf = float(Fraction(pred_n))
        gf = float(Fraction(gold_n))
        return abs(pf - gf) <= tol
    except Exception:
        pass

    # Sympy compare
    p_sym = _try_parse(pred_n)
    g_sym = _try_parse(gold_n)
    if p_sym is not None and g_sym is not None:
        try:
            return sympy.simplify(p_sym - g_sym) == 0
        except Exception:
            pass

    return pred_n.lower() == gold_n.lower()


def grade_completion(
    completion: str,
    gold: str,
    answer_type: str = "math",
) -> Tuple[bool, Optional[str]]:
    """Return (is_correct, extracted_prediction)."""
    if answer_type == "gsm8k":
        pred = extract_gsm8k_answer(completion)
        gold_extracted = extract_gsm8k_answer(gold) or normalize_answer(gold)
    elif answer_type in ("math", "aime", "mock"):
        pred = extract_math_answer(completion)
        gold_extracted = extract_gold_math(gold)
    else:
        pred = extract_math_answer(completion)
        gold_extracted = normalize_answer(gold)

    correct = answers_equal(pred, gold_extracted)
    return correct, pred


def gold_label(example) -> str:
    """Normalized reference answer for an example."""
    answer_type = example.metadata.get("answer_type", "math")
    if answer_type == "gsm8k":
        return extract_gsm8k_answer(example.gold_answer) or normalize_answer(example.gold_answer) or example.gold_answer
    return extract_gold_math(example.gold_answer) or example.gold_answer