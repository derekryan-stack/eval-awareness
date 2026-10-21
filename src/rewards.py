"""Reward functions and answer checking shared by training and evaluation.

A completion is "<reasoning></think><response>" (the chat template opens the turn with
<think>). Rewards are selected by name in the config's `rewards` list; see build_rewards.
"""

import re
from fractions import Fraction


def completion_text(completion):
    # Chat prompts yield completions as [{"role": "assistant", "content": ...}].
    return completion[0]["content"] if isinstance(completion, list) else completion


def split_completion(text):
    """Return (reasoning, response, finished_thinking)."""
    reasoning, sep, response = text.partition("</think>")
    return reasoning, response, bool(sep)


def last_boxed(text):
    """Contents of the last \\boxed{...}, handling nested braces; None if absent."""
    start = text.rfind("\\boxed{")
    if start == -1:
        return None
    i, depth = start + len("\\boxed{"), 1
    for j in range(i, len(text)):
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            return text[i:j]
    return None


def to_number(s):
    """Parse '11/16', '\\dfrac{11}{16}', '0.6875', '5' into a Fraction; None if not numeric."""
    s = re.sub(r"\\[dt]?frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\[dt]?frac(\d)(\d)", r"\1/\2", s)  # \frac12
    s = s.replace("\\left", "").replace("\\right", "").replace("$", "").replace(" ", "").rstrip(".")
    m = re.fullmatch(r"\(?(-?\d+(?:\.\d+)?)\)?(?:/\(?(-?\d+(?:\.\d+)?)\)?)?", s)
    if not m:
        return None
    num = Fraction(m.group(1))
    return num / Fraction(m.group(2)) if m.group(2) else num


def is_correct(response, answer):
    """Numeric answers: compare the last \\boxed{} (or last number) to `answer` within 1e-4.
    Text answers: whole-word, case-insensitive match after </think>."""
    target = to_number(str(answer))
    if target is None:
        return re.search(rf"(?<!\w){re.escape(str(answer).strip())}(?!\w)", response, re.IGNORECASE) is not None
    candidate = last_boxed(response)
    if candidate is None:
        # Rewrite \frac{6}{11} as 6/11 first so it isn't read as the number 11.
        flat = re.sub(r"\\[dt]?frac\{\s*(-?\d+)\s*\}\{\s*(\d+)\s*\}", r"\1/\2", response)
        numbers = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", flat)
        candidate = numbers[-1] if numbers else None
    value = to_number(candidate) if candidate is not None else None
    return value is not None and abs(float(value) - float(target)) < 1e-4


def task_reward(prompts, completions, answer, **kwargs):
    """1.0 if the response after </think> has the reference answer, else 0.0."""
    rewards = []
    for completion, ref in zip(completions, answer):
        _, response, finished = split_completion(completion_text(completion))
        rewards.append(1.0 if finished and is_correct(response, ref) else 0.0)
    return rewards


def make_cot_length_penalty(tokenizer, coef, max_tokens):
    """-coef * (reasoning tokens / max_tokens), capped at -coef. Unfinished reasoning gets -coef."""

    def cot_length_penalty(prompts, completions, **kwargs):
        rewards, lengths = [], []
        for completion in completions:
            reasoning, _, finished = split_completion(completion_text(completion))
            n = len(tokenizer(reasoning, add_special_tokens=False)["input_ids"]) if finished else max_tokens
            lengths.append(n)
            rewards.append(-coef * min(n, max_tokens) / max_tokens)
        if "log_metric" in kwargs:
            kwargs["log_metric"]("cot_tokens/mean", sum(lengths) / len(lengths))
        return rewards

    return cot_length_penalty


def build_rewards(cfg, tokenizer):
    """Map the config's `rewards` entries to reward functions and weights."""
    funcs, weights = [], []
    for entry in cfg.get("rewards", [{"name": "task"}]):
        if entry["name"] == "task":
            funcs.append(task_reward)
        elif entry["name"] == "cot_length":
            funcs.append(make_cot_length_penalty(tokenizer, entry["coef"], cfg["training"]["max_completion_length"]))
        else:
            raise SystemExit(f"Unknown reward {entry['name']!r}; expected 'task' or 'cot_length'")
        weights.append(entry.get("weight", 1.0))
    return funcs, weights
