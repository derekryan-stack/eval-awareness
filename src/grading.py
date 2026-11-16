"""Answer checking shared by the task reward, evaluate.py and analyze.py (moved unchanged from rewards.py)."""

import re
from fractions import Fraction


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
    if m.group(2):
        den = Fraction(m.group(2))
        return None if den == 0 else num / den
    return num


def is_correct(response, answer):
    """Numeric answers: compare the last \\boxed{} (or last number) to `answer` within 1e-4.
    Text answers: whole-word, case-insensitive match in the response."""
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
