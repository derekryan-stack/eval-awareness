"""Parse and validate judge outputs into {"score", "category", "evidence", "explanation"}."""

import json
import re

from judge.rubric import CATEGORIES, MAX_SCORE


class JudgeParseError(ValueError):
    pass


def _candidate_objects(text):
    """Every JSON object that decodes starting at some '{' in text, in order."""
    decoder = json.JSONDecoder()
    found = []
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            found.append(obj)
    return found


def _repair_escapes(text):
    """Make LaTeX inside JSON strings parse as literal text: \\( \\frac \\boxed -> \\\\( \\\\frac ...

    Escape pairs are consumed left to right so an already-escaped \\\\ is left alone. \\b and \\f
    (backspace / form feed) are treated as LaTeX (\\boxed, \\frac): a judge never means them.
    """
    return re.sub(r"\\(.)", lambda m: m.group(0) if m.group(1) in '\\"/nrtu' else "\\\\" + m.group(1), text, flags=re.S)


def _coerce_score(value):
    if isinstance(value, bool):
        raise JudgeParseError(f"score must be an integer 0-{MAX_SCORE}, got {value!r}")
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        value = int(value.strip())
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if not isinstance(value, int):
        raise JudgeParseError(f"score must be an integer 0-{MAX_SCORE}, got {value!r}")
    if value not in CATEGORIES:
        raise JudgeParseError(f"score {value} is outside 0-{MAX_SCORE}")
    return value


def parse_judgment(raw):
    """Return a validated dict plus "warnings" (list). Raises JudgeParseError if unusable.

    Tolerates: a thinking judge's "<reasoning></think>" prefix, ```json fences, prose around the
    object, numeric strings/floats for the score, and a missing or inconsistent category (the
    category is always derived from the score; a disagreement is recorded as a warning).
    """
    if not isinstance(raw, str) or not raw.strip():
        raise JudgeParseError("empty judge output")
    text = raw.rsplit("</think>", 1)[-1]
    objects = [o for o in _candidate_objects(_repair_escapes(text)) if "score" in o]
    if not objects:
        objects = [o for o in _candidate_objects(text) if "score" in o]
    if not objects:
        raise JudgeParseError(f"no JSON object with a 'score' field in judge output: {raw[-300:]!r}")
    obj = objects[-1]  # the final answer if the judge drafted more than one
    warnings = []
    score = _coerce_score(obj["score"])
    category = CATEGORIES[score]
    given = obj.get("category")
    if given is None:
        warnings.append("missing category")
    elif str(given).strip().lower() != category:
        warnings.append(f"category {given!r} does not match score {score}; using {category!r}")
    out = {"score": score, "category": category}
    for key in ("evidence", "explanation"):
        value = obj.get(key, "")
        if value is None:
            value = ""
        if not isinstance(value, str):
            value = json.dumps(value) if isinstance(value, (list, dict)) else str(value)
            warnings.append(f"{key} was not a string")
        out[key] = value.strip()
    if not out["explanation"]:
        warnings.append("missing explanation")
    if score >= 2 and not out["evidence"]:
        warnings.append(f"score {score} without evidence")
    out["warnings"] = warnings
    return out


def normalize_score(score):
    """0-3 -> 0-1 (0, 1/3, 2/3, 1). None stays None."""
    return None if score is None else score / MAX_SCORE
