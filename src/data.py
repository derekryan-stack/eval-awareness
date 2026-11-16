"""Dataset loading and validation for GRPO training and evaluation.

Input: JSONL, one object per line, with a prompt field and an answer field (names configurable).
Extra fields (e.g. "family") are kept and passed to reward functions as keyword arguments.

Invalid examples are never dropped silently: each is reported with its line number and reason,
and written to <run_dir>/dropped_examples.jsonl when a run directory is given.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


class DataError(ValueError):
    pass


@dataclass
class LoadReport:
    path: str
    total_lines: int = 0
    kept: int = 0
    dropped: list = field(default_factory=list)  # [{"line": n, "reason": ..., "row": ...}]

    def summary(self):
        reasons = {}
        for d in self.dropped:
            key = d["reason"].split(":")[0]
            reasons[key] = reasons.get(key, 0) + 1
        return {"path": self.path, "total_lines": self.total_lines, "kept": self.kept, "dropped": len(self.dropped), "dropped_by_reason": reasons}


def read_jsonl(path):
    """Yield (line_number, parsed object or exception) for every non-blank line."""
    path = Path(path)
    if not path.is_file():
        raise DataError(f"Dataset file not found: {path}")
    with path.open() as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield i, json.loads(line)
            except json.JSONDecodeError as e:
                yield i, e


def build_messages(prompt, system_prompt=None):
    messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
    return messages + [{"role": "user", "content": prompt}]


def prompt_token_count(tokenizer, messages):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def load_examples(path, prompt_field="prompt", answer_field="answer", system_prompt=None, max_samples=None,
                  tokenizer=None, max_prompt_length=None, require_answer=True):
    """Validate and normalize a JSONL dataset.

    Returns (rows, report). Each row is {"prompt": [chat messages], "answer": str, **extra columns}.
    Raises DataError if the file is missing, empty, or has no valid examples.
    """
    report = LoadReport(path=str(path))
    rows = []

    def drop(line, reason, row):
        report.dropped.append({"line": line, "reason": reason, "row": row if isinstance(row, dict) else repr(row)})

    for line, obj in read_jsonl(path):
        report.total_lines += 1
        if isinstance(obj, Exception):
            drop(line, f"invalid_json: {obj}", None)
            continue
        if not isinstance(obj, dict):
            drop(line, f"not_an_object: got {type(obj).__name__}", obj)
            continue
        prompt = obj.get(prompt_field)
        if not isinstance(prompt, str):
            drop(line, f"missing_prompt: field {prompt_field!r} is {'absent' if prompt_field not in obj else type(prompt).__name__}", obj)
            continue
        if not prompt.strip():
            drop(line, "empty_prompt", obj)
            continue
        answer = obj.get(answer_field)
        if require_answer:
            if answer is None or isinstance(answer, (dict, list, bool)):
                drop(line, f"missing_answer: field {answer_field!r} is {'absent' if answer_field not in obj else type(answer).__name__}", obj)
                continue
            if not str(answer).strip():
                drop(line, "empty_answer", obj)
                continue
        messages = build_messages(prompt, system_prompt)
        if tokenizer is not None and max_prompt_length:
            n = prompt_token_count(tokenizer, messages)
            if n > max_prompt_length:
                drop(line, f"prompt_too_long: {n} tokens > max_prompt_length {max_prompt_length}", obj)
                continue
        extras = {k: v for k, v in obj.items() if k not in (prompt_field, answer_field, "prompt", "answer")}
        rows.append({"prompt": messages, "answer": "" if answer is None else str(answer).strip(), **extras})
        if max_samples and len(rows) >= max_samples:
            break

    report.kept = len(rows)
    for d in report.dropped:
        log.warning("Dropped %s line %d: %s", path, d["line"], d["reason"])
    if report.total_lines == 0:
        raise DataError(f"Dataset {path} is empty")
    if not rows:
        raise DataError(f"Dataset {path} has no valid examples ({len(report.dropped)} dropped): {report.summary()['dropped_by_reason']}")
    _check_consistent_columns(rows, path)
    return rows, report


def _check_consistent_columns(rows, path):
    """datasets.Dataset.from_list needs every row to have the same keys; fill gaps with None and warn."""
    keys = set().union(*(r.keys() for r in rows))
    for k in sorted(keys):
        n_missing = sum(k not in r for r in rows)
        if n_missing:
            log.warning("%s: column %r missing in %d/%d rows; filled with null", path, k, n_missing, len(rows))
            for r in rows:
                r.setdefault(k, None)


def to_hf_dataset(rows):
    from datasets import Dataset

    return Dataset.from_list(rows)


def write_drop_log(report, out_path):
    with Path(out_path).open("a") as f:
        for d in report.dropped:
            f.write(json.dumps({"path": report.path, **d}, default=str) + "\n")


def load_from_config(cfg, tokenizer=None, run_dir=None):
    """Load the training set (and validate the eval set if configured). Returns (train_rows, reports)."""
    d, t = cfg["dataset"], cfg["training"]
    common = dict(prompt_field=d["prompt_field"], answer_field=d["answer_field"], system_prompt=d["system_prompt"],
                  tokenizer=tokenizer, max_prompt_length=t["max_prompt_length"])
    train, train_report = load_examples(d["path"], max_samples=d["max_samples"], **common)
    reports = {"train": train_report}
    if d.get("eval_path"):
        _, reports["eval"] = load_examples(d["eval_path"], **common)
    if run_dir is not None:
        for r in reports.values():
            if r.dropped:
                write_drop_log(r, Path(run_dir) / "dropped_examples.jsonl")
    return train, reports
