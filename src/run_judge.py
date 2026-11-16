"""Run the evaluation-awareness judge on sample completions, independently of training.

Input JSONL rows need "prompt" plus either "completion" (raw "<reasoning></think><answer>" text,
as produced by training) or "reasoning" + "response" (as written by src/evaluate.py). An optional
"expected_score" (0-3) turns the run into a reliability check: the summary reports exact and
within-one agreement and a confusion matrix.

Usage:
    python src/run_judge.py --input data/judge_samples.jsonl                     # backend from configs/judge.yaml
    JUDGE_BACKEND=hf JUDGE_MODEL_PATH=weights/Olmo-3.1-32B-Think python src/run_judge.py --input ...
    python src/run_judge.py --input results/eval/<job>/base.jsonl --output results/judge/base_judged.jsonl
    python src/run_judge.py --input data/judge_samples.jsonl --backend mock     # pipeline check, no model

Writes one line per input to --output: the input fields plus "judgment" (score, category,
evidence, explanation, normalized, valid, error, warnings, raw, cached).
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import REPO_ROOT  # noqa: E402
from judge import CATEGORIES, build_judge, load_judge_config  # noqa: E402


def read_items(path, limit=None):
    items = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "prompt" not in row or not ("completion" in row or "reasoning" in row or "response" in row):
                raise SystemExit(f"{path}:{n}: rows need 'prompt' and 'completion' (or 'reasoning'/'response')")
            row.setdefault("id", n)
            items.append(row)
            if limit and len(items) >= limit:
                break
    if not items:
        raise SystemExit(f"{path} has no rows")
    return items


def agreement(rows):
    labelled = [(r["expected_score"], r["judgment"]["score"]) for r in rows if r.get("expected_score") is not None and r["judgment"]["valid"]]
    if not labelled:
        return None
    exact = sum(e == s for e, s in labelled) / len(labelled)
    within1 = sum(abs(e - s) <= 1 for e, s in labelled) / len(labelled)
    matrix = {e: {s: 0 for s in CATEGORIES} for e in CATEGORIES}
    for e, s in labelled:
        matrix[e][s] += 1
    disagreements = [{"id": r["id"], "expected": r["expected_score"], "judge": r["judgment"]["score"], "explanation": r["judgment"]["explanation"]}
                     for r in rows if r.get("expected_score") is not None and r["judgment"]["valid"] and r["expected_score"] != r["judgment"]["score"]]
    return {"n_labelled": len(labelled), "exact_agreement": exact, "within_one_agreement": within1, "confusion_matrix_expected_x_judge": matrix, "disagreements": disagreements}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--output", help="Default: results/judge/<input stem>_judged.jsonl")
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "judge.yaml"))
    p.add_argument("--backend", help="Override backend type: hf, anthropic, mock")
    p.add_argument("--model-path", help="hf backend checkpoint (or JUDGE_MODEL_PATH)")
    p.add_argument("--model", help="anthropic backend model id (or JUDGE_MODEL)")
    p.add_argument("--target", choices=["full", "reasoning", "response"])
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--batch-size", type=int, default=16, help="Items sent to the backend per call")
    args = p.parse_args(argv)

    cfg = load_judge_config(args.config, overrides={"backend.type": args.backend, "backend.model_path": args.model_path,
                                                    "backend.model": args.model, "target": args.target})
    if args.no_cache:
        cfg["cache_path"] = None
    out_path = Path(args.output) if args.output else REPO_ROOT / "results" / "judge" / f"{Path(args.input).stem}_judged.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cfg.get("log_path"):
        cfg["log_path"] = str(out_path.with_suffix(".log.jsonl"))
    judge = build_judge(cfg)
    items = read_items(args.input, args.limit)
    print(f"Judging {len(items)} items with backend {judge.backend.signature()} (cache: {cfg.get('cache_path')})", flush=True)

    rows = []
    with out_path.open("w") as f:
        for i in range(0, len(items), args.batch_size):
            batch = items[i:i + args.batch_size]
            for item, j in zip(batch, judge.judge_batch(batch)):
                row = {**item, "judgment": {**j.to_output(), "normalized": j.normalized, "valid": j.valid, "error": j.error,
                                            "warnings": j.warnings, "raw": j.raw, "cached": j.cached}}
                rows.append(row)
                f.write(json.dumps(row) + "\n")
                exp = f" expected={item['expected_score']}" if item.get("expected_score") is not None else ""
                status = f"score={j.score} ({j.category})" if j.valid else f"INVALID: {j.error}"
                print(f"[{item['id']}] {status}{exp}{' [cached]' if j.cached else ''}", flush=True)

    valid = [r for r in rows if r["judgment"]["valid"]]
    summary = {
        "n": len(rows),
        "valid": len(valid),
        "invalid": len(rows) - len(valid),
        "score_distribution": dict(sorted(Counter(r["judgment"]["score"] for r in valid).items())),
        "mean_normalized": sum(r["judgment"]["normalized"] for r in valid) / len(valid) if valid else None,
        "stats": judge.stats,
        "agreement": agreement(rows),
        "output": str(out_path),
    }
    (out_path.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    judge.cache.close()
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
