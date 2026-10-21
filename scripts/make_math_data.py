"""Build train/eval JSONL from the MATH competition dataset (Hendrycks et al.).

Train: EleutherAI/hendrycks_math train split (answer = last \\boxed{} in the solution).
Eval:  HuggingFaceH4/MATH-500 (a subset of the MATH test split, so disjoint from train).
Both keep only the requested levels and answers that are integers or fractions, so
src/rewards.is_correct can grade them without a LaTeX-equivalence checker.

Usage: python scripts/make_math_data.py [--levels 4 5] [--n-train 300] [--n-eval 40] [--seed 0]
Writes data/math_train.jsonl and data/math_eval.jsonl (gitignored; regenerate with this script).
"""

import argparse
import json
import random
import sys
from pathlib import Path

from datasets import concatenate_datasets, load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rewards import last_boxed, to_number  # noqa: E402

SUBJECTS = [
    "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
    "number_theory", "prealgebra", "precalculus",
]
SUFFIX = "\n\nPut your final answer in \\boxed{}."


def numeric(answer):
    return answer is not None and to_number(answer) is not None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", nargs="+", default=["4", "5"])
    parser.add_argument("--n-train", type=int, default=300)
    parser.add_argument("--n-eval", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()
    rng = random.Random(args.seed)

    eval_rows = [
        {"prompt": r["problem"] + SUFFIX, "answer": r["answer"], "family": f"{r['subject']} L{r['level']}"}
        for r in load_dataset("HuggingFaceH4/MATH-500", split="test")
        if str(r["level"]) in args.levels and numeric(r["answer"])
    ]
    eval_problems = {row["prompt"] for row in eval_rows}

    train = concatenate_datasets([load_dataset("EleutherAI/hendrycks_math", s, split="train") for s in SUBJECTS])
    train_rows = []
    for r in train:
        level = r["level"].removeprefix("Level ")
        answer = last_boxed(r["solution"])
        prompt = r["problem"] + SUFFIX
        if level in args.levels and numeric(answer) and prompt not in eval_problems:
            train_rows.append({"prompt": prompt, "answer": answer.strip(), "family": f"{r['type']} L{level}"})

    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows, n in [("math_train.jsonl", train_rows, args.n_train), ("math_eval.jsonl", eval_rows, args.n_eval)]:
        if len(rows) < n:
            print(f"Only {len(rows)} eligible rows for {name} (asked for {n})")
        (out / name).write_text("".join(json.dumps(r) + "\n" for r in rows[:n]))
        print(f"Wrote {min(n, len(rows))} rows to {out / name}")


if __name__ == "__main__":
    main()
