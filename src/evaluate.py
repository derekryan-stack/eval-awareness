"""Score a model (base or base + LoRA adapter) on a JSONL eval set with prompt/answer fields.

Writes one record per sample to <output-dir>/<label>.jsonl with reasoning token count,
whether thinking finished, and correctness. Summarize several labels with src/analyze.py.

Usage:
    python src/evaluate.py --model-path weights/Olmo-3.1-32B-Think --eval-file data/prob_eval.jsonl --label base
    python src/evaluate.py ... --adapter-path results/prob_cot_length/adapter --label cot_length
"""

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
from transformers import set_seed

from infer import generate, load_model
from rewards import is_correct

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path")
    parser.add_argument("--adapter-path")
    parser.add_argument("--eval-file", default=str(REPO_ROOT / "data" / "prob_eval.jsonl"))
    parser.add_argument("--label", required=True, help="Name for this model in the results, e.g. base")
    parser.add_argument("--samples", type=int, default=2, help="Completions per prompt")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "eval"))
    args = parser.parse_args()

    model_path = args.model_path or os.environ.get("MODEL_PATH")
    if not model_path or not Path(model_path).exists():
        raise SystemExit(f"Model path missing or not found: {model_path!r}")
    rows = list(load_dataset("json", data_files=args.eval_file, split="train"))
    if not rows or not {"prompt", "answer"} <= set(rows[0]):
        raise SystemExit(f"{args.eval_file} needs 'prompt' and 'answer' fields")
    output = Path(args.output_dir) / f"{args.label}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("")

    set_seed(args.seed)
    print(f"Loading model from {model_path} (adapter: {args.adapter_path})...", flush=True)
    model, tokenizer = load_model(model_path, args.adapter_path)

    jobs = [(i, s) for i in range(len(rows)) for s in range(args.samples)]
    for start in range(0, len(jobs), args.batch_size):
        batch = jobs[start:start + args.batch_size]
        results = generate(model, tokenizer, [rows[i]["prompt"] for i, _ in batch], args)
        with output.open("a") as f:
            for (i, s), r in zip(batch, results):
                row = rows[i]
                record = {
                    "label": args.label,
                    "adapter_path": args.adapter_path,
                    "prompt_index": i,
                    "sample": s,
                    "family": row.get("family"),
                    "answer": str(row["answer"]),
                    "reasoning_tokens": len(tokenizer(r["reasoning"], add_special_tokens=False)["input_ids"]),
                    "completion_tokens": r["num_tokens"],
                    "finished_thinking": r["finished_thinking"],
                    "correct": r["finished_thinking"] and is_correct(r["response"], row["answer"]),
                } | {k: r[k] for k in ["prompt", "reasoning", "response"]}
                f.write(json.dumps(record) + "\n")
        done = start + len(batch)
        print(f"[{args.label}] {done}/{len(jobs)} completions", flush=True)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
