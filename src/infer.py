"""Prompt the base model (optionally with a LoRA adapter) and show its reasoning and answer.

The chat template opens the assistant turn with <think>, so each completion is the
chain of thought, then </think>, then the final response. They are printed separately.

Usage:
    python src/infer.py --model-path weights/Olmo-3.1-32B-Think --prompts-file prompts/probability.txt
    python src/infer.py --model-path ... "What is 2 + 2?" "Name a prime over 100."
    python src/infer.py --model-path ...              # interactive: one prompt per line, Ctrl-D to quit

MODEL_PATH also works; the CLI flag wins. Results are appended to --output (JSONL).
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_model(model_path, adapter_path=None):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto", local_files_only=True
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompts, args):
    """Generate one completion per prompt in a single left-padded batch."""
    texts = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    sampling = {"do_sample": True, "temperature": args.temperature, "top_p": args.top_p} if args.temperature > 0 else {"do_sample": False}
    start = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.pad_token_id, **sampling)
    elapsed = time.time() - start

    results = []
    for prompt, ids in zip(prompts, out[:, inputs["input_ids"].shape[1]:]):
        ids = ids[ids != tokenizer.pad_token_id]
        text = tokenizer.decode(ids, skip_special_tokens=True)
        finished = "</think>" in text
        reasoning, _, response = text.partition("</think>")
        results.append({
            "prompt": prompt,
            "reasoning": reasoning.strip(),
            "response": response.strip(),
            "finished_thinking": finished,
            "num_tokens": len(ids),
            "seconds": round(elapsed, 1),
        })
    return results


def show(result):
    bar = "=" * 80
    print(f"\n{bar}\nPROMPT: {result['prompt']}\n{bar}")
    print(f"--- REASONING ({result['num_tokens']} tokens total, {result['seconds']}s for batch) ---")
    print(result["reasoning"])
    print("--- RESPONSE ---")
    print(result["response"] if result["finished_thinking"] else "[no </think>: hit max_new_tokens while still reasoning]")
    print(flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompts", nargs="*", help="Prompts to run; omit for interactive mode")
    parser.add_argument("--model-path")
    parser.add_argument("--adapter-path", help="Optional LoRA adapter to load on top of the base model")
    parser.add_argument("--prompts-file", help="Text file with one prompt per line")
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.6, help="0 for greedy decoding")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(REPO_ROOT / "results" / "infer" / "generations.jsonl"))
    args = parser.parse_args()

    model_path = args.model_path or os.environ.get("MODEL_PATH")
    if not model_path or not Path(model_path).exists():
        raise SystemExit(f"Model path missing or not found: {model_path!r}")
    prompts = list(args.prompts)
    if args.prompts_file:
        prompts += [line.strip() for line in Path(args.prompts_file).read_text().splitlines() if line.strip()]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    print(f"Loading model from {model_path}...", flush=True)
    model, tokenizer = load_model(model_path, args.adapter_path)
    print(f"Loaded on {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}", flush=True)

    def run(batch):
        results = generate(model, tokenizer, batch, args)
        with output.open("a") as f:
            for r in results:
                show(r)
                record = r | {
                    "time": datetime.now(timezone.utc).isoformat(),
                    "model_path": model_path,
                    "adapter_path": args.adapter_path,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_new_tokens": args.max_new_tokens,
                    "seed": args.seed,
                }
                f.write(json.dumps(record) + "\n")

    if prompts:
        run(prompts)
    else:
        print("Interactive mode: enter a prompt per line, Ctrl-D to quit.", flush=True)
        while True:
            try:
                prompt = input("\n> ").strip()
            except EOFError:
                break
            if prompt:
                run([prompt])
    print(f"Saved generations to {output}")


if __name__ == "__main__":
    main()
