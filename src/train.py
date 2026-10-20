"""Minimal GRPO + LoRA training trial.

Loads the base model and a small dataset, runs a few GRPO steps with LoRA and a
simple task reward, saves the adapter + metadata, then reloads the adapter and
generates a response to confirm it works.

Usage:
    python src/train.py --config configs/trial.yaml \
        --model-path /cluster/path/to/model --dataset-path /cluster/path/to/data.jsonl

MODEL_PATH / DATASET_PATH / OUTPUT_DIR environment variables also work; CLI flags win.
"""

import argparse
import gc
import importlib.metadata
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import GRPOConfig, GRPOTrainer

REPO_ROOT = Path(__file__).resolve().parent.parent


def now():
    return datetime.now(timezone.utc).isoformat()


def load_config(args):
    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg["model"]["path"] = args.model_path or os.environ.get("MODEL_PATH") or cfg["model"]["path"]
    cfg["dataset"]["path"] = args.dataset_path or os.environ.get("DATASET_PATH") or cfg["dataset"]["path"]
    cfg["output"]["dir"] = args.output_dir or os.environ.get("OUTPUT_DIR") or cfg["output"]["dir"]
    if not cfg["model"]["path"] or not Path(cfg["model"]["path"]).exists():
        raise SystemExit(f"Model path missing or not found: {cfg['model']['path']!r}")
    if not cfg["dataset"]["path"] or not Path(cfg["dataset"]["path"]).exists():
        raise SystemExit(f"Dataset path missing or not found: {cfg['dataset']['path']!r}")
    return cfg


def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto", local_files_only=True
    )
    return model, tokenizer


def load_data(cfg):
    path = cfg["dataset"]["path"]
    ds = load_dataset("json", data_files=path, split="train")
    missing = {"prompt", "answer"} - set(ds.column_names)
    if missing:
        raise SystemExit(f"Dataset is missing fields {sorted(missing)}; found {ds.column_names}")
    ds = ds.select(range(min(len(ds), cfg["dataset"]["max_samples"])))
    # Chat-format prompts so TRL applies the model's chat template.
    return ds.map(
        lambda row: {"prompt": [{"role": "user", "content": row["prompt"]}], "answer": str(row["answer"])},
        remove_columns=ds.column_names,
    )


def completion_text(completion):
    # Chat prompts yield completions as [{"role": "assistant", "content": ...}].
    return completion[0]["content"] if isinstance(completion, list) else completion


def task_reward(prompts, completions, answer, **kwargs):
    """1.0 if the reference answer appears after the reasoning (</think>), else 0.0."""
    rewards = []
    for completion, ref in zip(completions, answer):
        final = completion_text(completion).rsplit("</think>", 1)[-1]
        rewards.append(1.0 if ref.strip().lower() in final.lower() else 0.0)
    return rewards


def generate(model, tokenizer, prompt, max_new_tokens):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "trial.yaml"))
    parser.add_argument("--model-path")
    parser.add_argument("--dataset-path")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    cfg = load_config(args)
    t, l = cfg["training"], cfg["lora"]
    output_dir = Path(cfg["output"]["dir"])
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    metadata = {
        "status": "running",
        "start_time": now(),
        "seed": t["seed"],
        "model": cfg["model"],
        "dataset": cfg["dataset"],
        "training": t,
        "lora": l,
        "output_dir": str(output_dir),
        "versions": {"python": platform.python_version()}
        | {p: importlib.metadata.version(p) for p in ["torch", "transformers", "peft", "trl", "datasets"]},
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }

    def save_metadata():
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    save_metadata()
    try:
        set_seed(t["seed"])
        print("Loading model...")
        model, tokenizer = load_model(cfg["model"]["path"])
        print("Base model check:", repr(generate(model, tokenizer, cfg["verify"]["prompt"], 32)))

        dataset = load_data(cfg)
        print(f"Loaded {len(dataset)} training examples")

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[task_reward],
            args=GRPOConfig(
                output_dir=str(output_dir / "trainer"),
                seed=t["seed"],
                max_steps=t["max_steps"],
                learning_rate=t["learning_rate"],
                per_device_train_batch_size=t["per_device_train_batch_size"],
                gradient_accumulation_steps=t["gradient_accumulation_steps"],
                num_generations=t["num_generations"],
                max_completion_length=t["max_completion_length"],
                beta=t["beta"],
                gradient_checkpointing=True,
                bf16=True,
                logging_steps=1,
                save_strategy="no",
                report_to="none",
            ),
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=LoraConfig(
                r=l["r"],
                lora_alpha=l["alpha"],
                lora_dropout=l["dropout"],
                target_modules=l["target_modules"],
                task_type="CAUSAL_LM",
            ),
        )
        print("Training...")
        result = trainer.train()

        adapter_dir = output_dir / "adapter"
        trainer.model.save_pretrained(str(adapter_dir))
        (output_dir / "metrics.json").write_text(
            json.dumps({"train": result.metrics, "log_history": trainer.state.log_history}, indent=2)
        )
        metadata["global_step"] = trainer.state.global_step
        print(f"Saved adapter after {trainer.state.global_step} step(s) to {adapter_dir}")

        # Reload from disk to confirm the saved adapter is usable on its own.
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()
        base, tokenizer = load_model(cfg["model"]["path"])
        reloaded = PeftModel.from_pretrained(base, str(adapter_dir))
        response = generate(reloaded, tokenizer, cfg["verify"]["prompt"], cfg["verify"]["max_new_tokens"])
        print("Reloaded adapter response:", repr(response))

        metadata["verification_response"] = response
        metadata["status"] = "succeeded"
    except Exception as e:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        metadata["end_time"] = now()
        save_metadata()


if __name__ == "__main__":
    main()
