"""Shared GRPO + LoRA training entry point for every experiment.

An experiment is a config file (configs/*.yaml) plus reward modules (src/rewards/*.py). This
script contains no experiment-specific logic: it loads the config, model, tokenizer, dataset and
rewards, trains, saves the adapter and metadata into a fresh run directory, then reloads the
adapter and generates a response to confirm it works.

Usage:
    python src/train.py --config configs/trial.yaml                      # paths from MODEL_PATH / DATASET_PATH
    python src/train.py --config configs/trial.yaml --smoke-test         # 1 tiny step + save/reload/generate
    python src/train.py --config configs/trial.yaml --dry-run            # validate config/data/rewards, no model weights
    python src/train.py --config configs/trial.yaml --set training.learning_rate=5e-6 --seed 7

See src/config.py for precedence of CLI flags, environment variables, and the YAML file.
"""

import argparse
import dataclasses
import gc
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import REPO_ROOT, ConfigError, apply_smoke_test, check_paths, load_config  # noqa: E402


class Terminated(BaseException):
    """Raised on SIGTERM (SLURM timeout / scancel) so the run is recorded as interrupted."""


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "trial.yaml"))
    p.add_argument("--model-path")
    p.add_argument("--dataset-path")
    p.add_argument("--eval-dataset-path")
    p.add_argument("--output-dir", help="Exact run directory (must not already contain a run)")
    p.add_argument("--seed", type=int)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Override any config value, e.g. lora.r=8")
    p.add_argument("--smoke-test", action="store_true", help="One tiny GRPO step, then save, reload and generate")
    p.add_argument("--dry-run", action="store_true", help="Validate config, data and rewards without loading model weights")
    return p.parse_args(argv)


def resolve_config(args):
    cfg = load_config(
        args.config,
        set_overrides=args.set,
        cli_overrides={
            "model.path": args.model_path,
            "dataset.path": args.dataset_path,
            "dataset.eval_path": args.eval_dataset_path,
            "output.dir": args.output_dir,
            "training.seed": args.seed,
        },
    )
    return apply_smoke_test(cfg) if args.smoke_test else cfg


def build_grpo_config(cfg, output_dir, reward_weights=None):
    from trl import GRPOConfig

    t, lg = cfg["training"], cfg["logging"]
    known = {f.name for f in dataclasses.fields(GRPOConfig)}
    extra = dict(t.get("grpo_kwargs") or {})
    unknown = sorted(set(extra) - known)
    if unknown:
        raise ConfigError(f"training.grpo_kwargs has fields GRPOConfig (installed TRL) does not accept: {unknown}")
    kwargs = dict(
        output_dir=str(output_dir),
        seed=t["seed"],
        data_seed=t["seed"],
        max_steps=t["max_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_steps=t["warmup_steps"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        num_generations=t["num_generations"],
        max_completion_length=t["max_completion_length"],
        temperature=t["temperature"],
        top_p=t["top_p"],
        beta=t["beta"],
        reward_weights=reward_weights or [float(e.get("weight", 1.0)) for e in cfg["rewards"]],
        gradient_checkpointing=t["gradient_checkpointing"],
        bf16=t["bf16"],
        logging_steps=lg["logging_steps"],
        save_strategy="steps" if lg.get("save_steps") else "no",
        report_to=lg["report_to"],
        log_completions=lg["log_completions"],
    )
    if lg.get("save_steps"):
        kwargs["save_steps"] = lg["save_steps"]
    kwargs.update(extra)
    return GRPOConfig(**kwargs)


def dry_run(cfg):
    """Everything except loading weights: config, paths, tokenizer/template, dataset schema, rewards."""
    from data import load_from_config
    from modeling import check_thinking_template, load_tokenizer
    from rewards import RewardContext, build_rewards

    have_model = bool(cfg["model"]["path"]) and Path(cfg["model"]["path"]).is_dir()
    check_paths(cfg, need_model=False, need_dataset=True)
    tokenizer = load_tokenizer(cfg["model"]["path"]) if have_model else None
    rows, reports = load_from_config(cfg, tokenizer=tokenizer)
    _, _, specs = build_rewards(cfg, RewardContext(config=cfg, tokenizer=tokenizer), dataset_columns=set(rows[0]))
    print(f"Config OK: experiment={cfg['experiment']['name']} seed={cfg['training']['seed']}")
    print(f"Dataset: {reports['train'].summary()}")
    if "eval" in reports:
        print(f"Eval dataset: {reports['eval'].summary()}")
    print(f"Rewards: {specs}")
    if tokenizer is not None:
        print(f"Chat template: {check_thinking_template(tokenizer)}")
    else:
        print("Model path not available: skipped tokenizer, chat-template and prompt-length checks")
    return 0


def train(cfg):
    import torch
    from transformers import set_seed
    from transformers.trainer_utils import enable_full_determinism

    import data as data_mod
    from experiment import RunRecord, capture_logs, dataset_identifier, make_metrics_callback, make_run_dir, summarize_log_history
    from modeling import build_lora_config, check_thinking_template, describe_checkpoint, generate, load_model, load_tokenizer, validate_lora_targets
    from rewards import RewardContext, build_rewards

    check_paths(cfg)
    run_dir = make_run_dir(cfg)
    restore_logs = capture_logs(run_dir)
    record = RunRecord(run_dir, cfg)
    print(f"Run directory: {run_dir}", flush=True)

    def on_sigterm(signum, _frame):
        raise Terminated(f"received signal {signum}")

    old_handler = signal.signal(signal.SIGTERM, on_sigterm)
    t = cfg["training"]
    try:
        (enable_full_determinism if t["deterministic"] else set_seed)(t["seed"])
        record.update(gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

        record.phase("load_data")
        tokenizer = load_tokenizer(cfg["model"]["path"])
        template = check_thinking_template(tokenizer)
        if not template["generation_prompt_ends_with_think"]:
            print(f"WARNING: generation prompt does not open <think>; reasoning parsing assumes it does: {template}", flush=True)
        rows, reports = data_mod.load_from_config(cfg, tokenizer=tokenizer, run_dir=run_dir)
        dataset = data_mod.to_hf_dataset(rows)
        record.update(
            chat_template=template,
            dataset={"train": dataset_identifier(cfg["dataset"]["path"], reports["train"]),
                     **({"eval": dataset_identifier(cfg["dataset"]["eval_path"], reports["eval"])} if "eval" in reports else {})},
        )
        print(f"Loaded {len(dataset)} training examples ({len(reports['train'].dropped)} dropped)", flush=True)

        ctx = RewardContext(config=cfg, tokenizer=tokenizer, run_dir=run_dir)
        reward_funcs, reward_weights, specs = build_rewards(cfg, ctx, dataset_columns=set(dataset.column_names))
        record.update(rewards=specs)

        record.phase("load_model")
        model, tokenizer = load_model(cfg, tokenizer)
        record.update(model={**describe_checkpoint(cfg["model"]["path"]), "name": cfg["model"]["name"],
                             "stop_token_ids": model.generation_config.eos_token_id},
                      lora_target_matches=validate_lora_targets(model, cfg["lora"]["target_modules"]))
        base_check = generate(model, tokenizer, [cfg["verify"]["prompt"]], max_new_tokens=32)[0]
        print("Base model check:", repr(base_check), flush=True)
        record.update(base_check_response=base_check)

        record.phase("init_trainer")
        from trl import GRPOTrainer

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_funcs,
            args=build_grpo_config(cfg, run_dir / "trainer", reward_weights),
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=build_lora_config(cfg),
            callbacks=[make_metrics_callback(run_dir)],
        )
        trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in trainer.model.parameters())
        record.update(trainable_parameters=trainable, total_parameters=total)
        print(f"Trainable parameters: {trainable:,} / {total:,}", flush=True)

        record.phase("train")
        result = trainer.train()
        record.update(global_step=trainer.state.global_step, train_metrics=result.metrics)

        record.phase("save_adapter")
        adapter_dir = run_dir / "adapter"
        trainer.model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        files = sorted(p.name for p in adapter_dir.iterdir())
        if "adapter_config.json" not in files or not any(f.startswith("adapter_model") for f in files):
            raise RuntimeError(f"Adapter save incomplete in {adapter_dir}: {files}")
        import json

        (run_dir / "metrics.json").write_text(json.dumps(
            {"train": result.metrics, **summarize_log_history(trainer.state.log_history), "log_history": trainer.state.log_history},
            indent=2, default=float))
        record.update(adapter_dir=str(adapter_dir), adapter_files=files)
        print(f"Saved adapter after {trainer.state.global_step} step(s) to {adapter_dir}", flush=True)

        record.phase("verify_adapter")
        # Free the training copy before reloading so a 32B model fits twice in sequence on one GPU.
        trainer.accelerator.free_memory()
        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        from peft import PeftModel

        base, tokenizer = load_model(cfg)
        reloaded = PeftModel.from_pretrained(base, str(adapter_dir))
        reloaded.eval()
        response = generate(reloaded, tokenizer, [cfg["verify"]["prompt"]], max_new_tokens=cfg["verify"]["max_new_tokens"])[0]
        print("Reloaded adapter response:", repr(response), flush=True)
        if not response.strip():
            raise RuntimeError("Reloaded adapter generated an empty response")
        record.update(verification_response=response)
        record.finish("succeeded")
        print(f"Run succeeded: {run_dir}", flush=True)
        return run_dir
    except (KeyboardInterrupt, Terminated) as e:
        record.finish("interrupted", e)
        raise
    except BaseException as e:
        record.finish("failed", e)
        raise
    finally:
        signal.signal(signal.SIGTERM, old_handler)
        restore_logs()


def main(argv=None):
    from data import DataError
    from experiment import RunDirError
    from modeling import ModelSetupError
    from rewards import RewardConfigError

    args = parse_args(argv)
    try:
        cfg = resolve_config(args)
        if args.dry_run:
            return dry_run(cfg)
        train(cfg)
        return 0
    except (ConfigError, DataError, RewardConfigError, RunDirError, ModelSetupError) as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
