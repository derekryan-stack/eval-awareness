"""Model/tokenizer, LoRA, GRPO trainer init, and the full smoke-test pipeline on a tiny OLMo 3 model (CPU)."""

import json

import pytest

from conftest import REPO
from config import load_config
import train


def tiny_cfg(model_dir, data, tmp_path, config="trial.yaml", extra=()):
    return ["--config", str(REPO / "configs" / config), "--model-path", str(model_dir), "--dataset-path", str(data),
            "--set", "training.bf16=false", "--set", "model.dtype=float32", "--set", "model.device_map=null",
            "--set", f"output.root={tmp_path / 'results'}", *extra]


def test_tokenizer_and_thinking_template(tiny_model_dir):
    from modeling import check_thinking_template, load_tokenizer, stop_token_ids

    tok = load_tokenizer(tiny_model_dir)
    assert tok.padding_side == "left" and tok.pad_token == "<|pad|>"
    assert check_thinking_template(tok)["generation_prompt_ends_with_think"]
    from transformers import GenerationConfig

    ids = stop_token_ids(tok, GenerationConfig(eos_token_id=tok.eos_token_id), ["<|im_end|>", "<|not_a_token|>"])
    assert ids == [tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")]


def test_model_loading_and_lora_targets(tiny_model_dir, clean_env):
    from modeling import ModelSetupError, build_lora_config, load_model, validate_lora_targets

    cfg = load_config(REPO / "configs" / "trial.yaml", set_overrides=["model.dtype=float32", "model.device_map=null"],
                      cli_overrides={"model.path": str(tiny_model_dir)})
    model, tok = load_model(cfg)
    assert tok.convert_tokens_to_ids("<|im_end|>") in model.generation_config.eos_token_id
    assert validate_lora_targets(model, cfg["lora"]["target_modules"]) == {m: 2 for m in ["q_proj", "k_proj", "v_proj", "o_proj"]}
    with pytest.raises(ModelSetupError, match="q_prj"):
        validate_lora_targets(model, ["q_prj"])
    lora = build_lora_config(cfg)
    assert lora.r == 16 and lora.lora_alpha == 32 and set(lora.target_modules) == {"q_proj", "k_proj", "v_proj", "o_proj"}


def test_grpo_trainer_initializes(tiny_model_dir, trial_jsonl, tmp_path, clean_env):
    from trl import GRPOTrainer

    import data
    from modeling import build_lora_config, load_model
    from rewards import RewardContext, build_rewards

    cfg = load_config(REPO / "configs" / "prob_cot_length.yaml", set_overrides=["model.dtype=float32", "model.device_map=null", "training.bf16=false"],
                      cli_overrides={"model.path": str(tiny_model_dir), "dataset.path": str(trial_jsonl)})
    model, tok = load_model(cfg)
    rows, _ = data.load_from_config(cfg, tokenizer=tok)
    funcs, weights, _ = build_rewards(cfg, RewardContext(config=cfg, tokenizer=tok))
    args = train.build_grpo_config(cfg, tmp_path / "trainer", weights)
    assert args.num_generations == 4 and args.beta == 0.01 and args.reward_weights == [0.0, 1.0]
    trainer = GRPOTrainer(model=model, reward_funcs=funcs, args=args, train_dataset=data.to_hf_dataset(rows),
                          processing_class=tok, peft_config=build_lora_config(cfg))
    assert trainer.reward_func_names == ["task_reward", "cot_length_penalty"]
    assert any("lora_" in n for n, p in trainer.model.named_parameters() if p.requires_grad)


def test_unknown_grpo_kwarg_rejected(clean_env, tmp_path):
    from config import ConfigError

    cfg = load_config(REPO / "configs" / "trial.yaml", set_overrides=["training.grpo_kwargs={not_a_field: 1}"])
    with pytest.raises(ConfigError, match="not_a_field"):
        train.build_grpo_config(cfg, tmp_path)


def test_dry_run(tiny_model_dir, trial_jsonl, tmp_path, clean_env, capsys):
    assert train.main(tiny_cfg(tiny_model_dir, trial_jsonl, tmp_path, extra=["--dry-run"])) == 0
    out = capsys.readouterr().out
    assert "Config OK" in out and "generation_prompt_ends_with_think': True" in out
    assert not (tmp_path / "results").exists()  # dry run writes nothing


def test_missing_paths_give_clean_error(tmp_path, clean_env, capsys):
    rc = train.main(["--config", str(REPO / "configs" / "trial.yaml"), "--set", f"output.root={tmp_path}", "--set", "dataset.path=null"])
    assert rc == 2 and "MODEL_PATH" in capsys.readouterr().err


def test_smoke_test_end_to_end(tiny_model_dir, trial_jsonl, tmp_path, clean_env):
    """Full pipeline: 1 GRPO step with two reward components, save adapter, reload, generate."""
    argv = tiny_cfg(tiny_model_dir, trial_jsonl, tmp_path, config="prob_cot_length.yaml", extra=["--smoke-test"])
    assert train.main(argv) == 0
    (run_dir,) = (tmp_path / "results" / "prob_cot_length-smoke").iterdir()
    meta = json.loads((run_dir / "metadata.json").read_text())
    assert meta["status"] == "succeeded" and meta["global_step"] == 1 and meta["verification_response"]
    assert meta["seed"] == 42 and meta["smoke_test"] and meta["dataset"]["train"]["sha256"]
    assert {"adapter_config.json", "adapter_model.safetensors"} <= set(meta["adapter_files"])
    assert [r["metric"] for r in meta["rewards"]] == ["task_reward", "cot_length_penalty"]
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert set(metrics["reward_components"]) >= {"rewards/task_reward/mean", "rewards/cot_length_penalty/mean"}
    assert metrics["kl"] and metrics["final"]["cot_tokens/mean"] > 0
    lines = (run_dir / "metrics.jsonl").read_text().splitlines()
    assert any("kl" in json.loads(line) for line in lines)
    assert "Reloaded adapter response" in (run_dir / "train.log").read_text()
    assert (run_dir / "config.yaml").exists()

    # Reproducible from the saved config alone (fresh run dir, nothing overwritten).
    assert train.main(["--config", str(run_dir / "config.yaml")]) == 0
    assert len(list((tmp_path / "results" / "prob_cot_length-smoke").iterdir())) == 2


def test_failed_run_is_recorded(tiny_model_dir, trial_jsonl, tmp_path, clean_env):
    argv = tiny_cfg(tiny_model_dir, trial_jsonl, tmp_path, extra=["--smoke-test", "--set", "lora.target_modules=[q_prj]"])
    assert train.main(argv) == 2
    (run_dir,) = (tmp_path / "results" / "trial-smoke").iterdir()
    meta = json.loads((run_dir / "metadata.json").read_text())
    assert meta["status"] == "failed" and "q_prj" in meta["error"] and meta["phase"] == "load_model"
