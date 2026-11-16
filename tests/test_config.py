import pytest
import yaml

from conftest import REPO
from config import ConfigError, apply_smoke_test, check_paths, load_config

CONFIGS = sorted((REPO / "configs").glob("*.yaml"))
TRAIN_CONFIGS = [c for c in CONFIGS if c.name != "judge.yaml"]


@pytest.mark.parametrize("path", TRAIN_CONFIGS, ids=lambda p: p.name)
def test_all_repo_training_configs_load(path, clean_env):
    cfg = load_config(path)
    assert cfg["experiment"]["name"]
    assert cfg["rewards"]


def test_trial_config_values(clean_env):
    cfg = load_config(REPO / "configs" / "trial.yaml")
    assert cfg["experiment"]["name"] == "trial"
    assert cfg["training"]["seed"] == 42
    assert cfg["training"]["max_completion_length"] == 3072
    assert cfg["lora"]["target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert cfg["model"]["path"] is None
    assert cfg["dataset"]["path"] == str(REPO / "data" / "prob_train.jsonl")  # repo-relative, stored absolute


def test_env_overrides(clean_env, monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", "/models/olmo")
    monkeypatch.setenv("DATASET_PATH", "/data/train.jsonl")
    monkeypatch.setenv("SEED", "7")
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path))
    cfg = load_config(REPO / "configs" / "trial.yaml")
    assert cfg["model"]["path"] == "/models/olmo"
    assert cfg["dataset"]["path"] == "/data/train.jsonl"
    assert cfg["training"]["seed"] == 7
    assert cfg["output"]["root"] == str(tmp_path)


def test_precedence_cli_over_set_over_env(clean_env, monkeypatch):
    monkeypatch.setenv("SEED", "7")
    cfg = load_config(REPO / "configs" / "trial.yaml", set_overrides=["training.seed=8", "lora.r=4"], cli_overrides={"training.seed": 9})
    assert cfg["training"]["seed"] == 9
    assert cfg["lora"]["r"] == 4


def test_set_parses_yaml_values(clean_env):
    cfg = load_config(REPO / "configs" / "trial.yaml", set_overrides=["training.learning_rate=5e-6", "lora.target_modules=[q_proj, v_proj]"])
    assert cfg["training"]["learning_rate"] == 5e-6
    assert cfg["lora"]["target_modules"] == ["q_proj", "v_proj"]


def write(tmp_path, data):
    p = tmp_path / "exp.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


def test_batch_divisibility_rejected(clean_env, tmp_path):
    p = write(tmp_path, {"training": {"per_device_train_batch_size": 3, "num_generations": 2}})
    with pytest.raises(ConfigError, match="divisible by num_generations"):
        load_config(p)


def test_batch_divisibility_accounts_for_grad_accum(clean_env, tmp_path):
    p = write(tmp_path, {"training": {"per_device_train_batch_size": 2, "gradient_accumulation_steps": 2, "num_generations": 4}})
    assert load_config(p)["training"]["num_generations"] == 4


@pytest.mark.parametrize("training", [{"num_generations": 1}, {"learning_rate": 0}, {"beta": -1}, {"max_steps": "10"}, {"temperature": 0}])
def test_invalid_training_values(clean_env, tmp_path, training):
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, {"training": training}))


def test_unknown_section_rejected(clean_env, tmp_path):
    with pytest.raises(ConfigError, match="Unknown top-level"):
        load_config(write(tmp_path, {"trainig": {}}))


def test_all_zero_weights_rejected(clean_env, tmp_path):
    with pytest.raises(ConfigError, match="no training signal"):
        load_config(write(tmp_path, {"rewards": [{"name": "task", "weight": 0}]}))


def test_legacy_output_dir_becomes_root(clean_env, tmp_path):
    cfg = load_config(write(tmp_path, {"output": {"dir": "results/old_style"}}))
    assert cfg["output"]["root"] == "results/old_style"
    assert cfg["output"]["dir"] is None


def test_smoke_test_shrinks_config(clean_env):
    cfg = apply_smoke_test(load_config(REPO / "configs" / "prob_cot_length.yaml"))
    t = cfg["training"]
    assert t["max_steps"] == 1 and t["num_generations"] == 2 and t["max_completion_length"] <= 32
    assert cfg["experiment"]["name"].endswith("-smoke")
    assert cfg["rewards"] == load_config(REPO / "configs" / "prob_cot_length.yaml")["rewards"]


def test_check_paths_messages(clean_env, tmp_path):
    cfg = load_config(REPO / "configs" / "trial.yaml")
    cfg["dataset"]["path"] = None
    with pytest.raises(ConfigError, match="MODEL_PATH") as e:
        check_paths(cfg)
    assert "DATASET_PATH" in str(e.value)
    cfg["model"]["path"] = str(tmp_path)  # exists but no config.json
    cfg["dataset"]["path"] = str(tmp_path / "data" / "prob_train.jsonl")
    with pytest.raises(ConfigError, match="config.json") as e:
        check_paths(cfg)
    assert "make_prob_data.py" in str(e.value)
