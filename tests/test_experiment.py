import json

import pytest
import yaml

from conftest import REPO
from config import load_config
from experiment import RunDirError, RunRecord, make_run_dir, summarize_log_history


@pytest.fixture
def cfg(clean_env, tmp_path):
    c = load_config(REPO / "configs" / "trial.yaml")
    c["output"]["root"] = str(tmp_path / "results")
    return c


def test_run_dirs_are_unique_and_never_reused(cfg):
    a = make_run_dir(cfg, timestamp="20260101-000000")
    b = make_run_dir(cfg, timestamp="20260101-000000")
    assert a != b and a.parent == b.parent and a.parent.name == "trial"
    assert a.name == "20260101-000000_seed42" and b.name == "20260101-000000_seed42_1"


def test_slurm_job_id_in_name(cfg, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "1234")
    assert make_run_dir(cfg, timestamp="t").name == "t_seed42_job1234"


def test_explicit_output_dir_refuses_existing_run(cfg, tmp_path):
    cfg["output"]["dir"] = str(tmp_path / "exact")
    d = make_run_dir(cfg)
    (d / "metadata.json").write_text("{}")
    with pytest.raises(RunDirError, match="refusing to overwrite"):
        make_run_dir(cfg)


def test_metadata_and_config_saved(cfg):
    d = make_run_dir(cfg)
    rec = RunRecord(d, cfg)
    meta = json.loads((d / "metadata.json").read_text())
    assert meta["status"] == "running" and meta["seed"] == 42 and meta["experiment"] == "trial"
    assert meta["model"]["name"] == "OLMo-3.1-32B-Think" and "transformers" in meta["versions"] and "commit" in meta["git"]
    assert yaml.safe_load((d / "config.yaml").read_text()) == cfg  # the resolved config round-trips
    rec.finish("failed", ValueError("boom"))
    meta = json.loads((d / "metadata.json").read_text())
    assert meta["status"] == "failed" and meta["error"] == "ValueError: boom" and meta["end_time"] and "traceback" in meta


def test_summarize_log_history():
    hist = [{"step": 1, "reward": 0.5, "rewards/task_reward/mean": 0.5, "kl": 0.0},
            {"step": 2, "reward": 0.7, "rewards/task_reward/mean": 0.6, "rewards/cot_length_penalty/mean": -0.2, "kl": 0.01},
            {"step": 2, "train_runtime": 10.0}]
    s = summarize_log_history(hist)
    assert [p["value"] for p in s["kl"]] == [0.0, 0.01]
    assert set(s["reward_components"]) == {"rewards/task_reward/mean", "rewards/cot_length_penalty/mean"}
    assert s["final"]["step"] == 2 and "train_runtime" not in s["final"]
