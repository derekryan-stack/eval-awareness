"""Rudimentary checks for the six experimental conditions: configs, new rewards, judge_fast config."""

import pytest

from conftest import REPO
from config import load_config
from judge.judge import Judgment, load_judge_config
from rewards import REGISTRY, RewardConfigError, RewardContext, build_rewards
from rewards.awareness import AwarenessPenalty, AwarenessReward
from rewards.formatting import FormattingReward

CONDITIONS = {
    "task_only": ["task"],
    "awareness_penalty": ["task", "awareness_penalty"],
    "awareness_reward": ["task", "awareness_reward"],
    "length_penalty": ["task", "cot_length"],
    "preference": ["task", "preference"],
    "formatting": ["task", "formatting"],
}


@pytest.fixture
def ctx(clean_env):
    return RewardContext(config=load_config(REPO / "configs" / "trial.yaml"))


@pytest.mark.parametrize("cond,expected", CONDITIONS.items())
def test_condition_config_loads_and_rewards_registered(cond, expected, clean_env):
    cfg = load_config(REPO / "configs" / f"prob_{cond}.yaml")
    names = [r["name"] for r in cfg["rewards"]]
    assert names == expected
    assert all(n in REGISTRY for n in names)
    assert cfg["experiment"]["name"] == f"prob_{cond}"


def test_conditions_differ_only_in_rewards(clean_env):
    base = load_config(REPO / "configs" / "prob_task_only.yaml")
    for cond in CONDITIONS:
        cfg = load_config(REPO / "configs" / f"prob_{cond}.yaml")
        for section in ("training", "lora", "dataset", "model"):
            assert cfg[section] == base[section], (cond, section)


def test_formatting_reward(ctx):
    f = FormattingReward(ctx, max_response_chars=20)
    out = f(prompts=[None] * 5, completions=[
        "r</think>\\boxed{1}",             # ok
        "r</think>no box",                 # no boxed
        "r</think>\\boxed{1} \\boxed{2}",  # two boxed
        "r</think>\\boxed{1} and a long tail of text",  # too long
        "never finished \\boxed{1}",       # unfinished
    ])
    assert out == [1.0, 0.0, 0.0, 0.5, 0.0]


class StubJudge:
    def __init__(self, results):
        self.results = results

    def judge_batch(self, items):
        assert all({"prompt", "reasoning", "response", "finished_thinking"} <= set(i) for i in items)
        return self.results


def _j(score):
    if score is None:
        return Judgment(score=None, category=None, valid=False)
    return Judgment(score=score, category="x", normalized=score / 3, valid=True)


def test_awareness_penalty_and_reward_signs_and_invalid(ctx):
    judge = StubJudge([_j(3), _j(0), _j(None)])
    kw = dict(prompts=["q"] * 3, completions=["a</think>b"] * 3)
    pen = AwarenessPenalty(ctx, judge=judge)
    rew = AwarenessReward(ctx, judge=judge)
    assert pen(**kw) == [-1.0, 0.0, None]
    assert rew(**kw) == [1.0, 0.0, None]
    assert pen.__name__ == "awareness_penalty" and rew.__name__ == "awareness_reward"


def test_awareness_with_mock_backend_via_registry(ctx, monkeypatch, tmp_path):
    monkeypatch.setenv("JUDGE_BACKEND", "mock")
    monkeypatch.setenv("JUDGE_CACHE_PATH", str(tmp_path / "cache.sqlite"))
    ctx.config["rewards"] = [{"name": "awareness_penalty", "weight": 1.0, "judge_config": "configs/judge.yaml"}]
    funcs, weights, _ = build_rewards(ctx.config, ctx)
    out = funcs[0](prompts=["q"], completions=["thinking</think>answer"])
    assert out == [-0.0] or out == [0.0]  # mock judge returns score 0


def test_judge_fast_config(monkeypatch):
    for v in ("JUDGE_BACKEND", "JUDGE_MODEL_PATH", "JUDGE_MODEL", "JUDGE_CACHE_PATH"):
        monkeypatch.delenv(v, raising=False)
    cfg = load_judge_config(REPO / "configs" / "judge_fast.yaml")
    assert cfg["backend"]["type"] == "hf"          # offline: no API backend
    assert cfg["backend"]["max_new_tokens"] <= 512
    assert cfg["cache_path"] != load_judge_config(REPO / "configs" / "judge.yaml")["cache_path"]


def test_preference_requires_model_path(ctx, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    monkeypatch.delenv("PREFERENCE_RM_PATH", raising=False)
    ctx.config["rewards"] = [{"name": "preference"}]
    with pytest.raises(RewardConfigError, match="PREFERENCE_RM_PATH"):
        build_rewards(ctx.config, ctx)
