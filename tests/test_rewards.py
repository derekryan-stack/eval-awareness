import math

import pytest

from conftest import REPO
from config import load_config
from grading import is_correct, to_number
from rewards import REGISTRY, Reward, RewardConfigError, RewardContext, build_rewards, register
from rewards.base import REGISTRY as BASE_REGISTRY
from thinking import split_completion


def chat(text):
    return [{"role": "assistant", "content": text}]


class Logger:
    def __init__(self):
        self.metrics = {}

    def __call__(self, name, value):
        self.metrics[name] = value


@pytest.fixture
def ctx(tiny_tokenizer, clean_env):
    return RewardContext(config=load_config(REPO / "configs" / "trial.yaml"), tokenizer=tiny_tokenizer)


def test_split_completion():
    p = split_completion("reason</think>answer")
    assert (p.reasoning, p.response, p.finished_thinking) == ("reason", "answer", True)
    p = split_completion(chat("<think>reason</think>answer"))
    assert p.reasoning == "reason" and p.finished_thinking
    p = split_completion("still thinking")
    assert p.finished_thinking is False and p.response == ""


@pytest.mark.parametrize("response,answer,ok", [
    ("\\boxed{\\frac{11}{16}}", "11/16", True),
    ("so 0.6875", "11/16", True),
    ("The answer is 6/11.", "6/11", True),
    ("Paris.", "Paris", True),
    ("Parisian", "Paris", False),
    ("\\boxed{1/0}", "1/2", False),  # used to raise ZeroDivisionError inside the reward
])
def test_is_correct(response, answer, ok):
    assert is_correct(response, answer) is ok
    assert to_number("1/0") is None


def test_registry_contains_existing_rewards():
    assert {"task", "cot_length"} <= set(REGISTRY)


def test_task_reward_matches_preliminary_behavior(ctx):
    funcs, weights, specs = build_rewards(ctx.config, ctx)
    (task,) = funcs
    assert task.__name__ == "task_reward"  # TRL logs rewards/task_reward/mean
    out = task(prompts=[None] * 3, completions=[chat("x</think>\\boxed{42}"), chat("\\boxed{42}"), chat("x</think>41")],
               answer=["42", "42", "42"], trainer_state=None, log_metric=Logger(), log_extra=None, completion_ids=[[1]] * 3)
    assert out == [1.0, 0.0, 0.0]


def test_multiple_rewards_weights_and_component_metrics(ctx):
    ctx.config["rewards"] = [{"name": "task", "weight": 0.0}, {"name": "cot_length", "coef": 1.0, "weight": 1.0}]
    funcs, weights, specs = build_rewards(ctx.config, ctx, dataset_columns={"prompt", "answer"})
    assert [f.__name__ for f in funcs] == ["task_reward", "cot_length_penalty"]
    assert weights == [0.0, 1.0]
    assert specs[1]["params"] == {"coef": 1.0}
    log = Logger()
    out = funcs[1](prompts=[None, None], completions=[chat("a b c</think>x"), chat("never finished")], log_metric=log)
    max_len = ctx.max_completion_length
    assert out[1] == -1.0 and -1.0 < out[0] < 0
    assert "cot_tokens/mean" in log.metrics and log.metrics["cot_tokens/mean"] > max_len / 2


def test_duplicate_reward_needs_label(ctx):
    ctx.config["rewards"] = [{"name": "task"}, {"name": "task"}]
    with pytest.raises(RewardConfigError, match="label"):
        build_rewards(ctx.config, ctx)
    ctx.config["rewards"] = [{"name": "task"}, {"name": "task", "label": "task_again"}]
    funcs, _, _ = build_rewards(ctx.config, ctx)
    assert funcs[1].__name__ == "task_again"


def test_unknown_reward_and_params(ctx):
    ctx.config["rewards"] = [{"name": "nope"}]
    with pytest.raises(RewardConfigError, match="registered rewards"):
        build_rewards(ctx.config, ctx)
    ctx.config["rewards"] = [{"name": "cot_length", "coeff": 1}]
    with pytest.raises(RewardConfigError, match="unknown parameter"):
        build_rewards(ctx.config, ctx)


def test_required_columns_checked(ctx):
    with pytest.raises(RewardConfigError, match="answer"):
        build_rewards(ctx.config, ctx, dataset_columns={"prompt"})


def test_new_reward_plugs_in_without_touching_trainer(ctx):
    @register("_test_len")
    class LenReward(Reward):
        def __init__(self, ctx, label=None, scale=1.0):
            super().__init__(ctx, label=label)
            self.scale = scale

        def score(self, completions, prompts, **columns):
            self.log_metric("_test/extra", 3)
            return [None if not c.finished_thinking else self.scale * len(c.response) for c in completions]

    try:
        ctx.config["rewards"] = [{"name": "_test_len", "scale": 2.0, "weight": 0.5}]
        funcs, weights, _ = build_rewards(ctx.config, ctx)
        log = Logger()
        out = funcs[0](prompts=[0, 0], completions=["r</think>abc", "unfinished"], log_metric=log, family=["a", "b"])
        assert out == [6.0, None] and weights == [0.5]
        assert log.metrics == {"_test/extra": 3.0, "rewards/_test_len/invalid_frac": 0.5}
    finally:
        BASE_REGISTRY.pop("_test_len")


def test_wrong_length_output_is_an_error(ctx):
    class Bad(Reward):
        registry_name = "bad"

        def score(self, completions, prompts, **columns):
            return [1.0]

    with pytest.raises(RuntimeError, match="returned 1 values"):
        Bad(ctx)(prompts=[0, 0], completions=["a", "b"])


def test_nan_becomes_none(ctx):
    class NanReward(Reward):
        registry_name = "nan"

        def score(self, completions, prompts, **columns):
            return [math.nan, 1]

    assert NanReward(ctx)(prompts=[0, 0], completions=["a", "b"]) == [None, 1.0]
