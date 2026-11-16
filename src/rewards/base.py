"""The reward-function interface shared by every experiment.

Write a reward by subclassing Reward in its own module under src/rewards/ and decorating it with
@register("<name>"). The module is imported automatically; select it from a config with

    rewards:
      - name: <name>
        weight: 1.0        # multiplies this component in the total GRPO reward (default 1.0)
        label: <metric>    # optional: metric name, needed when the same reward is used twice
        <param>: <value>   # any other key is passed to __init__ as a keyword argument

Contract for score():
  * receives one ParsedCompletion per completion (reasoning / response / finished_thinking),
    the chat-format prompts, and every extra dataset column (e.g. answer, family) as lists
    aligned with the completions;
  * returns one float per completion, or None where the reward is undefined (e.g. a judge call
    failed). None becomes NaN, which TRL ignores in the weighted sum for that component;
  * may call self.log_metric(name, value) for extra scalars (averaged per logging step).

TRL logs each component separately as rewards/<metric_name>/mean and /std (unweighted), plus the
weighted total as `reward`; see train.py for where they are written in a run directory.
"""

import math
from dataclasses import dataclass, field
from typing import Any

from thinking import split_completion

# kwargs TRL adds to every reward call that are not dataset columns.
TRL_CALL_KWARGS = {"trainer_state", "log_extra", "log_metric", "environments", "completion_ids"}

REGISTRY = {}


class RewardConfigError(ValueError):
    pass


def register(name):
    def decorator(cls):
        if name in REGISTRY and REGISTRY[name] is not cls:
            raise RewardConfigError(f"Reward name {name!r} registered twice ({REGISTRY[name].__name__}, {cls.__name__})")
        cls.registry_name = name
        REGISTRY[name] = cls
        return cls

    return decorator


@dataclass
class RewardContext:
    """What a reward may need at construction time. Built once by train.py."""

    config: dict
    tokenizer: Any = None
    run_dir: Any = None
    extras: dict = field(default_factory=dict)

    @property
    def max_completion_length(self):
        return self.config["training"]["max_completion_length"]


class Reward:
    registry_name = None
    # Default metric name (TRL uses the callable's __name__); override per class or with `label:`.
    metric_name = None
    # Dataset columns this reward needs; checked against the dataset before training starts.
    required_columns = ()

    def __init__(self, ctx: RewardContext, label=None):
        self.ctx = ctx
        self.__name__ = label or self.metric_name or self.registry_name
        self._log_metric = None

    def score(self, completions, prompts, **columns):
        raise NotImplementedError

    def log_metric(self, name, value):
        if self._log_metric is not None:
            self._log_metric(name, float(value))

    def __call__(self, prompts, completions, **kwargs):
        log = kwargs.get("log_metric")
        columns = {k: v for k, v in kwargs.items() if k not in TRL_CALL_KWARGS}
        self._log_metric = log
        try:
            parsed = [split_completion(c) for c in completions]
            rewards = list(self.score(parsed, prompts=prompts, **columns))
        finally:
            self._log_metric = None
        if len(rewards) != len(completions):
            raise RuntimeError(f"Reward {self.__name__!r} returned {len(rewards)} values for {len(completions)} completions")
        out = [None if r is None or (isinstance(r, float) and math.isnan(r)) else float(r) for r in rewards]
        n_invalid = sum(r is None for r in out)
        if n_invalid and log is not None:
            log(f"rewards/{self.__name__}/invalid_frac", n_invalid / len(out))
        return out

    def __repr__(self):
        return f"{type(self).__name__}(name={self.__name__!r})"
