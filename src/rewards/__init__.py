"""Reward registry. Every module in this package is imported so its @register classes are available.

Add a reward: create src/rewards/<name>.py with a Reward subclass decorated @register("<name>"),
then list it under `rewards:` in a config. train.py does not change.
"""

import importlib
import inspect
import pkgutil

from grading import is_correct  # noqa: F401  (backward compatibility: `from rewards import is_correct`)
from rewards.base import REGISTRY, Reward, RewardConfigError, RewardContext, register  # noqa: F401

for _mod in pkgutil.iter_modules(__path__):
    if _mod.name != "base" and not _mod.name.startswith("_"):
        importlib.import_module(f"rewards.{_mod.name}")

RESERVED_KEYS = {"name", "weight", "label"}


def build_rewards(cfg, ctx, dataset_columns=None):
    """Instantiate the config's rewards. Returns (reward_funcs, weights, specs) for GRPOTrainer.

    specs is a JSON-serializable summary recorded in the run metadata.
    """
    funcs, weights, specs = [], [], []
    for entry in cfg["rewards"]:
        name = entry["name"]
        if name not in REGISTRY:
            raise RewardConfigError(f"Unknown reward {name!r}; registered rewards: {sorted(REGISTRY)}")
        cls = REGISTRY[name]
        params = {k: v for k, v in entry.items() if k not in RESERVED_KEYS}
        accepted = inspect.signature(cls.__init__).parameters
        unknown = [k for k in params if k not in accepted]
        if unknown and not any(p.kind is p.VAR_KEYWORD for p in accepted.values()):
            raise RewardConfigError(
                f"Reward {name!r} got unknown parameter(s) {unknown}; it accepts {[p for p in accepted if p not in ('self', 'ctx', 'label')]}"
            )
        func = cls(ctx, label=entry.get("label"), **params)
        if dataset_columns is not None:
            missing = [c for c in cls.required_columns if c not in dataset_columns]
            if missing:
                raise RewardConfigError(f"Reward {name!r} needs dataset column(s) {missing}; dataset has {sorted(dataset_columns)}")
        funcs.append(func)
        weights.append(float(entry.get("weight", 1.0)))
        specs.append({"name": name, "metric": func.__name__, "weight": weights[-1], "params": params, "class": cls.__name__})

    names = [f.__name__ for f in funcs]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise RewardConfigError(f"Duplicate reward metric names {dupes}: give repeated rewards distinct `label:` values")
    return funcs, weights, specs
