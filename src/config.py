"""Experiment configuration: YAML + defaults + environment overrides + --set overrides, then validation.

Precedence (highest first):
    1. CLI flags (--model-path, --dataset-path, --output-dir, --seed) and --set section.key=value
    2. Environment variables (machine-specific paths; see ENV_OVERRIDES)
    3. The YAML config
    4. DEFAULTS below

The resolved config is a plain dict that is saved verbatim to <run_dir>/config.yaml, so a run
can be reproduced from that file alone:  python src/train.py --config <run_dir>/config.yaml
(with output.dir cleared so it gets a fresh run directory).
"""

import copy
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULTS = {
    "experiment": {
        "name": None,  # defaults to the config file's stem
        "description": "",
    },
    "model": {
        "name": "OLMo-3.1-32B-Think",  # identifier recorded in metadata
        "path": None,  # local checkpoint directory (MODEL_PATH)
        "dtype": "bfloat16",
        "device_map": "auto",
        "attn_implementation": None,  # e.g. "sdpa" / "flash_attention_2"; None = transformers default
        # Extra stop strings' token ids: generation also stops at <|im_end|> (end of a chat turn),
        # not only at <|endoftext|>, which is the only eos in OLMo's generation_config.json.
        "extra_eos_tokens": ["<|im_end|>"],
    },
    "dataset": {
        "path": None,  # training JSONL (DATASET_PATH)
        "eval_path": None,  # optional held-out JSONL (EVAL_DATASET_PATH), validated but not used by GRPO
        "prompt_field": "prompt",
        "answer_field": "answer",
        "system_prompt": None,  # None keeps the model's default system prompt from the chat template
        "max_samples": None,  # take the first N valid examples
    },
    "training": {
        "seed": 42,
        "max_steps": 3,
        "learning_rate": 1.0e-5,
        "lr_scheduler_type": "linear",
        "warmup_steps": 0,
        # Counted in completions, not prompts: prompts per step = batch * grad_accum * n_gpus / num_generations.
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "num_generations": 2,
        # TRL 1.16 has no max_prompt_length; longer prompts are dropped (and logged) at load time.
        "max_prompt_length": 1024,
        "max_completion_length": 2048,
        "temperature": 1.0,
        "top_p": 1.0,
        "beta": 0.01,  # KL coefficient; TRL only logs `kl` when beta > 0
        "bf16": True,
        "gradient_checkpointing": True,
        "deterministic": False,  # True = transformers.enable_full_determinism (slower)
        "grpo_kwargs": {},  # any other GRPOConfig field, passed through verbatim (checked against GRPOConfig)
    },
    "rewards": [{"name": "task"}],
    "lora": {
        "r": 16,
        "alpha": 32,
        "dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
    "logging": {
        "logging_steps": 1,
        "report_to": "none",  # "wandb" / "tensorboard" if installed
        "log_completions": False,  # TRL prints sample completions each logging step
        "save_steps": None,  # intermediate checkpoints every N steps (None = only the final adapter)
    },
    "verify": {
        "prompt": "What is 2 + 2? Answer with a single number.",
        "max_new_tokens": 64,
    },
    "output": {
        "root": "results",  # runs go to <root>/<experiment.name>/<timestamp>_seed<seed>[_job<id>]
        "dir": None,  # exact run directory (OUTPUT_DIR); must not already contain a run
    },
}

# Environment variable -> config key. Only machine-specific values belong here.
ENV_OVERRIDES = {
    "MODEL_PATH": "model.path",
    "DATASET_PATH": "dataset.path",
    "EVAL_DATASET_PATH": "dataset.eval_path",
    "OUTPUT_ROOT": "output.root",
    "OUTPUT_DIR": "output.dir",
    "EXPERIMENT_NAME": "experiment.name",
    "SEED": "training.seed",
}

# Old configs (pre-Phase 1) had output.dir as a fixed directory that each run overwrote. It is now
# treated as the experiment root, so old configs still work without overwriting earlier runs.
_LEGACY_OUTPUT_DIR_IS_ROOT = True


class ConfigError(ValueError):
    pass


def deep_merge(base, override):
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def set_dotted(cfg, dotted, value):
    keys = dotted.split(".")
    node = cfg
    for key in keys[:-1]:
        if not isinstance(node.get(key), dict):
            raise ConfigError(f"Cannot set {dotted!r}: {key!r} is not a config section")
        node = node[key]
    node[keys[-1]] = value


def get_dotted(cfg, dotted):
    node = cfg
    for key in dotted.split("."):
        node = node[key]
    return node


def resolve_input_path(p):
    """Relative input paths: the current directory if the file is there, else the repo root. Stored absolute."""
    if not p:
        return p
    path = Path(os.path.expanduser(str(p)))
    if path.is_absolute():
        return str(path)
    return str((Path.cwd() / path) if (Path.cwd() / path).exists() else (REPO_ROOT / path))


def parse_set_overrides(pairs):
    """['training.learning_rate=1e-5', 'lora.r=8'] -> {...}; values are parsed as YAML scalars/lists."""
    parsed = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ConfigError(f"--set expects section.key=value, got {pair!r}")
        key, raw = pair.split("=", 1)
        parsed[key.strip()] = yaml.safe_load(raw)
    return parsed


def load_config(path, env=None, set_overrides=None, cli_overrides=None):
    """Load and resolve a config. Does not touch the filesystem beyond reading `path`.

    env: mapping used for ENV_OVERRIDES (defaults to os.environ).
    set_overrides: list of 'a.b=value' strings.
    cli_overrides: dict of dotted key -> value (None values are ignored).
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a YAML mapping")
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise ConfigError(f"Unknown top-level config sections in {path}: {sorted(unknown)}; expected {sorted(DEFAULTS)}")

    # Legacy: output.dir in a YAML file meant "the run directory" and was overwritten on every run.
    if _LEGACY_OUTPUT_DIR_IS_ROOT and (raw.get("output") or {}).get("dir") and "root" not in raw["output"]:
        raw["output"] = {"root": raw["output"]["dir"], "dir": None}

    cfg = deep_merge(DEFAULTS, raw)
    if raw.get("rewards") is not None:
        cfg["rewards"] = copy.deepcopy(raw["rewards"])  # lists replace, not merge

    env = os.environ if env is None else env
    for var, dotted in ENV_OVERRIDES.items():
        if env.get(var):
            value = env[var]
            set_dotted(cfg, dotted, int(value) if dotted == "training.seed" else value)
    for dotted, value in parse_set_overrides(set_overrides).items():
        set_dotted(cfg, dotted, value)
    for dotted, value in (cli_overrides or {}).items():
        if value is not None:
            set_dotted(cfg, dotted, value)

    for dotted in ("model.path", "dataset.path", "dataset.eval_path"):
        set_dotted(cfg, dotted, resolve_input_path(get_dotted(cfg, dotted)))
    if cfg["experiment"]["name"] is None:
        cfg["experiment"]["name"] = path.stem
    cfg["experiment"]["config_file"] = str(path)
    validate_config(cfg)
    return cfg


def apply_smoke_test(cfg):
    """Shrink a config to one tiny GRPO step: validates the pipeline, not the science."""
    cfg = copy.deepcopy(cfg)
    t = cfg["training"]
    t.update(
        max_steps=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        num_generations=2,
        max_completion_length=min(t["max_completion_length"], 32),
        warmup_steps=0,
    )
    t["grpo_kwargs"] = {k: v for k, v in t.get("grpo_kwargs", {}).items() if k not in ("steps_per_generation", "generation_batch_size")}
    cfg["dataset"]["max_samples"] = 4
    cfg["verify"]["max_new_tokens"] = min(cfg["verify"]["max_new_tokens"], 16)
    cfg["logging"]["save_steps"] = None
    cfg["experiment"]["name"] += "-smoke"
    cfg["experiment"]["smoke_test"] = True
    validate_config(cfg)
    return cfg


# PyYAML (YAML 1.1) reads 1e-5 (no decimal point) as a string; accept it for numeric fields.
FLOAT_KEYS = ["training.learning_rate", "training.beta", "training.temperature", "training.top_p", "lora.alpha", "lora.dropout"]


def _coerce_float(value, where):
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            raise ConfigError(f"{where} must be a number, got {value!r}") from None
    return value


def _coerce_numbers(cfg):
    for dotted in FLOAT_KEYS:
        section, key = dotted.split(".")
        cfg[section][key] = _coerce_float(cfg[section][key], dotted)
    for entry in cfg["rewards"] if isinstance(cfg["rewards"], list) else []:
        if isinstance(entry, dict) and "weight" in entry:
            entry["weight"] = _coerce_float(entry["weight"], f"rewards[{entry.get('name')}].weight")


def _require(cond, msg):
    if not cond:
        raise ConfigError(msg)


def validate_config(cfg):
    """Catch errors before a 10-minute model load rather than after."""
    _coerce_numbers(cfg)
    t, l = cfg["training"], cfg["lora"]
    for key in ["seed", "max_steps", "per_device_train_batch_size", "gradient_accumulation_steps",
                "num_generations", "max_prompt_length", "max_completion_length"]:
        _require(isinstance(t[key], int) and not isinstance(t[key], bool), f"training.{key} must be an integer, got {t[key]!r}")
    for key in ["max_steps", "per_device_train_batch_size", "gradient_accumulation_steps", "max_prompt_length", "max_completion_length"]:
        _require(t[key] >= 1, f"training.{key} must be >= 1, got {t[key]}")
    _require(t["num_generations"] >= 2, "training.num_generations must be >= 2 (GRPO normalizes rewards within a group)")
    _require(isinstance(t["learning_rate"], (int, float)) and t["learning_rate"] > 0, "training.learning_rate must be > 0")
    _require(isinstance(t["beta"], (int, float)) and t["beta"] >= 0, "training.beta (KL coefficient) must be >= 0")
    _require(t["temperature"] > 0, "training.temperature must be > 0 (GRPO needs sampled completions)")

    # TRL: generation_batch_size = per_device_bs * num_processes * steps_per_generation (default
    # grad_accum) must be divisible by num_generations. Checked here for one process; with more
    # GPUs under accelerate the product only grows, so this is the binding case.
    world = int(os.environ.get("WORLD_SIZE", "1"))
    steps_per_gen = t["grpo_kwargs"].get("steps_per_generation") or t["gradient_accumulation_steps"]
    gen_batch = t["per_device_train_batch_size"] * world * steps_per_gen
    _require(
        gen_batch % t["num_generations"] == 0,
        f"per_device_train_batch_size ({t['per_device_train_batch_size']}) x processes ({world}) x "
        f"steps_per_generation ({steps_per_gen}) = {gen_batch} must be divisible by num_generations "
        f"({t['num_generations']}). Note the batch size counts completions, not prompts.",
    )
    _require(isinstance(t["grpo_kwargs"], dict), "training.grpo_kwargs must be a mapping")

    _require(isinstance(l["r"], int) and l["r"] > 0, "lora.r must be a positive integer")
    _require(l["alpha"] > 0, "lora.alpha must be > 0")
    _require(0 <= l["dropout"] < 1, "lora.dropout must be in [0, 1)")
    tm = l["target_modules"]
    _require(tm == "all-linear" or (isinstance(tm, list) and tm and all(isinstance(m, str) for m in tm)),
             "lora.target_modules must be a non-empty list of module names or 'all-linear'")

    rewards = cfg["rewards"]
    _require(isinstance(rewards, list) and rewards, "rewards must be a non-empty list of {name: ..., weight: ...}")
    for entry in rewards:
        _require(isinstance(entry, dict) and isinstance(entry.get("name"), str), f"Each reward needs a 'name': {entry!r}")
        w = entry.get("weight", 1.0)
        _require(isinstance(w, (int, float)) and not isinstance(w, bool), f"Reward {entry['name']!r} weight must be a number")
    _require(any(e.get("weight", 1.0) != 0 for e in rewards), "All reward weights are 0: there is no training signal")

    d = cfg["dataset"]
    _require(d["max_samples"] is None or (isinstance(d["max_samples"], int) and d["max_samples"] >= 1),
             "dataset.max_samples must be null or a positive integer")
    _require(cfg["logging"]["logging_steps"] >= 1, "logging.logging_steps must be >= 1")
    _require(cfg["model"]["dtype"] in ("bfloat16", "float16", "float32"), "model.dtype must be bfloat16, float16 or float32")
    return cfg


def check_paths(cfg, need_model=True, need_dataset=True):
    """Raise ConfigError with an actionable message if required paths are missing."""
    problems = []
    if need_model:
        p = cfg["model"]["path"]
        if not p:
            problems.append("model.path is not set: export MODEL_PATH=/path/to/Olmo-3.1-32B-Think (or --model-path)")
        elif not Path(p).is_dir():
            problems.append(f"model.path {p!r} is not a directory")
        elif not (Path(p) / "config.json").exists():
            problems.append(f"model.path {p!r} has no config.json; is this a Hugging Face checkpoint directory?")
    if need_dataset:
        p = cfg["dataset"]["path"]
        if not p:
            problems.append("dataset.path is not set: export DATASET_PATH=/path/to/train.jsonl (or --dataset-path)")
        elif not Path(p).is_file():
            hint = " (generate it with scripts/make_prob_data.py)" if "prob_" in str(p) else ""
            problems.append(f"dataset.path {p!r} does not exist{hint}")
        ep = cfg["dataset"]["eval_path"]
        if ep and not Path(ep).is_file():
            problems.append(f"dataset.eval_path {ep!r} does not exist")
    if problems:
        raise ConfigError("Missing or invalid paths:\n  - " + "\n  - ".join(problems))
