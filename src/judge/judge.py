"""The evaluation-awareness judge: build prompt -> backend -> parse/validate -> cache -> log.

Usable on its own (src/run_judge.py) and, in later phases, from a reward module, e.g.

    judge = build_judge(load_judge_config("configs/judge.yaml"))
    results = judge.judge_batch([{"prompt": p, "completion": c} for p, c in ...])
    rewards = [r.normalized for r in results]       # 0..1, or None if the judge failed
"""

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from judge.backends import JudgeBackendError, build_backend
from judge.cache import JudgeCache, NullCache, cache_key
from judge.parsing import JudgeParseError, normalize_score, parse_judgment
from judge.rubric import PROMPT_VERSION, build_messages
from thinking import split_completion


@dataclass
class Judgment:
    score: int | None
    category: str | None
    evidence: str = ""
    explanation: str = ""
    normalized: float | None = None
    valid: bool = False
    error: str | None = None
    warnings: list = field(default_factory=list)
    raw: str | None = None
    cached: bool = False
    attempts: int = 0
    key: str | None = None

    def to_output(self):
        """The four-field judge result."""
        return {"score": self.score, "category": self.category, "evidence": self.evidence, "explanation": self.explanation}


class Judge:
    def __init__(self, backend, cache=None, log_path=None, max_attempts=2, target="full", include_prompt=True):
        self.backend = backend
        self.cache = cache if cache is not None else NullCache()
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_attempts = max(1, int(max_attempts))
        self.target = target
        self.include_prompt = include_prompt
        self.stats = {"calls": 0, "cache_hits": 0, "parse_failures": 0, "backend_errors": 0}

    def messages_for(self, item):
        """item: {"prompt": str, "completion": str} or {"prompt", "reasoning", "response"[, "finished_thinking"]}."""
        prompt = item.get("prompt", "")
        if isinstance(prompt, list):  # chat-format prompt: show the user turn(s)
            prompt = "\n\n".join(m["content"] for m in prompt if m.get("role") == "user")
        if "completion" in item:
            parsed = split_completion(item["completion"])
            reasoning, response, finished = parsed.reasoning, parsed.response, parsed.finished_thinking
        else:
            reasoning, response = item.get("reasoning", ""), item.get("response", "")
            finished = item.get("finished_thinking", True)
        return build_messages(prompt, reasoning, response, finished, target=self.target, include_prompt=self.include_prompt)

    def judge_batch(self, items):
        msgs = [self.messages_for(it) for it in items]
        sig = self.backend.signature()
        keys = [cache_key(sig, m) for m in msgs]
        results = [None] * len(items)
        pending = []
        for i, key in enumerate(keys):
            hit = self.cache.get(key)
            if hit is not None:
                self.stats["cache_hits"] += 1
                results[i] = Judgment(**{**hit, "cached": True, "key": key})
            else:
                pending.append(i)

        attempt = 0
        errors = {i: None for i in pending}
        raws = {i: None for i in pending}
        while pending and attempt < self.max_attempts:
            attempt += 1
            started = time.time()
            try:
                outputs = self.backend.complete([msgs[i] for i in pending])
                self.stats["calls"] += len(pending)
            except Exception as e:  # backend failure: record and retry the whole pending set
                self.stats["backend_errors"] += 1
                for i in pending:
                    errors[i] = f"backend_error: {type(e).__name__}: {e}"
                continue
            latency = (time.time() - started) / max(1, len(pending))
            still = []
            for i, raw in zip(pending, outputs):
                raws[i] = raw
                try:
                    parsed = parse_judgment(raw)
                except JudgeParseError as e:
                    self.stats["parse_failures"] += 1
                    errors[i] = f"parse_error: {e}"
                    still.append(i)
                    continue
                j = Judgment(score=parsed["score"], category=parsed["category"], evidence=parsed["evidence"],
                             explanation=parsed["explanation"], normalized=normalize_score(parsed["score"]), valid=True,
                             warnings=parsed["warnings"], raw=raw, attempts=attempt, key=keys[i])
                self.cache.put(keys[i], {k: v for k, v in asdict(j).items() if k not in ("cached", "key")})
                results[i] = j
                self._log(items[i], j, latency)
            pending = still

        for i in pending:
            j = Judgment(score=None, category=None, valid=False, error=errors[i], raw=raws[i], attempts=attempt, key=keys[i])
            results[i] = j
            self._log(items[i], j, None)
        for i, j in enumerate(results):
            if j.cached:
                self._log(items[i], j, 0.0)
        return results

    def judge(self, prompt, completion):
        return self.judge_batch([{"prompt": prompt, "completion": completion}])[0]

    def _log(self, item, judgment, latency):
        if not self.log_path:
            return
        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "prompt_version": PROMPT_VERSION,
            "backend": self.backend.signature(),
            "id": item.get("id"),
            "latency_s": None if latency is None else round(latency, 3),
            **asdict(judgment),
        }
        with self.log_path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")


def load_judge_config(path=None, env=None, overrides=None):
    """Judge config: YAML `judge:` section with env overrides for machine-specific values."""
    import os

    import yaml

    env = os.environ if env is None else env
    cfg = {"backend": {"type": "mock"}, "cache_path": "results/judge/cache.sqlite", "log_path": None,
           "max_attempts": 2, "target": "full", "include_prompt": True}
    if path:
        loaded = yaml.safe_load(Path(path).read_text()) or {}
        loaded = loaded.get("judge", loaded)
        cfg.update({k: v for k, v in loaded.items() if k != "backend"})
        cfg["backend"] = {**cfg["backend"], **(loaded.get("backend") or {})}
    for var, key in [("JUDGE_BACKEND", "type"), ("JUDGE_MODEL_PATH", "model_path"), ("JUDGE_MODEL", "model")]:
        if env.get(var):
            cfg["backend"][key] = env[var]
    if env.get("JUDGE_CACHE_PATH"):
        cfg["cache_path"] = env["JUDGE_CACHE_PATH"]
    for k, v in (overrides or {}).items():
        if v is not None:
            if k.startswith("backend."):
                cfg["backend"][k.split(".", 1)[1]] = v
            else:
                cfg[k] = v
    if cfg["target"] not in ("full", "reasoning", "response"):
        raise ValueError(f"judge target must be full, reasoning or response; got {cfg['target']!r}")
    return cfg


def build_judge(cfg):
    from config import REPO_ROOT

    def resolve(p):
        p = Path(p)
        return p if p.is_absolute() else REPO_ROOT / p

    try:
        backend = build_backend(cfg["backend"])
    except TypeError as e:
        raise JudgeBackendError(f"Bad judge backend settings {cfg['backend']}: {e}") from e
    cache = JudgeCache(resolve(cfg["cache_path"])) if cfg.get("cache_path") else NullCache()
    return Judge(backend, cache=cache, log_path=resolve(cfg["log_path"]) if cfg.get("log_path") else None,
                 max_attempts=cfg["max_attempts"], target=cfg["target"], include_prompt=cfg["include_prompt"])
