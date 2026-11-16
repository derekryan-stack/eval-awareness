"""Run directories, metadata, log capture, and incremental metric logging for one experiment run.

Layout of a run directory:
    config.yaml              resolved config (rerun with: python src/train.py --config <run>/config.yaml)
    metadata.json            status, seed, model/dataset identifiers, versions, git commit, timings, errors
    train.log                everything printed to stdout/stderr during the run
    metrics.jsonl            one line per logging step (written as training runs, survives crashes)
    metrics.json             final summary: train metrics, reward components, KL, full log history
    dropped_examples.jsonl   examples rejected by dataset validation (only if any)
    adapter/                 LoRA adapter (adapter_config.json + adapter_model.safetensors)

metadata.json "status" is one of: running, succeeded, failed, interrupted. A run completed
successfully iff status == "succeeded" (which also requires the adapter to reload and generate).
"""

import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from config import REPO_ROOT

PACKAGES = ["torch", "transformers", "peft", "trl", "datasets", "accelerate"]


def now():
    return datetime.now(timezone.utc).isoformat()


class RunDirError(RuntimeError):
    pass


def resolve_path(p):
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def make_run_dir(cfg, timestamp=None):
    """Create a fresh, unique run directory. Never reuses a directory that already holds a run."""
    out = cfg["output"]
    if out.get("dir"):
        run_dir = resolve_path(out["dir"])
        if run_dir.exists() and any(run_dir.iterdir()):
            raise RunDirError(f"Output directory {run_dir} already exists and is not empty; refusing to overwrite an earlier run")
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    stamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}_seed{cfg['training']['seed']}"
    if os.environ.get("SLURM_JOB_ID"):
        name += f"_job{os.environ['SLURM_JOB_ID']}"
    parent = resolve_path(out["root"]) / cfg["experiment"]["name"]
    parent.mkdir(parents=True, exist_ok=True)
    for i in range(1000):
        candidate = parent / (name if i == 0 else f"{name}_{i}")
        try:
            candidate.mkdir()  # atomic: two jobs starting in the same second get different dirs
            return candidate
        except FileExistsError:
            continue
    raise RunDirError(f"Could not create a unique run directory under {parent}")


def file_sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info():
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return None

    commit = run("rev-parse", "HEAD")
    return {"commit": commit, "branch": run("rev-parse", "--abbrev-ref", "HEAD"), "dirty": bool(run("status", "--porcelain")) if commit else None}


def versions():
    v = {"python": platform.python_version()}
    for p in PACKAGES:
        try:
            v[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            v[p] = None
    return v


def dataset_identifier(path, report=None):
    p = Path(path)
    info = {"path": str(p.resolve()), "sha256": file_sha256(p) if p.is_file() else None}
    if report is not None:
        info.update(report.summary())
    return info


class RunRecord:
    """Owns metadata.json and config.yaml for a run; save() is cheap and called after each phase."""

    def __init__(self, run_dir, cfg):
        self.run_dir = Path(run_dir)
        self.cfg = cfg
        self.metadata = {
            "experiment": cfg["experiment"]["name"],
            "status": "running",
            "phase": "setup",
            "start_time": now(),
            "end_time": None,
            "seed": cfg["training"]["seed"],
            "smoke_test": bool(cfg["experiment"].get("smoke_test")),
            "run_dir": str(self.run_dir),
            "config_file": cfg["experiment"].get("config_file"),
            "model": {"name": cfg["model"]["name"], "path": cfg["model"]["path"]},
            "dataset": {"path": cfg["dataset"]["path"]},
            "git": git_info(),
            "versions": versions(),
            "host": socket.gethostname(),
            "slurm": {k: os.environ[k] for k in ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_PARTITION", "SLURM_JOB_NODELIST", "CUDA_VISIBLE_DEVICES") if k in os.environ},
            "command": sys.argv,
        }
        (self.run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        self.save()

    def update(self, **fields):
        self.metadata.update(fields)
        self.save()

    def phase(self, name):
        print(f"[{now()}] phase: {name}", flush=True)
        self.update(phase=name)

    def save(self):
        tmp = self.run_dir / "metadata.json.tmp"
        tmp.write_text(json.dumps(self.metadata, indent=2, default=str))
        tmp.replace(self.run_dir / "metadata.json")  # atomic: never a half-written file

    def finish(self, status, error=None):
        self.metadata.update(status=status, end_time=now())
        if error is not None:
            import traceback

            self.metadata["error"] = f"{type(error).__name__}: {error}"
            self.metadata["traceback"] = "".join(traceback.format_exception(error))[-8000:]
        start = datetime.fromisoformat(self.metadata["start_time"])
        self.metadata["duration_seconds"] = round((datetime.now(timezone.utc) - start).total_seconds(), 1)
        self.save()


class Tee:
    """Duplicate a stream into a log file."""

    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, data):
        self.stream.write(data)
        self.fh.write(data)
        self.fh.flush()
        return len(data)

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self.stream, name)


def capture_logs(run_dir):
    """Tee stdout/stderr (and Python logging) into <run_dir>/train.log. Returns a restore() callable."""
    import logging

    fh = open(Path(run_dir) / "train.log", "a", buffering=1)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = Tee(old_out, fh), Tee(old_err, fh)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    if root.level > logging.INFO or root.level == logging.NOTSET:
        root.setLevel(logging.INFO)

    def restore():
        root.removeHandler(handler)
        sys.stdout, sys.stderr = old_out, old_err
        fh.close()

    return restore


def summarize_log_history(log_history):
    """Split TRL's log history into reward components, KL, and other metrics for metrics.json."""
    steps = [e for e in log_history if "step" in e and "train_runtime" not in e]
    components = sorted({k for e in steps for k in e if k.startswith("rewards/")})
    def series(key):
        return [{"step": e["step"], "value": e[key]} for e in steps if key in e]
    return {
        "reward_total": series("reward"),
        "reward_components": {k: series(k) for k in components},
        "kl": series("kl"),
        "final": steps[-1] if steps else {},
    }


def make_metrics_callback(run_dir):
    """A transformers TrainerCallback that appends each log entry to metrics.jsonl as it happens."""
    from transformers import TrainerCallback

    path = Path(run_dir) / "metrics.jsonl"

    class JsonlMetrics(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and state.is_world_process_zero:
                with path.open("a") as f:
                    f.write(json.dumps({"step": state.global_step, "time": now(), **logs}, default=float) + "\n")

    return JsonlMetrics()
