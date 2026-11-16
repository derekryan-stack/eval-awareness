#!/usr/bin/env bash
# Launch one GRPO + LoRA experiment on Misha. Submit from the repo root:
#
#   CONFIG=configs/trial.yaml SEED=42 sbatch scripts/submit_experiment.sh
#   CONFIG=configs/trial.yaml SMOKE_TEST=1 sbatch --time=01:00:00 scripts/submit_experiment.sh
#   CONFIG=configs/prob_cot_length.yaml sbatch --time=06:00:00 scripts/submit_experiment.sh --set training.max_steps=50
#   CONFIG=configs/trial.yaml DRY_RUN=1 bash scripts/submit_experiment.sh     # validate on the login node, no GPU
#
# Environment variables (sbatch exports your shell environment by default):
#   CONFIG        experiment config (default configs/trial.yaml)
#   SEED          overrides training.seed
#   ENV           conda env prefix (default: the shared Misha env below)
#   MODEL_PATH    base checkpoint dir (default weights/Olmo-3.1-32B-Think)
#   DATASET_PATH  training JSONL (default: dataset.path in the config)
#   OUTPUT_ROOT   parent of run directories (default results); OUTPUT_DIR sets an exact run dir
#   SMOKE_TEST=1  one tiny step + save/reload/generate;  DRY_RUN=1  validate only, no model weights
# Arguments after the script name are passed to src/train.py (e.g. --set key=value).
#
# Resources are sbatch flags, so override them on the command line rather than editing this file:
#   sbatch --gres=gpu:h200:2 --mem=320G --time=12:00:00 scripts/submit_experiment.sh
# The model is sharded across all visible GPUs with device_map="auto" (one process).
#
# Logs: results/slurm-<job name>-<job id>.out (scheduler) and <run dir>/train.log (training).
#SBATCH --job-name=grpo-exp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm-%x-%j.out
set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }

# Under sbatch the script runs from a spool copy; SLURM_SUBMIT_DIR is where sbatch was called.
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
[ -f src/train.py ] || die "run sbatch from the eval-awareness repo root (no src/train.py in $REPO)"

ENV="${ENV:-/gpfs/radev/project/krishnaswamy_smita/gss34/envs/eval-awareness}"
PY="$ENV/bin/python"
[ -x "$PY" ] || die "no Python at $PY; set ENV to the conda env prefix (see README Setup)"

CONFIG="${CONFIG:-configs/trial.yaml}"
[ -f "$CONFIG" ] || die "config $CONFIG not found"
export MODEL_PATH="${MODEL_PATH:-weights/Olmo-3.1-32B-Think}"
if [ -z "${DRY_RUN:-}" ]; then
  [ -d "$MODEL_PATH" ] || die "model checkpoint $MODEL_PATH not found; set MODEL_PATH"
fi
[ -z "${SEED:-}" ] || [[ "$SEED" =~ ^[0-9]+$ ]] || die "SEED must be an integer, got '$SEED'"

# Compute nodes may have no internet; everything is loaded from local files.
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

# Regenerate the deterministic probability dataset if the config needs it and it is missing.
DATA=$("$PY" - "$CONFIG" <<'EOF'
import sys
sys.path.insert(0, "src")
from config import ConfigError, load_config
try:
    print(load_config(sys.argv[1])["dataset"]["path"] or "")
except ConfigError as e:
    sys.exit(f"Config error: {e}")
EOF
) || die "could not load $CONFIG"
if [ -n "$DATA" ] && [ ! -f "$DATA" ]; then
  case "$(basename "$DATA")" in
    prob_train.jsonl|prob_eval.jsonl) echo "Generating $DATA"; "$PY" scripts/make_prob_data.py --out-dir "$(dirname "$DATA")" ;;
    *) die "dataset $DATA not found; set DATASET_PATH or generate it (scripts/make_math_data.py for MATH)" ;;
  esac
fi

ARGS=(--config "$CONFIG")
[ -n "${SEED:-}" ] && ARGS+=(--seed "$SEED")
[ -n "${SMOKE_TEST:-}" ] && ARGS+=(--smoke-test)
[ -n "${DRY_RUN:-}" ] && ARGS+=(--dry-run)

echo "=== $(date) job=${SLURM_JOB_ID:-local} host=$(hostname) config=$CONFIG seed=${SEED:-config} model=$MODEL_PATH"
if [ -z "${DRY_RUN:-}" ]; then
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv
  "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'CUDA not available in this job (check --gres)')"
fi
"$PY" -u src/train.py "${ARGS[@]}" "$@"
echo "=== Done $(date)"
