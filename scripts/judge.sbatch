#!/usr/bin/env bash
# Run the evaluation-awareness judge (src/run_judge.py) with the local hf backend on one H200.
# Usage (from the repo root):
#   JUDGE_MODEL_PATH=weights/Olmo-3.1-32B-Think sbatch scripts/judge.sbatch --input data/judge_samples.jsonl
#   sbatch scripts/judge.sbatch --input results/eval/job<id>/base.jsonl --output results/judge/base_judged.jsonl
#SBATCH --job-name=awareness-judge
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm-%x-%j.out
set -euo pipefail

ENV="${ENV:-/gpfs/radev/project/krishnaswamy_smita/gss34/envs/eval-awareness}"
export JUDGE_BACKEND="${JUDGE_BACKEND:-hf}"
export JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-weights/Olmo-3.1-32B-Think}"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
[ -x "$ENV/bin/python" ] || { echo "ERROR: no Python at $ENV/bin/python; set ENV" >&2; exit 1; }
if [ "$JUDGE_BACKEND" = hf ] && [ ! -d "$JUDGE_MODEL_PATH" ]; then
  echo "ERROR: judge model $JUDGE_MODEL_PATH not found; set JUDGE_MODEL_PATH" >&2; exit 1
fi
"$ENV/bin/python" -u src/run_judge.py "$@"
