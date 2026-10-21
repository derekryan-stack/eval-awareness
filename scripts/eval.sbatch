#!/usr/bin/env bash
# Run src/evaluate.py on one H200.
# Usage: sbatch scripts/eval.sbatch --label base [--adapter-path results/<run>/adapter] [evaluate.py args...]
#SBATCH --job-name=olmo-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=results/eval/slurm-%j.out
set -euo pipefail

ENV="${ENV:-/gpfs/radev/project/krishnaswamy_smita/gss34/envs/eval-awareness}"
MODEL_PATH="${MODEL_PATH:-weights/Olmo-3.1-32B-Think}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
"$ENV/bin/python" -u src/evaluate.py --model-path "$MODEL_PATH" "$@"
