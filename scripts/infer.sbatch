#!/usr/bin/env bash
# Run src/infer.py on one H200.
# Usage: sbatch scripts/infer.sbatch [infer.py args...]
#   e.g. sbatch scripts/infer.sbatch --prompts-file prompts/probability.txt
#SBATCH --job-name=olmo-infer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm-infer-%j.out
set -euo pipefail

ENV="${ENV:-/gpfs/radev/project/krishnaswamy_smita/gss34/envs/eval-awareness}"
MODEL_PATH="${MODEL_PATH:-weights/Olmo-3.1-32B-Think}"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
nvidia-smi
"$ENV/bin/python" -u src/infer.py --model-path "$MODEL_PATH" "$@"
