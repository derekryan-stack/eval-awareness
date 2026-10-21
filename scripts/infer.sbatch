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
#SBATCH --output=results/infer/slurm-%j.out
set -euo pipefail

ENV="${ENV:-/gpfs/radev/project/krishnaswamy_smita/gss34/envs/eval-awareness}"
MODEL_PATH="${MODEL_PATH:-weights/Olmo-3.1-32B-Think}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
nvidia-smi
"$ENV/bin/python" -u src/infer.py --model-path "$MODEL_PATH" "$@"
