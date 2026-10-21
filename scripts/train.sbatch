#!/usr/bin/env bash
# Run the GRPO + LoRA trial (scripts/train.sh) on one H200.
# Usage: sbatch scripts/train.sbatch
#   MODEL_PATH, DATASET_PATH, CONFIG, OUTPUT_DIR, ENV can be overridden: sbatch --export=ALL,OUTPUT_DIR=results/trial2 ...
#SBATCH --job-name=olmo-grpo-trial
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm-train-%j.out
set -euo pipefail

ENV="${ENV:-/gpfs/radev/project/krishnaswamy_smita/gss34/envs/eval-awareness}"
export MODEL_PATH="${MODEL_PATH:-weights/Olmo-3.1-32B-Think}"
export DATASET_PATH="${DATASET_PATH:-data/trial.jsonl}"
export PATH="$ENV/bin:$PATH"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
nvidia-smi
bash scripts/train.sh "$@"
