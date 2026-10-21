#!/usr/bin/env bash
# The whole CoT-length experiment in one H200 allocation (one queue wait instead of three):
#   1. evaluate the base model on the held-out probability problems
#   2. train a LoRA with GRPO using configs/prob_cot_length.yaml
#   3. evaluate the trained adapter on the same problems
#   4. write the table and figure (results/eval/summary.md, cot_length.png)
# Usage: sbatch scripts/run_length_experiment.sbatch
#SBATCH --job-name=cot-length-exp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=16:00:00
#SBATCH --output=results/slurm-cot-length-exp-%j.out
set -euo pipefail

ENV="${ENV:-/gpfs/radev/project/krishnaswamy_smita/gss34/envs/eval-awareness}"
export PATH="$ENV/bin:$PATH"
export MODEL_PATH="${MODEL_PATH:-weights/Olmo-3.1-32B-Think}"
export DATASET_PATH=data/prob_train.jsonl CONFIG=configs/prob_cot_length.yaml OUTPUT_DIR=results/prob_cot_length
EVAL_ARGS=(--model-path "$MODEL_PATH" --eval-file data/prob_eval.jsonl --max-new-tokens 8192 --batch-size 16)

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p results/eval
[ -f data/prob_train.jsonl ] || python scripts/make_prob_data.py
nvidia-smi

echo "=== [1/4] Base model eval $(date)"
python -u src/evaluate.py --label base "${EVAL_ARGS[@]}"

echo "=== [2/4] GRPO + LoRA training $(date)"
bash scripts/train.sh

echo "=== [3/4] Adapter eval $(date)"
python -u src/evaluate.py --label cot_length --adapter-path "$OUTPUT_DIR/adapter" "${EVAL_ARGS[@]}"

echo "=== [4/4] Analysis $(date)"
python src/analyze.py --labels base cot_length --runs cot_length="$OUTPUT_DIR"
echo "=== Done $(date)"
