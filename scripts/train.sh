#!/usr/bin/env bash
# Launch the GRPO + LoRA trial.
# Usage: MODEL_PATH=/path/to/model DATASET_PATH=/path/to/data.jsonl bash scripts/train.sh
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the base model checkpoint directory}"
: "${DATASET_PATH:?Set DATASET_PATH to the training JSONL file}"
CONFIG="${CONFIG:-configs/trial.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-results/trial}"

cd "$(dirname "$0")/.."
mkdir -p "$OUTPUT_DIR"

python src/train.py \
  --config "$CONFIG" \
  --model-path "$MODEL_PATH" \
  --dataset-path "$DATASET_PATH" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
