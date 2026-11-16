#!/usr/bin/env bash
# Run src/train.py with the Python on PATH (inside an allocation or an interactive GPU session).
# Usage: CONFIG=configs/trial.yaml [MODEL_PATH=...] [DATASET_PATH=...] [OUTPUT_DIR=...] bash scripts/train.sh [train.py args]
# For batch jobs use scripts/submit_experiment.sh, which also activates the env and checks paths.
set -euo pipefail

CONFIG="${CONFIG:-configs/trial.yaml}"
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" HF_HUB_DISABLE_TELEMETRY=1

# MODEL_PATH, DATASET_PATH, OUTPUT_DIR, OUTPUT_ROOT and SEED are read from the environment by src/config.py.
python -u src/train.py --config "$CONFIG" "$@"
