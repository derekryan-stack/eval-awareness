# Eval Awareness: GRPO + LoRA Trial

A minimal end-to-end trial. It loads the OLMo 3.1-32B-Think checkpoint, runs a few GRPO steps with LoRA and a simple task reward, saves the adapter, then reloads it and generates a response. Its only purpose is to validate the training setup; the actual research experiment comes later.

The model weights go into the weights dir, and the weights should not be committed.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Pin exact versions in `requirements.txt` once a trial succeeds on the cluster.

## Dataset format

The dataset is a JSONL file with `prompt` and `answer` fields:

```jsonl
{"prompt": "What is 7 * 6?", "answer": "42"}
{"prompt": "What is the capital of France?", "answer": "Paris"}
```

A completion gets reward 1.0 if the answer text appears after the model's `</think>` block, and 0.0 otherwise.

## Run

```bash
MODEL_PATH=/cluster/path/to/OLMo-3.1-32B-Think \
DATASET_PATH=/cluster/path/to/trial.jsonl \
bash scripts/train.sh
```

Optional: `OUTPUT_DIR` (default `results/trial`) and `CONFIG` (default `configs/trial.yaml`). To change hyperparameters, edit `configs/trial.yaml` or point `CONFIG` at a copy of it. You don't need to change the source code.

## Outputs

```
results/trial/
├── adapter/        # LoRA adapter
├── config.yaml     # resolved config
├── metadata.json   # seed, paths, versions, GPUs, times, status, verification response
└── metrics.json    # training metrics and per-step logs
```

The run succeeded if `metadata.json` shows `"status": "succeeded"`, `global_step` is at least 1, and `verification_response` is not empty.

## Troubleshooting

- **Out of memory:** lower `max_completion_length` or `num_generations`, or run on more GPUs (the model is sharded with `device_map="auto"`).
- **Batch size error:** `per_device_train_batch_size * gradient_accumulation_steps` must be divisible by `num_generations`.
- **LoRA target-module error:** check the module names with `print(model)` and update `lora.target_modules`.
- **GRPOConfig/GRPOTrainer argument errors:** the TRL API changes between versions, so compare against the docs for the installed version.
- **All rewards are 0:** a Think model may not finish reasoning within 512 tokens, so the adapter barely changes. That's acceptable for this trial, because it's validating the pipeline rather than results.
