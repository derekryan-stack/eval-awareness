Product Requirements Document

Eval Awareness Research: GRPO + LoRA Training Skeleton

Status: Initial implementation
Priority: High
Purpose: Validate the end-to-end training setup before implementing the full research experiment.

⸻

1. Overview

This project will establish a minimal, reproducible training pipeline for studying evaluation awareness in large language models.

The initial implementation will load the OLMo 3.1-32B-Think model from a checkpoint stored on the compute cluster, load a training dataset, configure LoRA and GRPO, run a small training trial, save the resulting adapter and metadata, and verify that the adapter can be reloaded for inference.

The goal is to validate the infrastructure and workflow before expanding to multiple reward strategies, seeds, and evaluation benchmarks.

2. Goals

The initial implementation must:

1. Load the base model and tokenizer from a configurable cluster path.
2. Load a dataset from a configurable path.
3. Configure LoRA and GRPO for a small training run.
4. Execute a short training trial using a simple task reward.
5. Save the trained adapter and experiment metadata.
6. Reload the adapter alongside the base model and generate a response.
7. Make the trial reproducible through configuration and recorded metadata.
8. Provide clear instructions for running the trial on the cluster.

3. Non-Goals

The initial implementation will not include:

* All six planned reward strategies.
* An LLM-based judge or a fully developed awareness reward.
* KL-divergence-threshold stopping.
* The complete 18-run experiment across six strategies and three seeds.
* EvalAwareBench, HarmBench adaptations, or the complete evaluation suite.
* Activation-probe training or analysis.
* Statistical analysis across experimental runs.
* A dashboard or other visualization interface.
* Model weights, datasets, or trained checkpoints stored in Git.

These features can be added after the basic training and checkpoint workflow is validated.

4. Technical Stack

The initial implementation is expected to use:

* Python for training code.
* PyTorch for model computation.
* Transformers for loading models and tokenizers.
* PEFT for LoRA.
* TRL for GRPO training.
* Datasets for loading training data.
* YAML for configuration.
* JSON/JSONL for experiment metadata and logs.

Package versions must be checked against the selected model checkpoint and cluster environment. Pin compatible versions once a successful end-to-end trial has been completed.

5. Repository Structure

Create the following minimal structure:

eval-awareness/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── trial.yaml
├── src/
│   └── train.py
├── scripts/
│   └── train.sh
└── results/

File responsibilities

File	Responsibility
README.md	Setup instructions, configuration, execution, and troubleshooting.
requirements.txt	Python dependencies, with compatible versions pinned after validation.
.gitignore	Exclude checkpoints, generated results, caches, secrets, and environment files.
configs/trial.yaml	Model, dataset, training, LoRA, and output configuration.
src/train.py	Load resources, initialize GRPO + LoRA, train, and save artifacts.
scripts/train.sh	Validate paths and launch the Python training script.
results/	Store generated experiment artifacts locally or on the cluster; do not commit them.

6. Configuration Requirements

The trial must be configurable without modifying the training source code.

Example configs/trial.yaml

model:
  name: "OLMo-3.1-32B-Think"
  path: null
dataset:
  path: null
  split: "train"
training:
  seed: 42
  max_steps: 3
  learning_rate: 1.0e-5
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 1
  num_generations: 2
  max_completion_length: 512
  beta: 0.01
lora:
  r: 16
  alpha: 32
  dropout: 0.05
output:
  dir: "results/trial"

Important: These values are illustrative starting points, not validated settings. In particular, the batch size, number of generations, completion length, LoRA configuration, and GRPO settings must be checked against available GPU memory and the installed TRL version.

The configuration must support specifying the model and dataset paths through configuration or command-line/environment overrides.

For example:

MODEL_PATH=/cluster/path/to/model
DATASET_PATH=/cluster/path/to/dataset

The actual paths should be supplied by the person running the experiment, rather than hardcoded into the repository.

7. Functional Requirements

FR-1: Configurable Model Loading

The training script must:

* Accept a configurable base-model path.
* Load the model and tokenizer from that location.
* Report a useful error if the path is invalid or the checkpoint cannot be loaded.
* Avoid downloading model weights unnecessarily when the cluster already contains them.
* Run a basic inference check before training when feasible.

FR-2: Configurable Dataset Loading

The training script must:

* Accept a configurable dataset path.
* Load the specified dataset split.
* Validate that the dataset contains the fields required by the training pipeline.
* Report missing fields and incompatible data formats clearly.

The trial dataset should be small enough to validate the pipeline without requiring the full research dataset.

FR-3: Minimal GRPO + LoRA Training

The script must:

1. Initialize the model and tokenizer.
2. Configure LoRA using a target-module configuration compatible with the checkpoint architecture.
3. Prepare the dataset in the format required by the installed GRPO implementation.
4. Define a simple task-reward function.
5. Initialize the GRPO trainer.
6. Execute a short training run.
7. Record basic training information and errors.

The initial reward function should be intentionally simple. Its purpose is to validate the training pipeline, not to measure evaluation awareness.

The implementation must verify the installed TRL version’s expected trainer arguments, reward-function signature, and dataset format instead of assuming that all versions use the same API.

FR-4: Checkpoint Persistence

After training, the script must save:

* The trained LoRA adapter.
* The resolved experiment configuration.
* Experiment metadata.
* Available training metrics and logs.

The output directory must be configurable.

Example:

results/trial/
├── adapter/
├── config.yaml
├── metadata.json
└── metrics.json

The exact artifact names may be adjusted to match the selected training framework.

The base model weights should not be copied into the experiment output directory unless explicitly required.

FR-5: Adapter Reload and Inference

The project must include a way to verify that the trained adapter can be loaded with the original base model.

The verification step must:

1. Load the base model and tokenizer.
2. Load the saved adapter.
3. Generate a response to a small test prompt.
4. Report whether loading and generation succeeded.

This step verifies that the saved artifact can be reused after the training process exits.

FR-6: Reproducibility Metadata

The experiment metadata must record, where available:

* Random seed.
* Base-model identifier and path.
* Dataset identifier and path.
* Resolved training configuration.
* LoRA configuration.
* Output path.
* Python and relevant library versions.
* Start time and end time.
* Training status.
* Relevant hardware information.

Set the relevant random seeds before training.

A fixed seed alone does not guarantee bit-for-bit reproducibility across different hardware, software versions, or nondeterministic GPU operations. The objective of the initial implementation is to make experiments traceable and repeatable under a consistent environment.

FR-7: Runtime Diagnostics

The training script must provide useful diagnostics for common failures, including:

* Invalid model or dataset paths.
* Missing dependencies.
* Incompatible model or trainer configuration.
* GPU out-of-memory errors.
* Invalid dataset schemas.
* LoRA target-module mismatches.
* Failed checkpoint saves.
* Failed adapter reloads.

Diagnostics should identify the failed stage rather than only reporting that the overall experiment failed.

8. Training Launcher

scripts/train.sh

The shell script should:

1. Enable error handling so failed commands do not silently pass.
2. Check that required model and dataset paths have been supplied.
3. Accept or forward the configuration path and relevant overrides.
4. Create the output directory if necessary.
5. Launch src/train.py.
6. Return a nonzero exit code if training fails.

Example invocation:

MODEL_PATH=/cluster/path/to/model \
DATASET_PATH=/cluster/path/to/dataset \
bash scripts/train.sh

The actual launcher should pass these variables to Python explicitly, either through command-line arguments or another clearly defined configuration mechanism.

The script is intended to launch training, not to manage cluster scheduling. If the cluster uses Slurm or another scheduler, scheduler-specific submission scripts can be added separately.

9. Acceptance Criteria

The initial implementation is complete when the following criteria are met.

* AC-1: The required Python environment can be initialized on the target cluster.
* AC-2: The base model and tokenizer load from the supplied path.
* AC-3: A small dataset loads and passes schema validation.
* AC-4: The GRPO trainer initializes successfully with LoRA enabled.
* AC-5: At least one optimizer update completes successfully.
* AC-6: The run saves a LoRA adapter and experiment metadata.
* AC-7: The saved adapter can be loaded with the base model.
* AC-8: The reloaded model successfully generates a response.
* AC-9: Large model weights, generated checkpoints, and other experiment artifacts are not committed to Git.
* AC-10: Another team member can follow the README, supply their own paths, and reproduce the trial without modifying the training source code.

A successful trainer initialization alone is insufficient. The trial must complete a training update, save an adapter, and reload that adapter successfully.

10. Implementation Plan

Phase 1: Environment and Model Validation

* Confirm access to the model checkpoint.
* Check GPU availability and memory.
* Confirm compatible Python, PyTorch, Transformers, PEFT, and TRL versions.
* Load the model and tokenizer.
* Run a short inference test.

Deliverable: A confirmed working model-loading environment.

Phase 2: Minimal Training Implementation

* Create the configuration file.
* Implement dataset loading and validation.
* Configure LoRA.
* Implement a simple task reward.
* Initialize the GRPO trainer.
* Run a short training trial.

Deliverable: A training run that completes at least one optimizer update.

Phase 3: Persistence and Reload

* Save the adapter.
* Save configuration and metadata.
* Reload the adapter with the base model.
* Generate a response from the reloaded model.

Deliverable: A verified end-to-end training and checkpoint workflow.

Phase 4: Documentation and Handoff

* Document environment setup.
* Document how to supply model and dataset paths.
* Document how to launch the trial.
* Document expected output artifacts.
* Record any cluster-specific limitations and known issues.

Deliverable: A reproducible trial that another team member can run.

11. Risks and Mitigations

Risk	Mitigation
Insufficient GPU memory for the 32B model	Start with an inference test; inspect available memory and supported precision, then adjust the configuration.
Incompatible library versions	Test compatibility before implementing the full training workflow and pin working versions.
Incorrect LoRA target modules	Verify module names against the loaded model architecture.
Dataset format mismatch	Validate required fields and convert the data into the trainer’s expected format.
Reward function incompatibility	Test the reward function independently and confirm the installed TRL API.
Adapter cannot be reloaded	Include a separate reload-and-inference check in the acceptance criteria.
Paths differ across team members	Use configuration and environment/CLI overrides rather than hardcoding paths.
Training trial takes too long	Keep the initial dataset and number of training steps small.
Large artifacts are accidentally committed	Add checkpoint and results directories to .gitignore.

12. Future Extensions

After the initial pipeline is validated, the project can expand to support:

1. Six reward strategies:
    * Task reward only.
    * LLM-judge verbalized-awareness penalty.
    * LLM-judge awareness reward.
    * Length penalty.
    * Generic preference reward.
    * Formatting reward.
2. Configurable multi-seed experiments.
3. KL-divergence-based stopping.
4. Standardized evaluation of base and fine-tuned models.
5. EvalAwareBench and the Hawthorne-trigger dataset.
6. An adapted HarmBench-based evaluation, with validation appropriate to the intended use.
7. An independent LLM judge.
8. Activation-probe analysis before and after post-training.
9. General-capability evaluation using the selected Vals Index setup.
10. Automated aggregation and statistical comparison across runs.

These extensions should build on the validated training and checkpoint interfaces rather than requiring the initial skeleton to be rewritten.

13. Definition of Done

The initial skeleton is done when a team member can:

1. Point the configuration to the existing cluster checkpoint.
2. Supply a compatible dataset path.
3. Launch a short GRPO + LoRA training run.
4. Complete at least one optimizer update.
5. Save the adapter and experiment metadata.
6. Reload the adapter alongside the base model.
7. Generate a response from the reloaded model.
8. Repeat the process using configuration changes rather than editing the training source code.

The immediate objective is a verified, reproducible end-to-end trial—not the complete evaluation-awareness experiment.