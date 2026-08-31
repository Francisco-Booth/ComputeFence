# ComputeFence

Pre-flight safety gate for ML training jobs on rented GPU compute. Built specifically for RunPod, Vast.ai, Lambda Labs, and similar bare metal providers.

## Install

```bash
pip install computefence
```

## Usage

```bash
computefence doctor
```

With a dataset:

```bash
computefence doctor --dataset train.csv --input-column text --label-column label
```

## Example output

ComputeFence v0.1.2 — Pre-flight diagnostic
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2 WARNINGS · 0 BLOCKERS · 3 PASSED

Environment
✓ Python 3.11.4
✓ PyTorch 2.1.0 detected
✓ CUDA available — NVIDIA A40

Storage
⚠ HF_HOME is not set. HuggingFace will use default local cache.
Fix: Set HF_HOME to a persistent volume path before training
⚠ No mounted persistent volumes detected at /workspace, /runpod-volume, or /vast
Fix: Attach a network volume before training to persist checkpoints and cache

Dataset
✓ No dataset path provided — skipping dataset checks

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2 warning(s) found. Review before launching.

## What it checks

- **CUDA and GPU visibility** — confirms PyTorch can see the GPU and training will not silently fall back to CPU
- **HuggingFace cache path** — confirms model files go to persistent storage not ephemeral disk
- **Accelerate GPU count** — confirms your distributed training config matches the GPUs actually on the instance
- **Dataset integrity** — optional scan for duplicates, missing values, and conflicting labels

## What it does not check

- Training script correctness
- Model architecture compatibility
- Learning rate or hyperparameter safety
- Runtime monitoring during the job
- Slow dataloader or data pipeline throughput

## Why this exists

I burned approximately £1,000 on GPU training runs that failed silently. CUDA fell back to CPU with no error. Class weights caused loss collapse to 0.693 immediately. My dataset had 28,432 duplicate rows and 312 conflicting labels I only found after the run.

Nothing existed that caught these before the job started. So I built it.

## Real operator results

David at Neuralic ran ComputeFence on a RunPod A100. It caught HF_HOME writing to /root/.cache and an Accelerate GPU count mismatch. He fixed both before launch.

## GitHub

github.com/Francisco-Booth/ComputeFence

## PyPI

pypi.org/project/computefence
