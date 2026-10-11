# ComputeFence

Pre-flight safety gate for ML training jobs on rented GPU compute. Built for RunPod, Vast.ai, Lambda Labs, and similar bare metal providers.

## Install

```bash
pip install computefence
```

Or with UV:

```bash
uvx computefence doctor
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

ComputeFence v0.2.0 — Pre-flight diagnostic
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2 WARNINGS · 0 BLOCKERS · 3 PASSED

Environment
✓ Python 3.11.4
✓ PyTorch 2.1.0 detected
✓ CUDA available — NVIDIA A40

Storage
⚠ HF_HOME is not set. HuggingFace will use default local cache.
Fix: export HF_HOME=/workspace/.cache/huggingface
⚠ Root disk (/) — 14.3 GB free of 460.4 GB (below 20 GB)
Fix: Free up disk space or attach a larger volume before launching

Dataset
✓ No dataset path provided — skipping dataset checks

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2 warning(s) found. Review before launching.


## What it checks

- **CUDA and GPU visibility** — confirms PyTorch can see the GPU and training will not silently fall back to CPU
- **HuggingFace cache path** — confirms model files go to persistent storage not ephemeral disk
- **Accelerate GPU count** — confirms your distributed training config matches the GPUs actually on the instance
- **Disk space headroom** — checks free space on workspace volumes and root disk. Warns below 20 GB, blocks below 5 GB
- **Dataset integrity** — optional scan for duplicates, missing values, and conflicting labels

## What it does not check

- Training script correctness
- Model architecture compatibility
- Learning rate or hyperparameter safety
- Runtime monitoring during the job
- Slow dataloader or data pipeline throughput
- Dataloader bottleneck causing low GPU utilisation

## Why this exists

I burned approximately £1,000 on GPU training runs that failed silently. CUDA fell back to CPU with no error. 24 seconds per iteration instead of 0.4. Class weights caused loss collapse to 0.693 immediately. My dataset had 28,432 duplicate rows and 312 conflicting labels I only found after the run.

Nothing existed that caught these before the job started. So I built it.

## Real operator results

David at Neuralic ran ComputeFence on a RunPod A100. It caught HF_HOME writing to /root/.cache and an Accelerate GPU count mismatch. He fixed both before launch.

Shahzeb Ali, a computer vision engineer running client training jobs on RunPod, confirmed the storage warning matches real pod behaviour and would not have caught the HF_HOME issue explicitly without the tool.

## The problem it solves

A healthy GPU does not mean you are training the right job.

Docker makes environments reproducible. It does not check whether your HuggingFace cache is writing to ephemeral storage, whether your Accelerate config matches the GPUs actually on the instance, or whether your disk has enough headroom for checkpoints. ComputeFence addresses the job layer not the environment layer.

Five independent operators confirmed this independently. Docker does not solve job-specific configuration mistakes.

## GitHub

github.com/Francisco-Booth/ComputeFence

## PyPI

pypi.org/project/computefence
