ComputeFence

Pre-flight safety gate for ML training jobs on rented GPU compute. Built for RunPod, Vast.ai, Lambda Labs, CoreWeave, Paperspace, and any bare metal GPU provider.

Install
bash
pip install computefence

Or with UV (no install required):

bash
uvx computefence doctor
Usage
bash
computefence doctor

With a dataset:

bash
computefence doctor --dataset train.csv --input-column text --label-column label

With checkpoint output directory validation:

bash
computefence doctor --output-dir ./checkpoints

Add to your pod startup script so it runs automatically before every job:

bash
pip install computefence && computefence doctor && python train.py
Example output
ComputeFence v0.2.4 — Pre-flight diagnostic
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2 WARNINGS  ·  0 BLOCKERS  ·  3 PASSED

Environment
  ✓ Python 3.11.4
  ✓ PyTorch 2.1.0 detected
  ✓ CUDA available — NVIDIA A40

Storage
  ⚠ HF_HOME is not set. HuggingFace will use default local cache.
    Fix: export HF_HOME=/workspace/.cache/huggingface
  ⚠ Root disk (/) — 14.3 GB free of 460.4 GB (below 20 GB)
    Fix: Free up disk space or move checkpoints to a larger volume: df -h to check usage

Dataset
  ✓ No dataset path provided — skipping dataset checks

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2 warning(s) found. Review before launching.
Anonymous run stats are collected to improve ComputeFence. To opt out: touch ~/.computefence_no_telemetry
What it checks
CUDA and GPU visibility — confirms PyTorch can see the GPU and training will not silently fall back to CPU
HuggingFace cache path — confirms model weights go to persistent storage not ephemeral disk that disappears on pod stop
Accelerate GPU count — confirms your distributed training config matches the GPUs actually on the instance
Disk space headroom — checks free space on workspace volumes and root disk. Warns below 20 GB, blocks below 5 GB
Checkpoint output directory — confirms your training script's output path is on persistent storage not ephemeral disk (--output-dir)
Dataset integrity — optional scan for duplicates, missing values, and conflicting labels (--dataset)
What it does not check
Training script correctness
Model architecture compatibility
Learning rate or hyperparameter safety
Runtime monitoring during the job
Dataloader throughput or GPU utilisation during training
Why this exists

I burned approximately £1,000 on GPU training runs that failed silently.

CUDA fell back to CPU with no error. 24 seconds per iteration instead of 0.4. Class weights caused loss collapse to 0.693 immediately. My dataset had 28,432 duplicate rows and 312 conflicting labels I only found after the run.

Nothing existed that caught these before the job started. So I built it.

Real operator results

David at Neuralic ran ComputeFence on a RunPod A100. It caught HF_HOME writing to /root/.cache and an Accelerate GPU count mismatch. He fixed both before launch.

Shahzeb Ali, a computer vision engineer running client training jobs on RunPod, confirmed the storage warning matches real pod behaviour and would not have caught the HF_HOME issue explicitly without the tool.

Seven overnight organic runs appeared in telemetry from a Vast.ai operator running the tool six times in two minutes before a real training job — without being prompted or paid to do so.

The problem it solves

A healthy GPU does not mean you are training the right job.

Docker makes environments reproducible. It does not check whether your HuggingFace cache is writing to ephemeral storage that disappears on pod stop, whether your Accelerate config matches the GPUs actually on the instance, or whether your training script's checkpoint output directory is on persistent storage. ComputeFence addresses the job configuration layer — not the environment layer.

HF_HOME and your checkpoint output directory are separate paths. Fixing one does not fix the other. Both disappear on pod restart if they point to ephemeral disk.

Thirteen independent ML engineers confirmed this problem independently across RunPod, Vast.ai, and AWS. Five confirmed that Docker does not solve job-specific configuration mistakes.

Founding Design Partner pilot

Running high-cost GPU training on RunPod or Vast.ai?

We offer a hands-on Founding Design Partner pilot at $99 for 3 months:

Personal audit of your launch templates and persistent storage configuration
ComputeFence installed into your pod startup scripts so pre-flight runs automatically on every job
Slack or Discord webhook alert when a check fails or blocks a launch
Monthly 30-minute call where you shape what gets built next
Money back if it does not catch one bad launch in 3 months

Three spots available. Email francisco@booth.ws to apply.

Links
GitHub: github.com/Francisco-Booth/ComputeFence
PyPI: pypi.org/project/computefence
