[![ComputeFence Action](https://img.shields.io/badge/GitHub%20Action-computefence--action-blue)](https://github.com/Francisco-Booth/computefence-action)

# ComputeFence

**Stop paying for silent failures.**

One command before your training job starts. Catches the configuration mistakes that cost you hours of GPU compute and lost checkpoints — before billing begins.

Built for RunPod, Vast.ai, Lambda Labs, CoreWeave, Paperspace, and any bare metal GPU provider.

---

## Install

```bash
pip install computefence
```

No install required with UV:

```bash
uvx computefence doctor
```

---


## ☁️ RunPod One-Click Template

If you train on RunPod, use the pre-built community template — ComputeFence is pre-installed and runs automatically on pod start.

**[→ Launch ComputeFence Pre-Flight Template](https://runpod.io/console/deploy?template=cq5psqzck6)**

- Runs `computefence doctor` on every pod start
- Logs preflight output to `/workspace/computefence-preflight.log`
- `HF_HOME` set to `/workspace/.cache/huggingface` by default
- Persistent `/workspace` volume included


## Usage

```bash
computefence doctor
```

With dataset validation:

```bash
computefence doctor --dataset train.csv --input-column text --label-column label
```

With checkpoint directory validation:

```bash
computefence doctor --output-dir ./checkpoints
```

Add to your pod startup script so it runs automatically before every job:

```bash
pip install computefence && computefence doctor && python train.py
```

---

## Drift Detection

Save a fingerprint of your working environment after a clean run:

```bash
computefence doctor --save-env
```

On your next pod, check whether anything has changed:

```bash
computefence doctor --check-drift
```

Example output when the pod environment has drifted:

Drift
⚠ [Drift] PyTorch version changed: 2.4.0+cu121 → 2.8.0+cu128
Fix: Pin torch version in your requirements or Dockerfile.
⚠ [Drift] CUDA version changed: 12.1 → 12.8
Fix: Ensure your pod template uses the same CUDA base image.
⚠ [Drift] GPU count changed: 2 → 1
Fix: Check your pod configuration matches the saved environment.


Drift findings are warnings only — they never block a launch. The fingerprint is written to `.computefence-lock.json` on your persistent volume (`/workspace` or `--path` to override). It survives pod stop and restart.

---

## Example Output

ComputeFence v0.2.14 — Pre-flight diagnostic
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2 WARNINGS · 0 BLOCKERS · 6 PASSED

Environment
✓ Python 3.12.3
✓ PyTorch 2.8.0+cu128 detected (CUDA available — NVIDIA RTX PRO 4500)
⚠ No Accelerate config found — skipping GPU count check
Fix: Run ‘accelerate config’ to create a configuration file

Storage
✓ HF_HOME is set to /workspace/.cache/huggingface/
✓ Persistent volume(s) detected: /workspace
✓ Volume /workspace — 49.9 GB free of 50.0 GB
✓ Root disk (/) — 49.9 GB free of 50.0 GB

Dataset
⚠ No dataset path provided — skipping dataset checks
Fix: Run with --dataset path/to/your/dataset.csv to enable dataset checks

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2 warning(s) found. Review before launching.

Anonymous run stats are collected to improve ComputeFence.
To opt out: touch ~/.computefence_no_telemetry


---

## What It Checks

**CUDA and GPU visibility**
Confirms PyTorch can see the GPU. Catches silent CPU fallback before it runs for eight hours at 24 seconds per iteration instead of 0.4.

**HuggingFace cache path**
Confirms model weights are writing to persistent storage. If HF_HOME points to ephemeral disk your weights disappear when the pod stops.

**Accelerate GPU count**
Confirms your distributed training config matches the GPUs actually on the instance. A mismatch leaves GPUs idle for the entire run with no error message.

**Disk space headroom**
Checks free space on workspace volumes and root disk. Warns below 20 GB. Blocks below 5 GB. Catches the failure where checkpoints stop saving at 90 percent completion because the disk filled.

**Checkpoint output directory**
Confirms your training script's output path is on persistent storage not ephemeral disk. HF_HOME and your checkpoint directory are separate paths — fixing one does not fix the other.

**Dataset integrity**
Optional scan for duplicates, missing values, and conflicting labels. Pass --dataset to enable.

**Environment drift detection**
Saves a 4-vector fingerprint (PyTorch/CUDA version, storage paths, GPU count and compute capability, critical library versions) and diffs it on the next pod. Catches silent environment changes between runs.

---

## What It Does Not Check

- Training script correctness
- Model architecture compatibility
- Learning rate or hyperparameter safety
- Runtime behaviour during the job
- GPU utilisation or dataloader throughput during training

ComputeFence addresses the job configuration layer before launch. It does not monitor the run.

---

## Why This Exists

I burned approximately £1,000 on GPU training runs that failed silently.

CUDA fell back to CPU with no error message. 24 seconds per iteration instead of 0.4. Class weights caused loss to collapse to 0.693 immediately. My dataset had 28,432 duplicate rows and 312 conflicting labels I only found after the run finished.

Nothing existed that caught these before the job started. So I built it.

---

## Real Operator Results

**David at Neuralic** ran ComputeFence on a RunPod A100. It caught HF_HOME writing to /root/.cache and an Accelerate GPU count mismatch. He fixed both before launch.

**Shahzeb Ali**, a computer vision engineer running client training jobs on RunPod, confirmed the storage warning matches real pod behaviour. He would not have caught the HF_HOME issue without the tool.

**An independent Vast.ai operator** ran computefence doctor six times overnight before a real training job without being prompted or paid to do so.

Thirteen independent ML engineers confirmed this problem across RunPod, Vast.ai, and AWS. Nine had lost checkpoints to ephemeral disk. Five confirmed Docker does not solve job-specific configuration mistakes.

---

## The Problem It Solves

A healthy GPU does not mean you are running the right job.

Docker makes environments reproducible. It does not check whether your HuggingFace cache is writing to ephemeral storage that disappears on pod stop, whether your Accelerate config matches the GPUs on the instance, or whether your checkpoint output directory is on persistent storage.

ComputeFence addresses the job configuration layer — not the environment layer.

HF_HOME and your checkpoint output directory are separate paths. Fixing one does not fix the other. Both disappear on pod restart if they point to ephemeral disk.

---

## Telemetry

Anonymous run statistics are collected on every run to help improve ComputeFence. No personal data, no code, no model weights. To opt out:

```bash
touch ~/.computefence_no_telemetry
```

---

## Founding Design Partner Pilot

Running high-cost GPU training jobs on RunPod or Vast.ai?

Three founding design partner spots at $99 for three months:

- Personal audit of your launch templates and persistent storage configuration
- ComputeFence installed into your pod startup scripts — pre-flight runs automatically before every job
- Slack or Discord webhook alert when a check fails or blocks a launch
- Monthly 30-minute call where you shape what gets built next
- Money back if it does not catch one actionable issue in three months

Email francisco@booth.ws to apply.

---

## Links

- GitHub: https://github.com/Francisco-Booth/ComputeFence
- PyPI: https://pypi.org/project/computefence
```


**Live sessions:** The maintainer is available Mon–Fri 09:00–18:00 UK time for real-time pre-flight checks. Open a GitHub issue or DM on Discord (FounderCEO).





