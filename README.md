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

If you train on RunPod, use the pre-built community template — ComputeFence runs automatically on pod start.

**[→ Launch ComputeFence Pre-Flight Template](https://runpod.io/console/deploy?template=cq5psqzck6)**

- Runs `computefence doctor` on every pod start
- `HF_HOME` set to `/workspace/.cache/huggingface` by default
- Persistent `/workspace` volume included

---

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

Skip specific checks by ID:

```bash
computefence doctor --skip accelerate_config,persistent_volume
```

JSON output for scripting:

```bash
computefence doctor --json
```

Add to your pod startup script so it runs automatically before every job:

```bash
pip install computefence==0.3.3 && computefence doctor && python train.py
```

---

## Exit codes

| Code | Meaning |
|---|---|
| 0 | All checks passed |
| 1 | Warnings — review before launching |
| 2 | Blockers — fix before launching |
| 64 | Usage error (unknown command) |

---

## What It Checks / What It Doesn't

| Checks | Does not check |
|---|---|
| PyTorch can see the GPU (silent CPU fallback) | Training script correctness |
| GPU architecture compatible with PyTorch build | Model architecture compatibility |
| HuggingFace cache on persistent storage | Learning rate or hyperparameters |
| Accelerate GPU count matches instance | Runtime behaviour during the job |
| Disk space headroom (warn <20 GB, block <5 GB) | GPU utilisation or dataloader throughput |
| Checkpoint output dir on persistent storage | Mid-run pod failures |
| Dataset duplicates, missing values, label conflicts | Export-stage OOM |
| Environment drift between pods | Merge integrity |

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

Example drift output:

Drift
⚠ [Drift] PyTorch version changed: 2.4.0+cu121 → 2.8.0+cu128
Fix: Pin torch version in your requirements or Dockerfile.
⚠ [Drift] CUDA version changed: 12.1 → 12.8
Fix: Ensure your pod template uses the same CUDA base image.
⚠ [Drift] GPU count changed: 2 → 1
Fix: Check your pod configuration matches the saved environment.


Drift findings are warnings only — they never block a launch. The fingerprint is written to `.computefence-lock.json` on your persistent volume (`/workspace` or `--path` to override).

---

## Example Output

ComputeFence v0.3.3 — Pre-flight diagnostic
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2 WARNINGS · 0 BLOCKERS · 6 PASSED

Environment
✓ Python 3.11.0
✓ PyTorch 2.4.1+cu124 detected (CUDA available — NVIDIA A40)
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
Pseudonymous run stats are not collected by default. To opt in: touch ~/.computefence_telemetry


---

## Telemetry

Opt-in pseudonymous run statistics. Nothing is sent by default.

To opt in:

```bash
touch ~/.computefence_telemetry
```

To opt out if previously opted in, or to honour `DO_NOT_TRACK`:

```bash
DO_NOT_TRACK=1 computefence doctor
```

No personal data, no file paths, no model weights. Fields sent: version, Python version, platform, GPU count, check results.

---

## Real Operator Results

**David at Neuralic** ran ComputeFence on a RunPod A100. It caught HF_HOME writing to /root/.cache and an Accelerate GPU count mismatch. He fixed both before launch.

**Shahzeb Ali**, a computer vision engineer running client training jobs on RunPod, confirmed the storage warning matches real pod behaviour. He would not have caught the HF_HOME issue without the tool.

**An independent Vast.ai operator** ran `computefence doctor` six times overnight before a real training job without being prompted or paid to do so.

Thirteen independent ML engineers confirmed this problem across RunPod, Vast.ai, and AWS. Nine had lost checkpoints to ephemeral disk. Five confirmed Docker does not solve job-specific configuration mistakes.

---

## Why This Exists

I burned approximately £1,000 on GPU training runs that failed silently.

CUDA fell back to CPU with no error message. 24 seconds per iteration instead of 0.4. Class weights caused loss to collapse to 0.693 immediately. My dataset had 28,432 duplicate rows and 312 conflicting labels I only found after the run finished.

Nothing existed that caught these before the job started. So I built it.

---

## Founding Design Partner Pilot

Running high-cost GPU training jobs on RunPod or Vast.ai?

Three founding design partner spots at $99 for three months:

- Personal audit of your launch templates and persistent storage configuration
- ComputeFence installed into your pod startup scripts — pre-flight runs automatically before every job
- Monthly 30-minute call where you shape what gets built next
- Money back if it does not catch one actionable issue in three months

Email francisco@booth.ws to apply.

---

## Links

- GitHub: https://github.com/Francisco-Booth/ComputeFence
- PyPI: https://pypi.org/project/computefence
```


**Live sessions:** The maintainer is available Mon–Fri 09:00–18:00 UK time for real-time pre-flight checks. Open a GitHub issue or DM on Discord (FounderCEO).





