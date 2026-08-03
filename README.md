# TrainGuard

Pre-flight validation and real-time monitoring for HuggingFace fine-tuning runs.

## The problem

Fine-tuning on rented GPU compute has no safety net. You start a run, walk away, 
and find out hours later that it failed silently — CUDA fell back to CPU, the model 
collapsed, the dataset had 28k duplicate rows. The billing clock ran the whole time.

TrainGuard catches these failures before they cost you money.

## What it catches

**Before the run starts**
- CUDA version incompatibility with PyTorch (silent CPU fallback)
- GPU not available — kills the instance rather than billing for CPU
- Dataset duplicates, conflicting labels, class imbalance, missing values
- Class weight configurations with high collapse risk
- Estimated cost and runtime before you commit

**During training**
- Loss curve collapse detection (the 0.693 signature)
- Auto-kill on collapse — stops billing immediately
- Live dashboard: loss, accuracy, GPU utilisation, cost so far
- Phone alert on completion or collapse

**After training**
- Per-category benchmark vs previous model version
- Deploy or reject recommendation with reason
- Structured run report saved to JSON

## CLI

```bash
# Validate a dataset before running
trainguard check --dataset train_large_v15.csv

# Wrap any existing training script
trainguard run --script train_deberta_3class.py \
               --dataset train_large_v15.csv \
               --benchmark benchmark_test_set_v4.csv \
               --prev-model v11 \
               --budget 5.00 \
               --notify phone

# Quick test mode — 10% data, 1 epoch
trainguard run --script train_deberta_3class.py --quick-test
```

## Example output

TrainGuard v1.0 — Pre-flight checks
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ CUDA 13.0 detected — compatible with PyTorch 2.11
✓ GPU confirmed — H100 NVL 80GB
✓ Dataset loaded — 921,127 rows
⚠ Duplicate rows detected — 28,432 (removing before run)
⚠ Conflicting labels detected — 312 (resolving in favour of ai)
✓ Class weights [0.853, 1.0] — within safe range
✓ Estimated cost — £3.80 / ~42 minutes on H100
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
All checks passed. Starting training run...

## Status

Early build. If you've burned money on a silent training failure, 
I'd like to hear what broke. Open an issue or email [your email].

⭐ Star to follow progress
