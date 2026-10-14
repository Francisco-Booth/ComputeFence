# Failures Caught by ComputeFence

Real configuration mistakes caught before GPU billing started.

---

## 1 — Silent CPU fallback on Vast.ai

**What happened:** CUDA fell back to CPU silently. Job ran for six hours at 24 seconds per iteration instead of 0.4 seconds. No error message. No warning.

**Cost:** ~£1,000 in wasted compute.

**Check that catches it:** GPU and CUDA visibility check.

---

## 2 — HuggingFace cache writing to ephemeral storage

**What happened:** Model weights cached to /root/.cache which is wiped on pod shutdown. Every fresh pod re-downloaded the same weights at full cost.

**Check that catches it:** HF_HOME persistence check.

---

## 3 — Accelerate config mismatch on single GPU pod

**What happened:** Accelerate config specified num_processes: 2 on a single GPU pod. Training launched, appeared to run, produced garbage output. No error thrown.

**Check that catches it:** Accelerate GPU count check (BLOCKER).

---

## 4 — Disk exhaustion from OLMo-7B checkpoints

**What happened:** 50GB Vast.ai instance filled with checkpoints mid-training. Job died silently when disk ran out.

**Check that catches it:** Disk space headroom check (warns below 20GB, blocks below 5GB).

---

## 5 — Checkpoint output directory on ephemeral storage

**What happened:** Training script saved checkpoints to /root/checkpoints. Pod stopped. All checkpoints lost. HF_HOME was set correctly but the output directory was not.

**Check that catches it:** --output-dir persistence check (BLOCKER).

---

## 6 — Dataset with 28,432 duplicate rows and 312 conflicting labels

**What happened:** Loss collapsed to 0.693 immediately on every training run. Several days lost diagnosing what appeared to be a model problem.

**Check that catches it:** Dataset integrity check (--dataset flag).

---

## 7 — PyTorch not installed before job launch

**What happened:** Training script failed immediately on a fresh pod where PyTorch had not been installed. Billed for the time spent diagnosing.

**Check that catches it:** PyTorch installation check with exact pip install fix command.

---

*ComputeFence runs these checks in under 30 seconds before your job starts.*