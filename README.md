# ComputeFence

Pre-flight validation for GPU training runs on rented infrastructure.
Built specifically for RunPod, Vast.ai, Lambda, and similar providers.

## Install

pip install computefence

## Usage

computefence doctor
computefence doctor --dataset train.csv

## What it checks

- CUDA and GPU availability
- HuggingFace cache volume path (catches the RunPod /root vs /workspace conflict)
- Accelerate GPU count vs config
- Dataset duplicates and missing values

## What it does not yet check

- Training script correctness
- Model architecture compatibility
- Learning rate or hyperparameter safety
- Runtime monitoring during the job

## Why this exists

I burned ~£1,000 on GPU training runs that failed silently. CUDA fell back to 
CPU with no error. Class weights caused loss collapse. My dataset had 28,432 
duplicate rows and 312 conflicting labels I only found during the rebuild.

Nothing existed that caught these before the job started. So I built it.
