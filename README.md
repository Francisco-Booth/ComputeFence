# ComputeFence

Pre-flight validation for GPU training runs on rented infrastructure.

## Install

pip install computefence

## Usage

computefence doctor
computefence doctor --dataset train.csv

## What it checks

- CUDA and GPU availability
- HuggingFace cache volume path
- Accelerate GPU count vs config
- Dataset duplicates and missing values

## What it does not yet check

- Training script correctness
- Model architecture compatibility
- Learning rate or hyperparameter safety
- Runtime monitoring during the job