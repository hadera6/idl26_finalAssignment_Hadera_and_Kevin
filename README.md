# MAI/IDL SS26 — Final Assignment
## BioHealth Diagnostics: Medical Image Classification Pipeline

**Author:** MG
**Enrollment:** [enrollment number]
**Submission:** July 2026

---

## Overview

This repository contains the reconstructed, corrected, and extended machine learning
pipeline for the MAI/IDL SS26 Final Assignment. Starting from a deliberately broken
codebase, the pipeline was audited, stabilised, benchmarked across four medical imaging
datasets, extended with a green initiative (lite models), and augmented with a
transfer learning framework for scarce-data scenarios.

---

## Repository Structure

```
Code/
  train.py        — main orchestrator: loops datasets and models, saves results
  data.py         — data loading, normalisation, augmentation, train/val/test split
  fit.py          — training loop, early stopping, best weight restoration
  models.py       — AlexNet, VGG16, ResNet18 and Lite variants
  config.json     — all hyperparameters and run configuration
  weights/        — saved model checkpoints (.pth files)
  results/        — benchmark.csv and profiling.csv output

REPORT.md         — consolidated benchmark report (Tasks 1, 2, 3)
AUDIT_LOG.md      — full codebase audit with git commit references
README.md         — this file
assignment_final.pdf — task specification
```

---

## Setup

### Requirements

```
torch
torchvision
scikit-learn
numpy
```

### Install

```bash
pip install torch torchvision scikit-learn numpy
```

### Data

Download from: https://cloud.fiw.fhws.de/s/LpYa2dCW85kwdNn

Place dataset `.pt` files in `Code/data/`:

```
Code/data/
  cells_data.pt
  chest_data.pt
  lesions_data.pt
  organs_data.pt
  orgs_data.pt
```

---

## How to Run

```bash
cd Code
python train.py
```

The pipeline reads `config.json`, trains each dataset/model combination, and saves
results to `results/benchmark.csv` and `results/profiling.csv`. Already-trained
models are skipped automatically.

To run a specific dataset only:

```json
"RUN_ONLY": ["organs"]
```

---

## Config Reference

### Global defaults

```json
"BATCH_SIZE": 64,  "EPOCHS": 20,  "LEARNING_RATE": 0.001,
"DROP_RATE": 0.5,  "PATIENCE": 10,  "SEED": 0
```

### Per-dataset keys

Each dataset block in `DATASETS` can override any global key:

```json
"cells": {
    "MODELS":        ["ResNet18"],   — models to train (list, one or more)
    "CHANNELS":      3,              — 3=RGB, 1=grayscale
    "NUM_CLASSES":   8,              — number of output classes
    "EPOCHS":        30,             — overrides global
    "LEARNING_RATE": 0.0005,
    "DROP_RATE":     0.3,
    "USE_SCHEDULER": true,           — cosine LR decay
    "NO_RESTORE":    false,          — keep final weights instead of best
    "AUGMENT":       false,          — orientation-safe augmentation
    "BATCH_SIZE":    64,
    "WEIGHT_DECAY":  0.0
}
```

### Available models

```
Full:   AlexNet  VGG16  ResNet18
Lite:   AlexNetLite  VGG16Lite  ResNet18Lite
```

---

## Transfer Learning

```json
"organs": {
    "MODELS":        ["ResNet18"],
    "TRANSFER_FROM": "orgs",         — source dataset checkpoint to load
    "FROZEN_STAGES": 0,              — 0=finetune, N=freeze N backbone stages
    "LEARNING_RATE": 0.0001
}
```

---

## Output Files

```
results/benchmark.csv   — dataset, model, training_state, test_acc, precision, recall, f1
results/profiling.csv   — dataset, model, train_time_s, latency_ms, peak_memory_mb
weights/                — {dataset}_{model}[_{training_state}].pth
```

---

## Results Summary

| Dataset | Best Model | Accuracy | Target | Status |
|---------|-----------|----------|--------|--------|
| cells | VGG16 | 98.51% | 90% | ✅ |
| chest | AlexNet | 91.03% | 87% | ✅ |
| lesions | ResNet18 | 75.61% | 67% | ✅ |
| orgs | ResNet18 | 92.45% | 83% | ✅ |
| organs (transfer) | ResNet18 freeze_2+aug | 70.50% | 40% | ✅ |

Full analysis and discussion: see `REPORT.md`.
Full audit trail: see `AUDIT_LOG.md`.
