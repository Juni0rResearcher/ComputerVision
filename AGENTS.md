# AGENTS.md - Codex

## Mission

Implement and run a controlled experiment comparing:

- Baseline: CrossEntropy
- Variant: Label Smoothing (`epsilon = 0.1`)

The goal is reproducible training, logging, and analysis outputs for report and
paper drafting.

---

## Allowed Changes

You may create and/or modify:

- `train.py`
- `loss.py`
- `model.py`
- `analyze_results.py`
- documentation files (`README.md`, `AGENTS.md`)
- plotting utilities and t-SNE computation used by analysis
- generated outputs under `results/` and `report.md`

---

## Forbidden Changes

Do not change the experimental protocol unless explicitly instructed:

- dataset choice
- train/test split
- augmentation pipeline
- number of epochs
- learning rate schedule
- seed list

---

## Experiments

Run exactly 3 seeds per experiment:

- `42`
- `0`
- `17`

Experiments:

- `baseline`
- `ls`

---

## Training Rules

Use:

- ResNet18 for CIFAR-10 (`3x3` conv1, stride 1, no maxpool)
- simple augmentations only:
  - `RandomCrop`
  - `RandomHorizontalFlip`

---

## Logging Requirements

Every epoch must log:

- `experiment`
- `seed`
- `epoch`
- `train_loss`
- `val_loss`
- `accuracy`

Also log run config and seed metadata.

---

## Visualization Requirements

Generate:

- `validation_loss.png`

Plot:

- epoch vs validation loss
- averaged across seeds for each experiment

Optional:

- `train_loss.png`

---

## t-SNE Rules

Compute t-SNE once per experiment using:

- best validation checkpoint
- 1000 test images

Output:

- `tsne_baseline.png`
- `tsne_ls.png`

---

## Output Directory

- `results/`

---

## Execution Order

1. Train baseline.
2. Train label smoothing.
3. Aggregate results.
4. Generate plots and t-SNE.
5. Save metrics and report.

---

## Performance Target

Each experiment should complete in under 10 minutes on a single GPU when
hardware permits.

---

## Reproducibility

All runs must:

- set random seed
- log seed
- log config
