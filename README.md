# Label Smoothing Study on CIFAR-10

## Objective

This project implements a controlled, reproducible experiment comparing:

- Baseline: Cross-Entropy
- Variant: Cross-Entropy with Label Smoothing (`epsilon = 0.1`)

The outputs are designed to feed a Prism paper draft workflow.

## Research Question

Does label smoothing (`epsilon = 0.1`) improve validation stability and feature
structure versus plain cross-entropy under a fixed CIFAR-10 training protocol?

## Fixed Protocol

- Dataset: CIFAR-10
- Model: ResNet18 adapted for CIFAR-10 (`3x3` conv1, stride 1, no maxpool)
- Augmentations (train only): `RandomCrop(32, padding=4)`, `RandomHorizontalFlip`
- Optimizer: SGD (`lr=0.1`, `momentum=0.9`, `weight_decay=5e-4`)
- Learning rate schedule: MultiStepLR milestones `[10, 15]`, gamma `0.1`
- Batch size: `128`
- Epochs: `20`
- Seeds: exactly `42`, `0`, `17`

## Repository Layout

- `model.py`: CIFAR-10 ResNet18 with feature extraction method
- `loss.py`: loss factory for baseline and label smoothing modes
- `train.py`: deterministic training loop with per-epoch logs and checkpoints
- `analyze_results.py`: seed aggregation, plotting, t-SNE, report generation
- `results/`: generated artifacts
- `report.md`: generated experiment summary for Prism handoff

## Required Artifacts

After full execution, `results/` should contain:

- run-level metrics (`metrics.csv` and `metrics.jsonl` per run)
- `aggregated_epoch_metrics.csv`
- `summary_metrics.csv`
- `validation_loss.png`
- `train_loss.png`
- `tsne_baseline.png`
- `tsne_ls.png`
- config and seed metadata

## Reproducibility

All runs set deterministic seeds and log both seed and config for each run.

## Execution Order

1. Train baseline for each seed.
2. Train label smoothing for each seed.
3. Aggregate metrics across seeds.
4. Generate plots and t-SNE.
5. Save summary metrics and `report.md`.

## Environment Setup (Windows PowerShell)

```powershell
# Create and activate virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
pip install torch torchvision pandas matplotlib scikit-learn
```

## Environment Setup (Ubuntu)

```bash
# Install Python venv tooling if needed
sudo apt update
sudo apt install -y python3-venv python3-pip

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
python -m pip install --upgrade pip
pip install torch torchvision pandas matplotlib scikit-learn
```

## How To Run

Run baseline seeds:

```bash
python train.py --mode baseline --seed 42
python train.py --mode baseline --seed 0
python train.py --mode baseline --seed 17
```

Run label smoothing seeds:

```bash
python train.py --mode ls --seed 42
python train.py --mode ls --seed 0
python train.py --mode ls --seed 17
```

Aggregate and create figures/report:

```bash
python analyze_results.py --results-dir results --output-dir results
```

## Notes

- Runtime target is under 10 minutes per experiment on a single GPU; measured
  runtime depends on hardware and environment.
- The script uses CIFAR-10 test split as validation for logged `val_loss` and
  `accuracy`.
- If `py` is not found, install Python 3.11 from python.org, reopen PowerShell,
  and rerun the setup commands.
- On Ubuntu, if `python` does not point to Python 3, use `python3` in the
  commands above.
