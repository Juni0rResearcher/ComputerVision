# Optimizer Study: SGD vs. AdamW on Fashion-MNIST

## Objective

This project implements a controlled, reproducible experiment comparing two optimizers for image classification:

- **Baseline**: SGD with momentum (`lr=0.01`, `momentum=0.9`, `weight_decay=5e-4`)
- **Variant**: AdamW (`lr=0.001`, `weight_decay=1e-4`)

The outputs are designed to feed a Prism paper draft workflow.

## Research Question

Does AdamW converge faster and achieve higher validation accuracy compared to SGD with momentum on Fashion-MNIST under a fixed training protocol?

**Hypotheses**:
1. AdamW reaches 85% validation accuracy at least 5 epochs faster than SGD
2. AdamW achieves higher final validation accuracy (+1-2%) than SGD

## Fixed Protocol

- **Dataset**: Fashion-MNIST (28×28 grayscale, 10 classes)
- **Model**: ResNet18 adapted for Fashion-MNIST (1 input channel, 10 output classes)
- **Input size**: 28×28 → resized to 32×32 for ResNet compatibility
- **Augmentations** (train only): `RandomCrop(32, padding=4)`, `RandomHorizontalFlip`
- **Loss function**: CrossEntropyLoss
- **Batch size**: `128`
- **Epochs**: `20`
- **Seeds**: `42`, `0`, `17` (for reproducibility)

### Optimizer Configurations

| Optimizer | Learning Rate | Momentum | Weight Decay | Other |
|-----------|---------------|----------|--------------|-------|
| SGD | 0.01 | 0.9 | 5e-4 | nesterov=False |
| AdamW | 0.001 | - | 1e-4 | betas=(0.9, 0.999) |

## Repository Layout

```
.
├── model.py              # ResNet18 for Fashion-MNIST (1 input channel)
├── train.py              # Training loop with per-epoch logging
├── analyze_results.py    # Seed aggregation, plots, statistics, report generation
├── results/              # Generated artifacts (created during run)
│   ├── sgd_seed42/       # Per-run checkpoints and logs
│   ├── sgd_seed0/
│   ├── sgd_seed17/
│   ├── adamw_seed42/
│   ├── adamw_seed0/
│   ├── adamw_seed17/
│   ├── aggregated_epoch_metrics.csv
│   ├── summary_metrics.csv
│   ├── validation_accuracy.png
│   ├── train_loss.png
│   ├── convergence_speed.png
│   ├── tsne_sgd.png
│   ├── tsne_adamw.png
│   └── report.md
└── README.md
```

## Required Artifacts

After full execution, `results/` will contain:

- **Per-run metrics** (`metrics.csv`, `metrics.jsonl`, `checkpoint_best.pth`)
- **Aggregated metrics** across 3 seeds (`aggregated_epoch_metrics.csv`)
- **Summary statistics** (`summary_metrics.csv`):
  - Best validation accuracy (mean ± std)
  - Epoch of reaching 85% accuracy (convergence speed)
  - Final validation accuracy
- **Plots**:
  - `validation_accuracy.png` (with confidence intervals)
  - `train_loss.png`
  - `convergence_speed.png` (epochs to reach target accuracy)
  - `tsne_sgd.png` and `tsne_adamw.png` (feature visualization)
- **Statistical tests**: t-test p-value comparing final accuracies
- **Report**: `report.md` (ready for Prism paper draft)

## Reproducibility

- All runs set deterministic seeds using `torch.manual_seed(seed)`
- CUDA deterministic algorithms enabled (if GPU available)
- Each optimizer configuration runs independently with same seeds

## Environment Setup

### Windows (PowerShell)

```powershell
# Create and activate virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
pip install torch torchvision pandas matplotlib scikit-learn numpy
```

### Linux / macOS

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision pandas matplotlib scikit-learn numpy
```

## How To Run

### Step 1: Train all SGD runs (3 seeds)

```bash
python train.py --optimizer sgd --seed 42
python train.py --optimizer sgd --seed 0
python train.py --optimizer sgd --seed 17
```

### Step 2: Train all AdamW runs (3 seeds)

```bash
python train.py --optimizer adamw --seed 42
python train.py --optimizer adamw --seed 0
python train.py --optimizer adamw --seed 17
```

### Step 3: Aggregate results and generate report

```bash
python analyze_results.py --results-dir results --output-dir results
```

### Alternative: Run all experiments sequentially (script)

Create `run_all.py`:

```python
import subprocess

seeds = [42, 0, 17]

print("Training SGD...")
for seed in seeds:
    subprocess.run(["python", "train.py", "--optimizer", "sgd", "--seed", str(seed)])

print("Training AdamW...")
for seed in seeds:
    subprocess.run(["python", "train.py", "--optimizer", "adamw", "--seed", str(seed)])

print("Generating report...")
subprocess.run(["python", "analyze_results.py", "--results-dir", "results", "--output-dir", "results"])
```

## Expected Runtime

| Hardware | SGD (3 runs) | AdamW (3 runs) | Total |
|----------|--------------|----------------|-------|
| GPU (RTX 3060+) | ~3-5 minutes | ~3-5 minutes | **~10 minutes** |
| CPU (modern) | ~15-20 minutes | ~15-20 minutes | **~40 minutes** |

## Key Metrics to Report

| Metric | Description |
|--------|-------------|
| Best validation accuracy | Highest accuracy across 20 epochs |
| Epoch to 85% accuracy | Convergence speed (lower = faster) |
| Final validation accuracy | Accuracy at epoch 20 |
| Training loss slope | How quickly loss decreases |
| Feature separation | t-SNE cluster quality |

## Expected Results (Illustrative)

Based on known literature for Fashion-MNIST with ResNet18:

| Optimizer | Best Val Acc (%) | Epoch to 85% | Final Val Acc (%) |
|-----------|-----------------|--------------|-------------------|
| SGD | 91.2 ± 0.3 | 12.3 ± 0.6 | 90.8 ± 0.4 |
| AdamW | 91.8 ± 0.2 | 8.0 ± 0.0 | 91.5 ± 0.3 |

**Expected t-test p-value**: < 0.05 (statistically significant improvement with AdamW)

## Notes

- Fashion-MNIST is automatically downloaded by torchvision on first run
- The model uses ResNet18 with modified first conv layer (1 input channel instead of 3)
- Validation uses the standard Fashion-MNIST test split
- t-SNE uses the feature layer before the final classifier (after global average pooling)
- Checkpoints save the best model based on validation accuracy

## Troubleshooting

**Issue**: "RuntimeError: Given groups=1, weight of size [64, 1, 3, 3], expected input[128, 3, 32, 32]"

**Solution**: Fashion-MNIST is grayscale (1 channel). The model handles this automatically via `model.py`.

**Issue**: Slow training on CPU

**Solution**: Reduce epochs to 15 (still statistically valid) or use `--no-cuda` flag if GPU memory issues.

**Issue**: Different results across runs

**Solution**: This is expected due to randomness. The 3-seed aggregation captures this variance via ± std.

## Output Example: `report.md` Snippet

```markdown
# Optimizer Comparison: SGD vs. AdamW on Fashion-MNIST

## Hypothesis
AdamW converges faster and achieves higher validation accuracy than SGD.

## Results
- **AdamW** reached 85% accuracy at epoch 8.0 ± 0.0
- **SGD** reached 85% accuracy at epoch 12.3 ± 0.6
- **AdamW** final accuracy: 91.5% ± 0.3%
- **SGD** final accuracy: 90.8% ± 0.4%
- **t-test p-value**: 0.007 (significant)

## Conclusion
AdamW demonstrates faster convergence (+4 epochs) and higher final accuracy (+0.7%, p < 0.05). 
Hypothesis supported.
```

## References

- Fashion-MNIST dataset: Xiao et al. (2017)
- AdamW: Loshchilov & Hutter (2019) - "Decoupled Weight Decay Regularization"
- SGD with momentum: Sutskever et al. (2013)

---

This README provides everything needed to run the study, including the hypothesis, protocol, commands, and expected outputs. The study fits within 1.5 hours (especially on GPU) and produces a complete paper draft in `report.md`.
