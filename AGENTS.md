# AGENTS.md – Agent Instructions for Optimizer Study (SGD vs. AdamW)

This document defines the **roles, responsibilities, and handoffs** for three AI agents working autonomously on the Fashion-MNIST optimizer comparison study.

**Total time budget**: 1.5 hours  
**Output**: PDF paper draft + GitHub repo link

---

## Agent Roles Overview

| Agent | Primary Tool | Responsibility |
|-------|--------------|----------------|
| **Codex** | GitHub Copilot / Code generation | Write and validate all Python scripts (`model.py`, `train.py`, `analyze_results.py`) |
| **Cursor** | Agentic coding & execution | Run experiments, handle errors, orchestrate training loop across 6 runs |
| **Prism** | Structured experiment tracking | Define hypothesis, aggregate results, generate statistical report, produce PDF |

---

## Agent 1: Codex

### Mission
Generate **correct, reproducible, deterministic** Python code for the entire experiment pipeline.

### Input Receives
- `README.md` (experiment protocol)
- Research question: *"Does AdamW converge faster and achieve higher accuracy than SGD on Fashion-MNIST?"*

### Output Produces
Must write the following files to the repository root:

#### 1. `model.py`
```python
# Requirements:
- ResNet18 adapted for Fashion-MNIST (1 input channel, 10 classes)
- First conv layer: nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
- Remove maxpool (CIFAR/Fashion-MNIST style)
- Method: get_features(x) returns penultimate layer features (before FC)
```

#### 2. `train.py`
```python
# Requirements:
- CLI arguments: --optimizer [sgd|adamw], --seed INT
- Load Fashion-MNIST (train + test splits)
- Transforms: RandomCrop(32, padding=4), RandomHorizontalFlip, Normalize
- Model: ResNet18 from model.py
- Optimizer configs:
    - SGD: lr=0.01, momentum=0.9, weight_decay=5e-4
    - AdamW: lr=0.001, weight_decay=1e-4, betas=(0.9, 0.999)
- Loss: CrossEntropyLoss
- Scheduler: MultiStepLR(milestones=[10, 15], gamma=0.1)
- Epochs: 20, batch_size: 128
- Log per epoch: train_loss, val_loss, val_accuracy
- Save checkpoint: results/{optimizer}_seed{seed}/checkpoint_best.pth
- Save metrics: metrics.csv (epoch, train_loss, val_loss, val_accuracy)
- Save metadata: config.json (optimizer, seed, lr, etc.)
- Set deterministic: torch.manual_seed(seed), torch.backends.cudnn.deterministic=True
```

#### 3. `analyze_results.py`
```python
# Requirements:
- CLI: --results-dir, --output-dir
- Load all results/{optimizer}_seed*/metrics.csv
- Aggregate per optimizer across 3 seeds: mean, std per epoch
- Compute:
    - Best validation accuracy (mean ± std)
    - Epoch to reach 85% accuracy (linear interpolation)
    - Final accuracy at epoch 20
- Generate plots (matplotlib):
    - validation_accuracy.png (line + confidence interval fill)
    - train_loss.png
    - convergence_speed.png (bar chart with error bars)
- t-SNE (sklearn.manifold.TSNE):
    - Load best checkpoints for seed=42 (both optimizers)
    - Extract features using model.get_features()
    - Generate tsne_sgd.png, tsne_adamw.png
- Statistical test: independent t-test (final accuracies across 3 seeds)
- Save summary_metrics.csv
- Save aggregated_epoch_metrics.csv
- Generate report.md (Markdown template with results filled in)
```

#### 4. `run_all.py` (optional orchestrator)
```python
# Sequentially runs all 6 experiments + analysis
# Includes error handling and retry logic (max 2 retries per run)
```

### Constraints
- No hardcoded paths (use `pathlib.Path`)
- All randomness controlled by seed
- GPU support if available, else CPU (automatic)
- Code must be self-contained (no external data download except torchvision)

### Success Criteria
- [ ] All scripts run without syntax errors
- [ ] `train.py --optimizer sgd --seed 42` produces valid metrics CSV
- [ ] `analyze_results.py` generates 4+ plots and `report.md`
- [ ] Code passes `flake8` (ignoring line length)

---

## Agent 2: Cursor

### Mission
Execute the experiment **end-to-end** in automatic mode, handling runtime errors, resource constraints, and log collection.

### Input Receives
- All Python scripts from Codex
- `README.md` (execution order)
- Access to GPU/CPU environment

### Output Produces
- **6 completed training runs** (3 seeds × 2 optimizers)
- **Aggregated results** (plots, CSV files, t-SNE)
- **GitHub repository** with all code + results
- **Log file** (`execution.log`) with timestamps and any errors

### Execution Steps (in order)

#### Phase 1: Environment Setup (5 min)
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install torch torchvision pandas matplotlib scikit-learn numpy

# Verify GPU available (optional)
python -c "import torch; print(torch.cuda.is_available())"
```

#### Phase 2: Code Validation (5 min)
```bash
# Run syntax checks
python -m py_compile model.py train.py analyze_results.py

# Test single run (1 epoch) to catch errors
python train.py --optimizer sgd --seed 42 --epochs 1  # dry run
```

#### Phase 3: Full Experiment Execution (40–60 min)
```bash
# Launch orchestrator with retry logic
python run_all.py 2>&1 | tee execution.log
```

**Error handling rules**:
- If CUDA out of memory → fallback to CPU
- If download timeout → retry up to 3 times
- If checkpoint save fails → continue but log warning
- If any seed fails → retry once, then mark as failed and continue others

**Progress tracking**:
- Print `[timestamp] Starting SGD seed 42...`
- Print `[timestamp] Completed SGD seed 42 (accuracy: 91.2%)`
- Save intermediate metrics after each run

#### Phase 4: Analysis & Report Generation (5 min)
```bash
python analyze_results.py --results-dir results --output-dir results
```

#### Phase 5: GitHub Push (5 min)
```bash
# Initialize repo
git init
git add .
git commit -m "Automatic experiment: SGD vs AdamW on Fashion-MNIST"

# Create remote (assumes GITHUB_TOKEN in environment)
gh repo create optimizer-study --public --source=. --remote=origin --push
```

#### Phase 6: PDF Generation (5 min)
```bash
# Convert report.md to PDF using pandoc
pandoc results/report.md -o results/report.pdf --pdf-engine=xelatex
```

### Fallback Behaviors
| Issue | Action |
|-------|--------|
| GPU unavailable | Use CPU (add `--no-cuda` flag automatically) |
| Disk space < 1GB | Skip checkpoint saving, keep only metrics |
| t-SNE timeout (>30 sec) | Skip t-SNE generation, note in report |
| Pandoc missing | Save report.md only, note PDF not generated |

### Success Criteria
- [ ] All 6 runs complete (≤5 failures acceptable if documented)
- [ ] `results/aggregated_epoch_metrics.csv` exists and has data
- [ ] `results/validation_accuracy.png` shows 2 lines with confidence bands
- [ ] GitHub repo contains all code + results
- [ ] Total execution time ≤ 1.5 hours

---

## Agent 3: Prism

### Mission
Provide **structured experiment tracking**, **hypothesis testing**, and **paper draft generation** with statistical rigor.

### Input Receives
- `results/report.md` (from Cursor)
- `results/summary_metrics.csv`
- `results/aggregated_epoch_metrics.csv`
- All plots (PNG files)

### Output Produces

#### 1. Prism Configuration (`prism_config.yaml`)
```yaml
experiment:
  name: "SGD vs AdamW on Fashion-MNIST"
  hypothesis: |
    H0: No difference between SGD and AdamW validation accuracy
    H1: AdamW achieves higher validation accuracy than SGD
  metrics:
    - final_accuracy
    - epoch_to_85_percent
    - best_accuracy
  statistical_test: independent_ttest
  alpha: 0.05
  effect_size: cohens_d
```

#### 2. Structured Results (`prism_results.json`)
```json
{
  "hypothesis_test": {
    "metric": "final_accuracy",
    "sgd_mean": 90.8,
    "sgd_std": 0.4,
    "adamw_mean": 91.5,
    "adamw_std": 0.3,
    "t_statistic": 3.21,
    "p_value": 0.007,
    "significant": true,
    "effect_size_cohens_d": 1.94
  },
  "convergence_speed": {
    "sgd_epochs_to_85": 12.3,
    "adamw_epochs_to_85": 8.0,
    "difference_epochs": -4.3
  }
}
```

#### 3. Paper Draft (`paper_draft.md` – ready for PDF)

**Template** (Prism fills in `{{variables}}`):

```markdown
# Optimizer Comparison on Fashion-MNIST: SGD vs. AdamW

## Abstract
{{auto_generated_abstract}}

## 1. Introduction
Deep learning optimization remains critical for computer vision tasks. This study compares 
SGD with momentum against AdamW on Fashion-MNIST classification.

**Research Question**: Does AdamW converge faster and achieve higher validation accuracy 
than SGD under a fixed training protocol?

## 2. Methods
### 2.1 Dataset
Fashion-MNIST (Xiao et al., 2017): 60k train / 10k test, 10 classes, 28×28 grayscale.

### 2.2 Model Architecture
ResNet18 adapted for 28×28 input (no maxpool, stride=1 first conv).

### 2.3 Training Protocol
- Epochs: 20, Batch size: 128
- Learning rate schedule: MultiStepLR [10, 15], gamma=0.1
- Augmentations: RandomCrop(32,4), RandomHorizontalFlip

### 2.4 Optimizer Configurations
| Optimizer | LR | Momentum | Weight Decay |
|-----------|----|----------|---------------|
| SGD | 0.01 | 0.9 | 5e-4 |
| AdamW | 0.001 | - | 1e-4 |

### 2.5 Statistical Analysis
Independent t-test (α=0.05) on final validation accuracies across 3 random seeds (42,0,17).

## 3. Results

### 3.1 Validation Accuracy
{{validation_accuracy_plot}}

| Optimizer | Best Val Acc (%) | Final Val Acc (%) | Epoch to 85% |
|-----------|-----------------|-------------------|---------------|
| SGD | {{sgd_best_mean}} ± {{sgd_best_std}} | {{sgd_final_mean}} ± {{sgd_final_std}} | {{sgd_epochs_to_85}} |
| AdamW | {{adamw_best_mean}} ± {{adamw_best_std}} | {{adamw_final_mean}} ± {{adamw_final_std}} | {{adamw_epochs_to_85}} |

### 3.2 Convergence Speed
{{convergence_speed_plot}}

AdamW reached 85% accuracy **{{difference_epochs}} epochs earlier** than SGD.

### 3.3 Statistical Test
- t-statistic: {{t_statistic}}
- p-value: {{p_value}}
- Effect size (Cohen's d): {{cohens_d}} ({{effect_size_interpretation}})

### 3.4 Feature Visualization
{{tsne_comparison_plot}}

AdamW shows more separated class clusters in t-SNE space.

## 4. Discussion
{{auto_generated_discussion}}

## 5. Conclusion
{{auto_generated_conclusion}}

## 6. Reproducibility
Code and data available at: {{github_repo_link}}
All seeds and hyperparameters logged.

## References
- Loshchilov & Hutter (2019). Decoupled Weight Decay Regularization. ICLR.
- Xiao et al. (2017). Fashion-MNIST: A Novel Image Dataset. arXiv:1708.07747.
```

#### 4. Final PDF (`paper.pdf`)
- Convert `paper_draft.md` to PDF using `pandoc` or `weasyprint`
- Include all plots inline
- Add page numbers and section numbering

### Constraints
- Must validate assumptions (normality via Shapiro–Wilk test before t-test)
- If normality fails → use Mann–Whitney U test automatically
- Report must include confidence intervals (95% CI)
- No manual editing of report allowed (fully automatic)

### Success Criteria
- [ ] `prism_config.yaml` defines experiment
- [ ] `prism_results.json` contains all statistical outputs
- [ ] `paper_draft.md` has all `{{variables}}` replaced with actual numbers
- [ ] PDF is generated and readable
- [ ] Report includes p-value, effect size, and interpretation

---

## Handoff Protocol

```
Codex ──(writes code)──> Cursor ──(executes, gets results)──> Prism ──(generates paper)──> PDF
```

### Handoff Checklist

| From | To | Artifacts Passed | Validation |
|------|----|--------------------|-------------|
| Codex | Cursor | `model.py`, `train.py`, `analyze_results.py`, `run_all.py` | All scripts run without syntax errors |
| Cursor | Prism | `results/` folder (CSVs, plots), `execution.log`, GitHub link | `summary_metrics.csv` exists and has 6 rows |
| Prism | Final | `paper.pdf`, `prism_results.json` | PDF contains all figures and statistical test results |

### Error Recovery Between Agents

**If Codex fails to generate valid code**:
- Cursor requests regeneration with specific error message
- Codex fixes and resubmits (max 3 iterations)

**If Cursor fails to execute experiments** (timeout, disk full):
- Cursor logs failure reason
- Prism receives partial results + disclaimer in report

**If Prism fails to generate PDF** (missing pandoc):
- Fallback: save `paper_draft.md` only
- Add note: "PDF generation failed, Markdown available"

---

## Timeline (1.5 hours total)

| Phase | Agent | Duration | Cumulative |
|-------|-------|----------|------------|
| 0. Code generation | Codex | 15 min | 15 min |
| 1. Environment setup | Cursor | 5 min | 20 min |
| 2. Code validation | Cursor | 5 min | 25 min |
| 3. Experiment execution | Cursor | 45 min | 70 min |
| 4. Analysis & plots | Cursor | 5 min | 75 min |
| 5. GitHub push | Cursor | 5 min | 80 min |
| 6. Prism report & PDF | Prism | 10 min | 90 min |

**Buffer**: 0 minutes (tight but feasible on GPU)

---

## Environment Variables Required

```bash
# For Cursor (GitHub push)
export GITHUB_TOKEN=ghp_xxxxx

# For Prism (optional, for LaTeX PDF)
export PATH=/usr/local/texlive/bin:$PATH
```

---

## Success Definition

The study is **successful** if at the end of 1.5 hours:

1. ✅ GitHub repo contains all code + results (public link)
2. ✅ PDF paper draft exists with:
   - Abstract
   - Methods section
   - Results table with mean ± std
   - Validation accuracy plot
   - p-value and effect size
   - Conclusion
3. ✅ No human intervention required during execution
4. ✅ All 6 training runs completed (or ≤1 failure with documented reason)

---

**Agent readiness confirmed.** Proceed with Codex → Cursor → Prism pipeline.
