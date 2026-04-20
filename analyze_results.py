from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import build_model


TARGET_ACCURACY = 85.0
SEED_FOR_TSNE = 42


@dataclass
class RunSummary:
    optimizer: str
    seed: int
    best_accuracy: float
    final_accuracy: float
    epoch_to_85: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze optimizer study results")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--max-tsne-samples", type=int, default=2000)
    return parser.parse_args()


def parse_run_dir(path: Path) -> Tuple[str, int]:
    name = path.name
    if "_seed" not in name:
        raise ValueError(f"Unexpected run directory name: {name}")
    optimizer, seed_str = name.split("_seed", maxsplit=1)
    return optimizer.lower(), int(seed_str)


def epoch_to_threshold(epochs: np.ndarray, values: np.ndarray, threshold: float) -> float:
    if len(epochs) == 0:
        return float("nan")
    if values[0] >= threshold:
        return float(epochs[0])

    for idx in range(1, len(epochs)):
        prev_v = values[idx - 1]
        curr_v = values[idx]
        if prev_v < threshold <= curr_v:
            prev_e = float(epochs[idx - 1])
            curr_e = float(epochs[idx])
            if curr_v == prev_v:
                return curr_e
            ratio = (threshold - prev_v) / (curr_v - prev_v)
            return prev_e + ratio * (curr_e - prev_e)
    return float("nan")


def load_runs(results_dir: Path) -> Tuple[pd.DataFrame, List[RunSummary]]:
    run_frames: List[pd.DataFrame] = []
    run_summaries: List[RunSummary] = []

    for metrics_path in sorted(results_dir.glob("*_seed*/metrics.csv")):
        run_dir = metrics_path.parent
        optimizer, seed = parse_run_dir(run_dir)

        df = pd.read_csv(metrics_path)
        required_cols = {"epoch", "train_loss", "val_loss", "val_accuracy"}
        missing = required_cols.difference(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {metrics_path}: {sorted(missing)}")

        df = df.copy()
        df["optimizer"] = optimizer
        df["seed"] = seed
        run_frames.append(df)

        epochs = df["epoch"].to_numpy(dtype=float)
        val_acc = df["val_accuracy"].to_numpy(dtype=float)
        summary = RunSummary(
            optimizer=optimizer,
            seed=seed,
            best_accuracy=float(np.max(val_acc)),
            final_accuracy=float(val_acc[-1]),
            epoch_to_85=float(epoch_to_threshold(epochs, val_acc, TARGET_ACCURACY)),
        )
        run_summaries.append(summary)

    if not run_frames:
        raise FileNotFoundError(f"No metrics.csv files found under {results_dir}")

    all_runs = pd.concat(run_frames, ignore_index=True)
    return all_runs, run_summaries


def aggregate_epoch_metrics(all_runs: pd.DataFrame) -> pd.DataFrame:
    grouped = all_runs.groupby(["optimizer", "epoch"], as_index=False).agg(
        train_loss_mean=("train_loss", "mean"),
        train_loss_std=("train_loss", "std"),
        val_loss_mean=("val_loss", "mean"),
        val_loss_std=("val_loss", "std"),
        val_accuracy_mean=("val_accuracy", "mean"),
        val_accuracy_std=("val_accuracy", "std"),
    )
    return grouped.sort_values(["optimizer", "epoch"]).reset_index(drop=True)


def build_summary(run_summaries: List[RunSummary]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = [
        {
            "optimizer": s.optimizer,
            "seed": s.seed,
            "best_accuracy": s.best_accuracy,
            "final_accuracy": s.final_accuracy,
            "epoch_to_85": s.epoch_to_85,
        }
        for s in run_summaries
    ]
    per_run = pd.DataFrame(rows)

    summary = per_run.groupby("optimizer", as_index=False).agg(
        best_accuracy_mean=("best_accuracy", "mean"),
        best_accuracy_std=("best_accuracy", "std"),
        final_accuracy_mean=("final_accuracy", "mean"),
        final_accuracy_std=("final_accuracy", "std"),
        epoch_to_85_mean=("epoch_to_85", "mean"),
        epoch_to_85_std=("epoch_to_85", "std"),
    )
    return per_run, summary


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled_var = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    if pooled_var <= 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / math.sqrt(pooled_var))


def effect_size_text(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def ci95_mean_diff(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    diff = float(np.mean(x) - np.mean(y))
    sx = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    sy = float(np.std(y, ddof=1)) if len(y) > 1 else 0.0
    se = math.sqrt((sx * sx) / max(len(x), 1) + (sy * sy) / max(len(y), 1))
    margin = 1.96 * se
    return diff - margin, diff + margin


def run_stat_tests(per_run: pd.DataFrame) -> Dict[str, object]:
    sgd = per_run.loc[per_run["optimizer"] == "sgd", "final_accuracy"].to_numpy(dtype=float)
    adamw = per_run.loc[per_run["optimizer"] == "adamw", "final_accuracy"].to_numpy(dtype=float)

    if len(sgd) < 2 or len(adamw) < 2:
        return {
            "test_used": "insufficient_data",
            "shapiro_sgd_p": float("nan"),
            "shapiro_adamw_p": float("nan"),
            "t_statistic": float("nan"),
            "u_statistic": float("nan"),
            "p_value": float("nan"),
            "cohens_d": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }

    try:
        shapiro_sgd = stats.shapiro(sgd)
        shapiro_adamw = stats.shapiro(adamw)
        normal = shapiro_sgd.pvalue > 0.05 and shapiro_adamw.pvalue > 0.05
    except Exception:
        shapiro_sgd = type("obj", (), {"pvalue": float("nan")})()
        shapiro_adamw = type("obj", (), {"pvalue": float("nan")})()
        normal = False

    t_stat = float("nan")
    u_stat = float("nan")

    if normal:
        t_res = stats.ttest_ind(adamw, sgd, equal_var=False)
        p_value = float(t_res.pvalue)
        t_stat = float(t_res.statistic)
        test_used = "independent_ttest"
    else:
        u_res = stats.mannwhitneyu(adamw, sgd, alternative="two-sided")
        p_value = float(u_res.pvalue)
        u_stat = float(u_res.statistic)
        test_used = "mann_whitney_u"

    d = cohens_d(adamw, sgd)
    ci_low, ci_high = ci95_mean_diff(adamw, sgd)

    return {
        "test_used": test_used,
        "shapiro_sgd_p": float(shapiro_sgd.pvalue),
        "shapiro_adamw_p": float(shapiro_adamw.pvalue),
        "t_statistic": t_stat,
        "u_statistic": u_stat,
        "p_value": p_value,
        "cohens_d": d,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "effect_size_interpretation": effect_size_text(d),
    }


def plot_metric_with_ci(
    aggregated: pd.DataFrame,
    metric_mean_col: str,
    metric_std_col: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    for optimizer in sorted(aggregated["optimizer"].unique()):
        sub = aggregated[aggregated["optimizer"] == optimizer]
        x = sub["epoch"].to_numpy(dtype=float)
        m = sub[metric_mean_col].to_numpy(dtype=float)
        s = np.nan_to_num(sub[metric_std_col].to_numpy(dtype=float), nan=0.0)
        plt.plot(x, m, label=optimizer.upper())
        plt.fill_between(x, m - s, m + s, alpha=0.2)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_convergence(summary: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    order = ["sgd", "adamw"]
    sub = summary.set_index("optimizer").reindex(order)

    means = sub["epoch_to_85_mean"].to_numpy(dtype=float)
    stds = np.nan_to_num(sub["epoch_to_85_std"].to_numpy(dtype=float), nan=0.0)

    x = np.arange(len(order))
    plt.bar(x, means, yerr=stds, capsize=6)
    plt.xticks(x, [o.upper() for o in order])
    plt.ylabel("Epoch to 85% Validation Accuracy")
    plt.title("Convergence Speed")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def load_checkpoint_for_tsne(checkpoint_path: Path, device: torch.device) -> Dict[str, object]:
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        # Compatibility with older torch versions without weights_only.
        ckpt = torch.load(checkpoint_path, map_location=device)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unexpected checkpoint format in {checkpoint_path}")
    return ckpt


@torch.no_grad()
def extract_features_labels(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    feats: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    count = 0

    for images, labels in loader:
        images = images.to(device)
        features = model.get_features(images).cpu().numpy()
        feats.append(features)
        labels_all.append(labels.numpy())
        count += len(labels)
        if count >= max_samples:
            break

    feat_arr = np.concatenate(feats, axis=0)[:max_samples]
    label_arr = np.concatenate(labels_all, axis=0)[:max_samples]
    return feat_arr, label_arr


def generate_tsne_plot(
    results_dir: Path,
    data_dir: Path,
    optimizer: str,
    output_path: Path,
    max_samples: int,
) -> bool:
    run_dir = results_dir / f"{optimizer}_seed{SEED_FOR_TSNE}"
    ckpt_path = run_dir / "checkpoint_best.pth"
    if not ckpt_path.exists():
        return False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=10).to(device)

    ckpt = load_checkpoint_for_tsne(ckpt_path, device)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict"))
    if state is None:
        return False
    model.load_state_dict(state, strict=True)

    transform = transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )
    ds = datasets.FashionMNIST(root=str(data_dir), train=False, download=True, transform=transform)
    # Use single-process loading for broader sandbox/macOS compatibility.
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

    features, labels = extract_features_labels(model, loader, device, max_samples=max_samples)
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto", random_state=42, perplexity=30)
    emb = tsne.fit_transform(features)

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(emb[:, 0], emb[:, 1], c=labels, s=8, cmap="tab10", alpha=0.75)
    plt.title(f"t-SNE Features ({optimizer.upper()}, seed={SEED_FOR_TSNE})")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    legend = plt.legend(*scatter.legend_elements(num=10), title="Class", loc="best", fontsize=8)
    plt.gca().add_artist(legend)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return True


def fmt(v: float) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "n/a"
    return f"{v:.3f}"


def generate_report(
    output_path: Path,
    summary: pd.DataFrame,
    stats_out: Dict[str, object],
) -> None:
    by_opt = summary.set_index("optimizer")

    def val(opt: str, col: str) -> float:
        return float(by_opt.loc[opt, col]) if opt in by_opt.index else float("nan")

    diff_epochs = val("adamw", "epoch_to_85_mean") - val("sgd", "epoch_to_85_mean")

    report = f"""# Optimizer Comparison: SGD vs. AdamW on Fashion-MNIST

## Hypothesis
AdamW converges faster and achieves higher validation accuracy than SGD.

## Primary Endpoint
Final validation accuracy at epoch 20.

## Summary Table

| Optimizer | Best Val Acc (%) | Final Val Acc (%) | Epoch to 85% |
|-----------|------------------|-------------------|--------------|
| SGD | {fmt(val('sgd', 'best_accuracy_mean'))} +/- {fmt(val('sgd', 'best_accuracy_std'))} | {fmt(val('sgd', 'final_accuracy_mean'))} +/- {fmt(val('sgd', 'final_accuracy_std'))} | {fmt(val('sgd', 'epoch_to_85_mean'))} +/- {fmt(val('sgd', 'epoch_to_85_std'))} |
| AdamW | {fmt(val('adamw', 'best_accuracy_mean'))} +/- {fmt(val('adamw', 'best_accuracy_std'))} | {fmt(val('adamw', 'final_accuracy_mean'))} +/- {fmt(val('adamw', 'final_accuracy_std'))} | {fmt(val('adamw', 'epoch_to_85_mean'))} +/- {fmt(val('adamw', 'epoch_to_85_std'))} |

## Statistical Test
- Normality check (Shapiro-Wilk):
  - SGD p-value: {fmt(float(stats_out.get('shapiro_sgd_p', float('nan'))))}
  - AdamW p-value: {fmt(float(stats_out.get('shapiro_adamw_p', float('nan'))))}
- Test used: {stats_out.get('test_used', 'n/a')}
- p-value: {fmt(float(stats_out.get('p_value', float('nan'))))}
- t-statistic: {fmt(float(stats_out.get('t_statistic', float('nan'))))}
- U-statistic: {fmt(float(stats_out.get('u_statistic', float('nan'))))}
- Cohen's d (AdamW - SGD): {fmt(float(stats_out.get('cohens_d', float('nan'))))} ({stats_out.get('effect_size_interpretation', 'n/a')})
- 95% CI of mean difference (AdamW - SGD): [{fmt(float(stats_out.get('ci95_low', float('nan'))))}, {fmt(float(stats_out.get('ci95_high', float('nan'))))}]

## Convergence
AdamW reaches 85% validation accuracy {fmt(-diff_epochs)} epochs earlier than SGD (negative means slower).

## Figures
- ![Validation Accuracy](validation_accuracy.png)
- ![Train Loss](train_loss.png)
- ![Convergence Speed](convergence_speed.png)
- ![t-SNE SGD](tsne_sgd.png)
- ![t-SNE AdamW](tsne_adamw.png)

## Notes
- This comparison uses Fashion-MNIST test split as evaluation split.
- t-SNE is qualitative and can vary with hyperparameters.
"""
    output_path.write_text(report, encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_runs, run_summaries = load_runs(args.results_dir)
    aggregated = aggregate_epoch_metrics(all_runs)
    per_run, summary = build_summary(run_summaries)
    stats_out = run_stat_tests(per_run)

    for col, val in stats_out.items():
        summary[col] = val

    aggregated_path = args.output_dir / "aggregated_epoch_metrics.csv"
    summary_path = args.output_dir / "summary_metrics.csv"
    report_path = args.output_dir / "report.md"
    prism_json_path = args.output_dir / "prism_results.json"

    aggregated.to_csv(aggregated_path, index=False)
    summary.to_csv(summary_path, index=False)

    plot_metric_with_ci(
        aggregated,
        metric_mean_col="val_accuracy_mean",
        metric_std_col="val_accuracy_std",
        ylabel="Validation Accuracy (%)",
        title="Validation Accuracy vs Epoch",
        output_path=args.output_dir / "validation_accuracy.png",
    )
    plot_metric_with_ci(
        aggregated,
        metric_mean_col="train_loss_mean",
        metric_std_col="train_loss_std",
        ylabel="Train Loss",
        title="Train Loss vs Epoch",
        output_path=args.output_dir / "train_loss.png",
    )
    plot_convergence(summary, args.output_dir / "convergence_speed.png")

    tsne_sgd_ok = False
    tsne_adamw_ok = False
    try:
        tsne_sgd_ok = generate_tsne_plot(
            args.results_dir,
            args.data_dir,
            optimizer="sgd",
            output_path=args.output_dir / "tsne_sgd.png",
            max_samples=args.max_tsne_samples,
        )
    except Exception as exc:
        print(f"warning: failed to generate tsne_sgd.png: {exc}")

    try:
        tsne_adamw_ok = generate_tsne_plot(
            args.results_dir,
            args.data_dir,
            optimizer="adamw",
            output_path=args.output_dir / "tsne_adamw.png",
            max_samples=args.max_tsne_samples,
        )
    except Exception as exc:
        print(f"warning: failed to generate tsne_adamw.png: {exc}")

    if not tsne_sgd_ok:
        print("warning: skipped t-SNE for SGD (checkpoint missing or extraction failed)")
    if not tsne_adamw_ok:
        print("warning: skipped t-SNE for AdamW (checkpoint missing or extraction failed)")

    prism_payload = {
        "hypothesis_test": {
            "metric": "final_accuracy",
            "sgd_mean": float(summary.loc[summary["optimizer"] == "sgd", "final_accuracy_mean"].iloc[0])
            if (summary["optimizer"] == "sgd").any()
            else float("nan"),
            "sgd_std": float(summary.loc[summary["optimizer"] == "sgd", "final_accuracy_std"].iloc[0])
            if (summary["optimizer"] == "sgd").any()
            else float("nan"),
            "adamw_mean": float(summary.loc[summary["optimizer"] == "adamw", "final_accuracy_mean"].iloc[0])
            if (summary["optimizer"] == "adamw").any()
            else float("nan"),
            "adamw_std": float(summary.loc[summary["optimizer"] == "adamw", "final_accuracy_std"].iloc[0])
            if (summary["optimizer"] == "adamw").any()
            else float("nan"),
            "t_statistic": float(stats_out.get("t_statistic", float("nan"))),
            "u_statistic": float(stats_out.get("u_statistic", float("nan"))),
            "p_value": float(stats_out.get("p_value", float("nan"))),
            "significant": bool(float(stats_out.get("p_value", 1.0)) < 0.05),
            "effect_size_cohens_d": float(stats_out.get("cohens_d", float("nan"))),
            "effect_size_interpretation": stats_out.get("effect_size_interpretation", "n/a"),
            "test_used": stats_out.get("test_used", "n/a"),
            "ci95_low": float(stats_out.get("ci95_low", float("nan"))),
            "ci95_high": float(stats_out.get("ci95_high", float("nan"))),
        },
        "convergence_speed": {
            "sgd_epochs_to_85": float(summary.loc[summary["optimizer"] == "sgd", "epoch_to_85_mean"].iloc[0])
            if (summary["optimizer"] == "sgd").any()
            else float("nan"),
            "adamw_epochs_to_85": float(
                summary.loc[summary["optimizer"] == "adamw", "epoch_to_85_mean"].iloc[0]
            )
            if (summary["optimizer"] == "adamw").any()
            else float("nan"),
        },
    }
    prism_payload["convergence_speed"]["difference_epochs"] = (
        prism_payload["convergence_speed"]["adamw_epochs_to_85"]
        - prism_payload["convergence_speed"]["sgd_epochs_to_85"]
    )

    prism_json_path.write_text(json.dumps(prism_payload, indent=2), encoding="utf-8")
    generate_report(report_path, summary, stats_out)

    print(f"saved: {aggregated_path}")
    print(f"saved: {summary_path}")
    print(f"saved: {report_path}")
    print(f"saved: {prism_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
