from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Fashion-MNIST with SGD or AdamW")
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--no-cuda", action="store_true")
    return parser.parse_args()


@dataclass
class RunConfig:
    optimizer: str
    seed: int
    epochs: int
    batch_size: int
    num_workers: int
    learning_rate: float
    weight_decay: float
    momentum: float | None
    betas: Tuple[float, float] | None
    scheduler: Dict[str, object]
    device: str
    python: str
    torch_version: str
    torchvision_version: str
    cuda_available: bool
    gpu_name: str | None
    git_commit: str | None


def get_git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        return out or None
    except Exception:
        return None


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(no_cuda: bool) -> torch.device:
    if no_cuda:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_dataloaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader]:
    # Fashion-MNIST statistics.
    mean = (0.2860,)
    std = (0.3530,)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    train_ds = _load_fashion_mnist_with_retry(
        data_dir=data_dir,
        train=True,
        transform=train_transform,
    )
    val_ds = _load_fashion_mnist_with_retry(
        data_dir=data_dir,
        train=False,
        transform=eval_transform,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return train_loader, val_loader


def _load_fashion_mnist_with_retry(
    data_dir: Path,
    train: bool,
    transform: transforms.Compose,
    max_retries: int = 3,
) -> datasets.FashionMNIST:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            return datasets.FashionMNIST(
                root=str(data_dir),
                train=train,
                download=True,
                transform=transform,
            )
        except Exception as exc:
            last_error = exc
            if attempt + 1 >= max_retries:
                break
            wait_seconds = 2 ** attempt
            split = "train" if train else "test"
            print(
                f"warning: failed to download/load Fashion-MNIST {split} split "
                f"(attempt {attempt + 1}/{max_retries}): {exc}"
            )
            print(f"retrying in {wait_seconds}s...")
            time.sleep(wait_seconds)

    assert last_error is not None
    raise RuntimeError(
        f"Could not load Fashion-MNIST after {max_retries} attempts"
    ) from last_error


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        total_loss += float(loss.item()) * bs
        total_count += bs

    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        pred = logits.argmax(dim=1)
        bs = labels.size(0)
        total_loss += float(loss.item()) * bs
        total_correct += int((pred == labels).sum().item())
        total_count += bs

    avg_loss = total_loss / max(total_count, 1)
    accuracy = 100.0 * total_correct / max(total_count, 1)
    return avg_loss, accuracy


def save_config(path: Path, config: RunConfig) -> None:
    data = asdict(config)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    set_deterministic(args.seed)

    import torchvision

    device = get_device(args.no_cuda)
    run_dir = args.results_dir / f"{args.optimizer}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = build_model(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()

    if args.optimizer == "sgd":
        lr = 0.01
        weight_decay = 5e-4
        momentum = 0.9
        betas = None
        optimizer = SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=False,
        )
    else:
        lr = 0.001
        weight_decay = 1e-4
        momentum = None
        betas = (0.9, 0.999)
        optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
        )

    scheduler = MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)

    config = RunConfig(
        optimizer=args.optimizer,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        betas=betas,
        scheduler={"name": "MultiStepLR", "milestones": [10, 15], "gamma": 0.1},
        device=str(device),
        python=sys.version,
        torch_version=torch.__version__,
        torchvision_version=torchvision.__version__,
        cuda_available=torch.cuda.is_available(),
        gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        git_commit=get_git_commit(),
    )
    save_config(run_dir / "config.json", config)

    metrics_path = run_dir / "metrics.csv"
    best_ckpt_path = run_dir / "checkpoint_best.pth"

    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "val_loss", "val_accuracy"],
        )
        writer.writeheader()

        best_acc = -1.0
        start_time = time.time()

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            row = {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_accuracy": f"{val_acc:.6f}",
            }
            writer.writerow(row)
            f.flush()

            print(
                f"epoch={epoch:02d}/{args.epochs} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_accuracy={val_acc:.2f}%"
            )

            if val_acc > best_acc:
                best_acc = val_acc
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_accuracy": best_acc,
                    "config": asdict(config),
                }
                try:
                    torch.save(checkpoint, best_ckpt_path)
                except Exception as exc:
                    print(f"warning: failed to save checkpoint: {exc}")

        elapsed = time.time() - start_time
        print(f"training completed in {elapsed:.1f}s; best_val_accuracy={best_acc:.2f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
