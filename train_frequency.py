"""Frequency Regularized ResNet-18 Training Script (GTSRB Master v1)

Experiment 3: Frequency-Domain Spectral Regularization (LFC / Spectral Dropout)
Addresses high-frequency noise shortcuts by attenuating or stochastically
dropping high-frequency Fourier components during training.

Optimization Protocol:
1. Bag-of-Tricks SGD (Momentum 0.9, Nesterov, No-Bias-Decay).
2. Linear Warmup (3 epochs) + Cosine Annealing decay down to 1e-5.
3. Automatic Mixed Precision (torch.cuda.amp) for maximum GPU throughput.
4. Model checkpointing on Best Validation Macro-F1 (test set untouched).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.augmentations.frequency import FrequencyAugDataset
from src.dataset import GTSRBDataset
from src.models.baseline_cnn import BaselineResNet18, get_parameter_groups


def set_seed(seed: int = 42) -> None:
    """Freezes all random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    use_amp: bool = True,
) -> float:
    """Runs one training epoch and returns the average loss."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    pbar = tqdm(loader, desc="Training", unit="batch", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return running_loss / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> tuple[float, float, float]:
    """Evaluates the model on validation data.
    Returns: (val_loss, val_accuracy, val_macro_f1)
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_preds: list[int] = []
    all_targets: list[int] = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        targets = labels.cpu().numpy()

        all_preds.extend(preds)
        all_targets.extend(targets)

    avg_loss = running_loss / total_samples
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return avg_loss, float(acc), float(macro_f1)


def main():
    parser = argparse.ArgumentParser(description="Train Frequency-Aware ResNet-18 on GTSRB Master v1.")
    parser.add_argument("--data-root", type=str, default="german", help="Path to raw GTSRB folder")
    parser.add_argument("--meta-dir", type=str, default="data/master_v1/metadata", help="Path to master manifests")
    parser.add_argument("--output-dir", type=str, default="checkpoints/frequency_resnet18", help="Output directory")
    parser.add_argument("--mode", type=str, default="dropout", choices=["dropout", "lfc", "blend"], help="Frequency transform mode")
    parser.add_argument("--radius", type=int, default=14, help="Spectral radius cut-off")
    parser.add_argument("--severity", type=float, default=3.0, help="Spectral dropout severity")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for training")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training and eval")
    parser.add_argument("--lr", type=float, default=0.05, help="Base learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--warmup-epochs", type=int, default=3, help="Warmup epochs")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes")
    parser.add_argument("--no-pretrained", action="store_true", help="Train from scratch")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training")
    args = parser.parse_args()

    set_seed(args.seed)

    data_root = Path(args.data_root).resolve()
    meta_dir = Path(args.meta_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.no_amp)

    print("=" * 75)
    print("FREQUENCY-AWARE RESNET-18 TRAINING (Experiment 3)")
    print(f"Device        : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Mode          : {args.mode} (Radius: {args.radius}, Severity: {args.severity})")
    print(f"Mixed Prec    : {use_amp}")
    print(f"Pretrained    : {not args.no_pretrained}")
    print(f"Batch Size    : {args.batch_size}")
    print(f"Epochs        : {args.epochs} (Warmup: {args.warmup_epochs})")
    print(f"Base LR       : {args.lr}")
    print(f"Seed          : {args.seed}")
    print(f"Checkpoint Dir: {out_dir}")
    print("=" * 75)

    # 1. Datasets & Loaders
    train_manifest = meta_dir / "train.csv"
    val_manifest = meta_dir / "val.csv"

    train_dataset = FrequencyAugDataset(
        manifest_path=train_manifest,
        data_root=data_root,
        target_size=64,
        mode=args.mode,
        radius=args.radius,
        severity=args.severity,
        is_training=True,
    )
    val_dataset = GTSRBDataset(val_manifest, data_root, target_size=64, is_training=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Train samples : {len(train_dataset):,} ({len(train_loader)} batches/epoch)")
    print(f"Val samples   : {len(val_dataset):,} ({len(val_loader)} batches)")

    # 2. Model & Optimization
    model = BaselineResNet18(num_classes=43, pretrained=(not args.no_pretrained), zero_init_residual=True)
    model.to(device)

    param_groups = get_parameter_groups(model, weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum=0.9, nesterov=True)

    cosine_epochs = max(1, args.epochs - args.warmup_epochs)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    criterion = nn.CrossEntropyLoss()

    best_val_macro_f1 = 0.0
    history: list[dict[str, Any]] = []
    total_start_time = time.time()

    print("\nStarting Training...\n")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        if epoch <= args.warmup_epochs:
            warmup_lr = args.lr * (epoch / args.warmup_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr
            current_lr = warmup_lr
        else:
            cosine_scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device, use_amp)
        epoch_duration = time.time() - epoch_start

        is_best = val_f1 > best_val_macro_f1
        if is_best:
            best_val_macro_f1 = val_f1
            best_model_path = out_dir / "best_frequency_resnet18.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": val_f1,
                "val_acc": val_acc,
                "config": vars(args),
            }, best_model_path)

        epoch_record = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
            "duration_sec": epoch_duration,
            "is_best": is_best,
        }
        history.append(epoch_record)

        marker = " [*BEST*]" if is_best else ""
        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"LR: {current_lr:.5f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc*100:.2f}% | "
            f"Val Macro-F1: {val_f1*100:.2f}% ({epoch_duration:.1f}s){marker}"
        )

    total_duration = time.time() - total_start_time
    print("\n" + "=" * 75)
    print("TRAINING COMPLETED!")
    print(f"Total Time        : {total_duration/60:.2f} minutes")
    print(f"Best Val Macro-F1 : {best_val_macro_f1*100:.2f}%")
    print(f"Best Checkpoint   : {out_dir / 'best_frequency_resnet18.pt'}")

    history_path = out_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": vars(args),
            "best_val_macro_f1": best_val_macro_f1,
            "total_duration_minutes": total_duration / 60,
            "epochs": history,
        }, f, indent=2)
    print(f"Training History  : {history_path}")


if __name__ == "__main__":
    main()

