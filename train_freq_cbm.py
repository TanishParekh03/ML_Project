"""
Production Concept Bottleneck Model (CBM) Training Script (GTSRB Master v1)

Experiment 6: Frequency-Aware Concept Bottleneck Model
- Trains ResNet-18 with 4-axis Vienna/StVO concept heads (Shape, Colour, Border, Symbol Category).
- Supervised jointly on class identity and legal concept primitives.
- Exact same optimization protocol as Baseline (Exp 1), AugMix (Exp 2), and Frequency (Exp 3):
    * SGD (Momentum 0.9, Nesterov, No-Bias-Decay)
    * 3-epoch Linear Warmup + Cosine Annealing decay down to 1e-5
    * Automatic Mixed Precision (torch.cuda.amp)
    * Validation Macro-F1 model selection (checkpoint: best_freq_cbm_resnet18.pt)
    * Comprehensive logging of task loss, concept losses, and concept accuracies.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import GTSRBDataset
from src.augmentations.frequency import FrequencyAugDataset
from src.models.baseline_cnn import get_parameter_groups
from src.models.cbm import CBMLoss, ConceptBottleneckModel, get_concept_lookup_tensors


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
    criterion: CBMLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    concept_luts: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    use_amp: bool = True,
) -> Tuple[float, Dict[str, float]]:
    """Runs one training epoch and returns the average losses."""
    model.train()
    running_total_loss = 0.0
    running_task_loss = 0.0
    running_concept_loss = 0.0
    running_aux_loss = 0.0
    total_samples = 0

    shape_lut, colour_lut, border_lut, symcat_lut = concept_luts

    pbar = tqdm(loader, desc="Training", unit="batch", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        # Vectorized ground-truth concept indexing on GPU
        target_shape = shape_lut[labels]
        target_colour = colour_lut[labels]
        target_border = border_lut[labels]
        target_symcat = symcat_lut[labels]

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss, metrics = criterion(
                outputs,
                target_class=labels,
                target_shape=target_shape,
                target_colour=target_colour,
                target_border=target_border,
                target_symcat=target_symcat,
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = images.size(0)
        running_total_loss += loss.item() * bs
        running_task_loss += metrics["loss_task"] * bs
        running_concept_loss += metrics["loss_concepts"] * bs
        running_aux_loss += metrics["loss_aux"] * bs
        total_samples += bs

        pbar.set_postfix({
            "total": f"{loss.item():.3f}",
            "task": f"{metrics['loss_task']:.3f}",
            "cpt": f"{metrics['loss_concepts']:.3f}",
        })

    avg_metrics = {
        "train_loss": running_total_loss / total_samples,
        "train_loss_task": running_task_loss / total_samples,
        "train_loss_concepts": running_concept_loss / total_samples,
        "train_loss_aux": running_aux_loss / total_samples,
    }
    return avg_metrics["train_loss"], avg_metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: CBMLoss,
    device: torch.device,
    concept_luts: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    use_amp: bool = True,
) -> Dict[str, float]:
    """Evaluates the model on validation data, recording class and concept metrics."""
    model.eval()
    running_total_loss = 0.0
    running_task_loss = 0.0
    running_concept_loss = 0.0
    total_samples = 0

    all_preds_class: list[int] = []
    all_targets_class: list[int] = []

    all_preds_shape: list[int] = []
    all_targets_shape: list[int] = []

    all_preds_colour: list[int] = []
    all_targets_colour: list[int] = []

    all_preds_border: list[int] = []
    all_targets_border: list[int] = []

    all_preds_symcat: list[int] = []
    all_targets_symcat: list[int] = []

    shape_lut, colour_lut, border_lut, symcat_lut = concept_luts

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        target_shape = shape_lut[labels]
        target_colour = colour_lut[labels]
        target_border = border_lut[labels]
        target_symcat = symcat_lut[labels]

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss, metrics = criterion(
                outputs,
                target_class=labels,
                target_shape=target_shape,
                target_colour=target_colour,
                target_border=target_border,
                target_symcat=target_symcat,
            )

        bs = images.size(0)
        running_total_loss += loss.item() * bs
        running_task_loss += metrics["loss_task"] * bs
        running_concept_loss += metrics["loss_concepts"] * bs
        total_samples += bs

        # Class predictions
        pred_class = outputs["logits"].argmax(dim=1).cpu().tolist()
        all_preds_class.extend(pred_class)
        all_targets_class.extend(labels.cpu().tolist())

        # Concept predictions
        all_preds_shape.extend(outputs["logits_shape"].argmax(dim=1).cpu().tolist())
        all_targets_shape.extend(target_shape.cpu().tolist())

        all_preds_colour.extend(outputs["logits_colour"].argmax(dim=1).cpu().tolist())
        all_targets_colour.extend(target_colour.cpu().tolist())

        all_preds_border.extend(outputs["logits_border"].argmax(dim=1).cpu().tolist())
        all_targets_border.extend(target_border.cpu().tolist())

        all_preds_symcat.extend(outputs["logits_symcat"].argmax(dim=1).cpu().tolist())
        all_targets_symcat.extend(target_symcat.cpu().tolist())

    val_acc = accuracy_score(all_targets_class, all_preds_class)
    val_macro_f1 = f1_score(all_targets_class, all_preds_class, average="macro", zero_division=0)

    shape_acc = accuracy_score(all_targets_shape, all_preds_shape)
    colour_acc = accuracy_score(all_targets_colour, all_preds_colour)
    border_acc = accuracy_score(all_targets_border, all_preds_border)
    symcat_acc = accuracy_score(all_targets_symcat, all_preds_symcat)

    return {
        "val_loss": running_total_loss / total_samples,
        "val_loss_task": running_task_loss / total_samples,
        "val_loss_concepts": running_concept_loss / total_samples,
        "val_acc": float(val_acc),
        "val_macro_f1": float(val_macro_f1),
        "val_acc_shape": float(shape_acc),
        "val_acc_colour": float(colour_acc),
        "val_acc_border": float(border_acc),
        "val_acc_symcat": float(symcat_acc),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Concept Bottleneck Model on GTSRB")
    parser.add_argument("--data-root", type=str, default="german", help="Path to raw GTSRB root")
    parser.add_argument("--manifest-dir", type=str, default="data/master_v1/metadata", help="Metadata directory")
    parser.add_argument("--output-dir", type=str, default="reports/cbm_resnet18", help="Output directory")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--warmup-epochs", type=int, default=3, help="Linear warmup epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Base learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for 2D weights")
    parser.add_argument("--lambda-c", type=float, default=1.0, help="Concept loss weight")
    parser.add_argument("--lambda-aux", type=float, default=0.5, help="Auxiliary symbol loss weight")
    parser.add_argument(
        "--bottleneck-mode",
        type=str,
        default="residual",
        choices=["residual", "pure", "concat"],
        help="CBM bottleneck structure: 'residual' (recommended), 'pure', or 'concat'",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and torch.cuda.is_available()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("GTSRB Concept Bottleneck Model (CBM) Training — Experiment 6")
    print(f"Device             : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Bottleneck Mode    : {args.bottleneck_mode}")
    print(f"Lambda Concepts    : {args.lambda_c}")
    print(f"Lambda Aux (Symbol): {args.lambda_aux}")
    print(f"Epochs             : {args.epochs} (Warmup: {args.warmup_epochs})")
    print(f"Batch Size         : {args.batch_size}")
    print(f"Learning Rate      : {args.lr}")
    print(f"Output Directory   : {out_dir.resolve()}")
    print("=" * 75)

    # 1. Dataset & DataLoaders
    train_manifest = Path(args.manifest_dir) / "train.csv"
    val_manifest = Path(args.manifest_dir) / "val.csv"

    train_dataset = FrequencyAugDataset(
        manifest_path=train_manifest,
        data_root=args.data_root,
        is_training=True,
        mode="dropout",
        radius=14,
        severity=3.0,
    )
    val_dataset = GTSRBDataset(val_manifest, args.data_root, is_training=False)

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
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"Train samples: {len(train_dataset):,} | Val samples: {len(val_dataset):,}")

    # 2. Concept Lookup Tensors
    concept_luts = get_concept_lookup_tensors(device=device)

    # 3. Model
    model = ConceptBottleneckModel(
        num_classes=43,
        pretrained=True,
        zero_init_residual=True,
        bottleneck_mode=args.bottleneck_mode,
    ).to(device)

    # 4. Multi-Task Criterion & Optimizer
    criterion = CBMLoss(lambda_c=args.lambda_c, lambda_aux=args.lambda_aux)
    param_groups = get_parameter_groups(model, weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum=args.momentum, nesterov=True)

    # Cosine scheduler for post-warmup
    decay_epochs = args.epochs - args.warmup_epochs
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=decay_epochs, eta_min=1e-5
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 5. Training Loop
    history = []
    best_val_macro_f1 = 0.0
    total_start_time = time.time()

    print("\nStarting Frequency-Aware CBM Training...\n")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Handle linear warmup
        if epoch <= args.warmup_epochs:
            warmup_lr = args.lr * (epoch / args.warmup_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr
            current_lr = warmup_lr
        else:
            cosine_scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, concept_luts, use_amp
        )
        val_metrics = evaluate(model, val_loader, criterion, device, concept_luts, use_amp)
        epoch_duration = time.time() - epoch_start

        val_f1 = val_metrics["val_macro_f1"]
        val_acc = val_metrics["val_acc"]
        is_best = val_f1 > best_val_macro_f1

        if is_best:
            best_val_macro_f1 = val_f1
            best_model_path = out_dir / "best_freq_cbm_resnet18.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": val_f1,
                "val_acc": val_acc,
                "concept_accuracies": {
                    "shape": val_metrics["val_acc_shape"],
                    "colour": val_metrics["val_acc_colour"],
                    "border": val_metrics["val_acc_border"],
                    "symcat": val_metrics["val_acc_symcat"],
                },
                "config": vars(args),
            }, best_model_path)

        epoch_record = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_loss_task": train_metrics["train_loss_task"],
            "train_loss_concepts": train_metrics["train_loss_concepts"],
            "val_loss": val_metrics["val_loss"],
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
            "val_acc_shape": val_metrics["val_acc_shape"],
            "val_acc_colour": val_metrics["val_acc_colour"],
            "val_acc_border": val_metrics["val_acc_border"],
            "val_acc_symcat": val_metrics["val_acc_symcat"],
            "duration_sec": epoch_duration,
            "is_best": is_best,
        }
        history.append(epoch_record)

        marker = " [*NEW BEST VAL F1*]" if is_best else ""
        print(f"\n┌── Epoch [{epoch:02d}/{args.epochs:02d}] ──── Time: {epoch_duration:.1f}s ──── LR: {current_lr:.5f} ───{marker}")
        print(f"│  Train Loss : {train_loss:.4f} (Task: {train_metrics['train_loss_task']:.4f}, Concepts: {train_metrics['train_loss_concepts']:.4f})")
        print(f"│  Val Perf   : Acc: {val_acc*100:.2f}% │ Macro-F1: {val_f1*100:.2f}% (Best: {best_val_macro_f1*100:.2f}%)")
        print(f"│  Concepts   : Shape: {val_metrics['val_acc_shape']*100:.1f}% │ Colour: {val_metrics['val_acc_colour']*100:.1f}% │ Border: {val_metrics['val_acc_border']*100:.1f}% │ Symbol: {val_metrics['val_acc_symcat']*100:.1f}%")
        print("└" + "─" * 72)

    total_duration = time.time() - total_start_time
    print("\n" + "=" * 75)
    print("CBM TRAINING COMPLETED!")
    print(f"Total Time        : {total_duration/60:.2f} minutes")
    print(f"Best Val Macro-F1 : {best_val_macro_f1*100:.2f}%")
    print(f"Best Checkpoint   : {out_dir / 'best_freq_cbm_resnet18.pt'}")

    history_path = out_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": vars(args),
            "best_val_macro_f1": best_val_macro_f1,
            "total_duration_minutes": total_duration / 60,
            "epochs": history,
        }, f, indent=2)
    print(f"Training History  : {history_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
