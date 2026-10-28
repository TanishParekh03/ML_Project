"""
Production Hybrid (AugMix + CBM) Training Script (GTSRB Master v1)

Experiment 5: Hybrid AugMix + Concept Bottleneck ResNet-18
- Combines data-centric consistency regularization (AugMix triplet + JSD) with
  representation-centric semantic bottleneck supervision (CBM).
- Tests Hypothesis H3: Do data-centric and representation-centric methods provide
  complementary robustness improvements?

Optimization Controls (Identical to Experiments 1, 2, 3, & 4):
* SGD (Momentum 0.9, Nesterov, No-Bias-Decay rule)
* 3-epoch Linear Warmup + Cosine Annealing decay down to 1e-5 (15 epochs)
* Automatic Mixed Precision (torch.cuda.amp)
* Validation Macro-F1 model selection (checkpoint: best_hybrid_resnet18.pt)
* Comprehensive logging of task loss, concept losses, JSD consistency losses, and concept accuracies.
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

from src.augmentations.augmix import AugMixDataset
from src.dataset import GTSRBDataset
from src.models.baseline_cnn import get_parameter_groups
from src.models.cbm import ConceptBottleneckModel, get_concept_lookup_tensors
from src.models.hybrid_loss import HybridCBMLoss


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
    criterion: HybridCBMLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    concept_luts: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    use_amp: bool = True,
) -> Tuple[float, Dict[str, float]]:
    """Runs one hybrid training epoch with triplet concatenation and dual JSD consistency."""
    model.train()
    running_total = 0.0
    running_task = 0.0
    running_cpt = 0.0
    running_jsd_task = 0.0
    running_jsd_cpt = 0.0
    total_samples = 0

    shape_lut, colour_lut, border_lut, symcat_lut = concept_luts

    pbar = tqdm(loader, desc="Training", unit="batch", leave=False)
    for batch in pbar:
        img_clean = batch["image_clean"].to(device, non_blocking=True)
        img_aug1 = batch["image_aug1"].to(device, non_blocking=True)
        img_aug2 = batch["image_aug2"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        B = img_clean.size(0)

        # Ground-truth concept targets for clean image
        target_shape = shape_lut[labels]
        target_colour = colour_lut[labels]
        target_border = border_lut[labels]
        target_symcat = symcat_lut[labels]

        optimizer.zero_grad(set_to_none=True)

        autocast_ctx = (
            torch.amp.autocast(device_type="cuda", enabled=use_amp)
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast")
            else torch.cuda.amp.autocast(enabled=use_amp)
        )
        with autocast_ctx:
            # Concatenated forward pass across the 3 views [3*B, 3, 64, 64]
            images_concat = torch.cat([img_clean, img_aug1, img_aug2], dim=0)
            outputs_concat = model(images_concat)

            # Slice outputs into clean, aug1, and aug2 chunks
            outputs_clean = {k: v[:B] for k, v in outputs_concat.items()}
            outputs_aug1 = {k: v[B : 2 * B] for k, v in outputs_concat.items()}
            outputs_aug2 = {k: v[2 * B :] for k, v in outputs_concat.items()}

            loss, metrics = criterion(
                outputs_clean=outputs_clean,
                outputs_aug1=outputs_aug1,
                outputs_aug2=outputs_aug2,
                target_class=labels,
                target_shape=target_shape,
                target_colour=target_colour,
                target_border=target_border,
                target_symcat=target_symcat,
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_total += loss.item() * B
        running_task += metrics["loss_task"] * B
        running_cpt += metrics["loss_concepts"] * B
        running_jsd_task += metrics["loss_jsd_task"] * B
        running_jsd_cpt += metrics["loss_jsd_cpt"] * B
        total_samples += B

        pbar.set_postfix({
            "total": f"{loss.item():.2f}",
            "task": f"{metrics['loss_task']:.2f}",
            "jsd_y": f"{metrics['loss_jsd_task']:.3f}",
            "jsd_c": f"{metrics['loss_jsd_cpt']:.3f}",
        })

    avg_metrics = {
        "train_loss": running_total / total_samples,
        "train_loss_task": running_task / total_samples,
        "train_loss_concepts": running_cpt / total_samples,
        "train_loss_jsd_task": running_jsd_task / total_samples,
        "train_loss_jsd_cpt": running_jsd_cpt / total_samples,
    }
    return avg_metrics["train_loss"], avg_metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    concept_luts: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    use_amp: bool = True,
) -> Dict[str, float]:
    """Evaluates the model on clean validation data, recording class and concept metrics."""
    model.eval()
    running_loss = 0.0
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
    ce = nn.CrossEntropyLoss()

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        target_shape = shape_lut[labels]
        target_colour = colour_lut[labels]
        target_border = border_lut[labels]
        target_symcat = symcat_lut[labels]

        autocast_ctx = (
            torch.amp.autocast(device_type="cuda", enabled=use_amp)
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast")
            else torch.cuda.amp.autocast(enabled=use_amp)
        )
        with autocast_ctx:
            outputs = model(images)
            loss = ce(outputs["logits"], labels)

        bs = images.size(0)
        running_loss += loss.item() * bs
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
        "val_loss": running_loss / total_samples,
        "val_acc": float(val_acc),
        "val_macro_f1": float(val_macro_f1),
        "val_acc_shape": float(shape_acc),
        "val_acc_colour": float(colour_acc),
        "val_acc_border": float(border_acc),
        "val_acc_symcat": float(symcat_acc),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Hybrid (AugMix + CBM) on GTSRB")
    parser.add_argument("--data-root", type=str, default="german", help="Path to raw GTSRB root")
    parser.add_argument("--manifest-dir", type=str, default="data/master_v1/metadata", help="Metadata directory")
    parser.add_argument("--output-dir", type=str, default="reports/hybrid_resnet18", help="Output directory")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--warmup-epochs", type=int, default=3, help="Linear warmup epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Base learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for 2D weights")
    parser.add_argument("--lambda-c", type=float, default=1.0, help="Concept loss weight")
    parser.add_argument("--lambda-aux", type=float, default=0.5, help="Auxiliary symbol loss weight")
    parser.add_argument("--lambda-jsd-task", type=float, default=12.0, help="Class JSD consistency weight")
    parser.add_argument("--lambda-jsd-cpt", type=float, default=4.0, help="Concept JSD consistency weight")
    parser.add_argument("--augmix-severity", type=float, default=3.0, help="AugMix severity level")
    parser.add_argument(
        "--bottleneck-mode",
        type=str,
        default="residual",
        choices=["residual", "pure", "concat"],
        help="CBM bottleneck mode",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP")
    parser.add_argument(
        "--ablation",
        type=str,
        default="none",
        choices=["none", "geometric_only", "decouple_concept_jsd"],
        help="Ablation mode: 'none' (Exp 5), 'geometric_only' (Ablation 5A), 'decouple_concept_jsd' (Ablation 5B)",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and torch.cuda.is_available()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GTSRB Hybrid (AugMix + Concept Bottleneck Model) Training — Experiment 5")
    print(f"Device               : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Bottleneck Mode      : {args.bottleneck_mode}")
    print(f"Lambda Concepts      : {args.lambda_c}")
    print(f"Lambda Aux (Symbol)  : {args.lambda_aux}")
    print(f"Lambda JSD (Task)    : {args.lambda_jsd_task}")
    print(f"Lambda JSD (Concepts): {args.lambda_jsd_cpt}")
    print(f"AugMix Severity      : {args.augmix_severity}")
    print(f"Epochs               : {args.epochs} (Warmup: {args.warmup_epochs})")
    print(f"Batch Size           : {args.batch_size}")
    print(f"Learning Rate        : {args.lr}")
    print(f"Output Directory     : {out_dir.resolve()}")
    print("=" * 80)

    # 1. Dataset & DataLoaders
    train_manifest = Path(args.manifest_dir) / "train.csv"
    val_manifest = Path(args.manifest_dir) / "val.csv"

    # Train with AugMixDataset (triplets); Val with standard GTSRBDataset (clean)
    exclude_color_ops = (args.ablation == "geometric_only")
    train_dataset = AugMixDataset(
        manifest_path=train_manifest,
        data_root=args.data_root,
        severity=args.augmix_severity,
        width=3,
        depth=-1,
        alpha=1.0,
        exclude_color_ops=exclude_color_ops,
        is_training=True,
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
    print(f"Train samples (triplets): {len(train_dataset):,} | Val samples: {len(val_dataset):,}")

    # 2. Concept Lookup Tensors
    concept_luts = get_concept_lookup_tensors(device=device)

    # 3. Model
    model = ConceptBottleneckModel(
        num_classes=43,
        pretrained=True,
        zero_init_residual=True,
        bottleneck_mode=args.bottleneck_mode,
    ).to(device)

    # 4. Multi-Scale Hybrid Criterion & Optimizer
    decouple_concept_jsd = (args.ablation == "decouple_concept_jsd")
    criterion = HybridCBMLoss(
        lambda_c=args.lambda_c,
        lambda_aux=args.lambda_aux,
        lambda_jsd_task=args.lambda_jsd_task,
        lambda_jsd_cpt=0.0 if decouple_concept_jsd else args.lambda_jsd_cpt,
        decouple_concept_jsd=decouple_concept_jsd,
    )
    param_groups = get_parameter_groups(model, weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum=args.momentum, nesterov=True)

    decay_epochs = args.epochs - args.warmup_epochs
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=decay_epochs, eta_min=1e-5
    )
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 5. Training Loop
    history = []
    best_val_macro_f1 = 0.0
    total_start_time = time.time()

    print("\nStarting Hybrid (AugMix + CBM) Training...\n")
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
        val_metrics = evaluate(model, val_loader, device, concept_luts, use_amp)
        epoch_duration = time.time() - epoch_start

        val_f1 = val_metrics["val_macro_f1"]
        val_acc = val_metrics["val_acc"]
        is_best = val_f1 > best_val_macro_f1

        if is_best:
            best_val_macro_f1 = val_f1
            ckpt_name = "best_hybrid_resnet18.pt" if args.ablation == "none" else f"best_hybrid_{args.ablation}_resnet18.pt"
    best_model_path = out_dir / ckpt_name
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
            "train_loss_jsd_task": train_metrics["train_loss_jsd_task"],
            "train_loss_jsd_cpt": train_metrics["train_loss_jsd_cpt"],
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

        marker = " [*BEST*]" if is_best else ""
        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"LR: {current_lr:.5f} | "
            f"Loss: {train_loss:.3f} (JSD_Y: {train_metrics['train_loss_jsd_task']:.3f}, JSD_C: {train_metrics['train_loss_jsd_cpt']:.3f}) | "
            f"Val Acc: {val_acc*100:.2f}% | "
            f"Val F1: {val_f1*100:.2f}% | "
            f"Cpt Acc (Sh/Col/Bor/Sym): {val_metrics['val_acc_shape']*100:.1f}%/"
            f"{val_metrics['val_acc_colour']*100:.1f}%/"
            f"{val_metrics['val_acc_border']*100:.1f}%/"
            f"{val_metrics['val_acc_symcat']*100:.1f}% "
            f"({epoch_duration:.1f}s){marker}"
        )

    total_duration = time.time() - total_start_time
    print("\n" + "=" * 80)
    print("HYBRID TRAINING COMPLETED!")
    print(f"Total Time        : {total_duration/60:.2f} minutes")
    print(f"Best Val Macro-F1 : {best_val_macro_f1*100:.2f}%")
    print(f"Best Checkpoint   : {out_dir / 'best_hybrid_resnet18.pt'}")

    history_path = out_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": vars(args),
            "best_val_macro_f1": best_val_macro_f1,
            "total_duration_minutes": total_duration / 60,
            "epochs": history,
        }, f, indent=2)
    print(f"Training History  : {history_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
