"""
Phase 2: In-Domain Model Training Protocol (BelgiumTSC and Chinese TSRD)
========================================================================
Executes in-domain training for:
1. Baseline ResNet-18 (Black-box CNN)
2. Frequency-Aware Concept Bottleneck Model (Freq-CBM ResNet-18)
3. Standard Concept Bottleneck Model (Residual CBM ResNet-18)

Adheres strictly to the master training dynamics:
- Bag-of-Tricks zero-gamma BN initialization
- Decoupled No-Bias-Decay rule (weight decay applied strictly to non-bias/norm weights)
- 3-epoch Linear Warmup + Cosine Annealing decay down to 1e-5
- Automatic Mixed Precision (torch.cuda.amp)
- Macro-F1 model selection checkpointing
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.baseline_cnn import BaselineResNet18, get_parameter_groups
from src.models.cbm import CBMLoss, ConceptBottleneckModel
from src.models.concept_encoder import (
    BORDER_VOCAB,
    COLOUR_VOCAB,
    SHAPE_VOCAB,
    SYMCAT_VOCAB,
    encode_all_classes,
)
from src.augmentations.frequency import frequency_dropout

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==============================================================================
# In-Domain Fast Letterbox Dataset
# ==============================================================================

class InDomainTSRDataset(Dataset):
    """Loads standardized letterboxed 64x64 images directly from pre-rendered paths."""

    def __init__(
        self,
        data_root: Path,
        manifest_path: Path,
        concept_csv: Optional[Path] = None,
        is_training: bool = False,
        use_frequency_dropout: bool = False,
    ):
        self.data_root = Path(data_root).resolve()
        self.manifest_path = Path(manifest_path).resolve()
        self.df = pd.read_csv(self.manifest_path)
        self.is_training = is_training
        self.use_frequency_dropout = use_frequency_dropout

        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        # Concept vocabulary mappings
        self.shapes = [SHAPE_VOCAB.get(s, 0) for s in self.df["shape"]]
        self.colours = [COLOUR_VOCAB.get(c, 0) for c in self.df["colour_profile"]]
        self.borders = [BORDER_VOCAB.get(b, 0) for b in self.df["border_type"]]
        self.symbols = [SYMCAT_VOCAB.get(s, 0) for s in self.df["symbol_category"]]
        self.class_ids = self.df["class_id"].values.astype(np.int64)
        self.paths = self.df["canonical_image_path"].values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rel_path = self.paths[idx]
        full_path = self.data_root / rel_path
        with Image.open(full_path) as img:
            img = img.convert("RGB")

            # Training spectral augmentation for Freq-CBM
            if self.is_training and self.use_frequency_dropout:
                if random.random() < 0.5:
                    img = frequency_dropout(img, severity=3.0, r_min=10)

            # Minimal geometry jitter during training
            if self.is_training and random.random() < 0.3:
                angle = random.uniform(-7.0, 7.0)
                img = TF.rotate(img, angle, fill=[0, 0, 0])

            tensor = TF.to_tensor(img)
            tensor = self.normalize(tensor)

        return {
            "image": tensor,
            "label": torch.tensor(self.class_ids[idx], dtype=torch.long),
            "shape": torch.tensor(self.shapes[idx], dtype=torch.long),
            "colour": torch.tensor(self.colours[idx], dtype=torch.long),
            "border": torch.tensor(self.borders[idx], dtype=torch.long),
            "symbol": torch.tensor(self.symbols[idx], dtype=torch.long),
        }


def get_dataset_concept_luts(concept_csv: Path, num_classes: int, device: torch.device):
    """Builds class-to-concept lookup tensors for any arbitrary class count."""
    enc = encode_all_classes(str(concept_csv))
    shape_arr = np.array([enc[i]["shape_idx"] for i in range(num_classes)], dtype=np.int64)
    colour_arr = np.array([enc[i]["colour_idx"] for i in range(num_classes)], dtype=np.int64)
    border_arr = np.array([enc[i]["border_idx"] for i in range(num_classes)], dtype=np.int64)
    symcat_arr = np.array([enc[i]["symcat_idx"] for i in range(num_classes)], dtype=np.int64)

    return (
        torch.from_numpy(shape_arr).to(device),
        torch.from_numpy(colour_arr).to(device),
        torch.from_numpy(border_arr).to(device),
        torch.from_numpy(symcat_arr).to(device),
    )


# ==============================================================================
# Training & Validation Loops
# ==============================================================================

def train_epoch_baseline(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    use_amp: bool = True,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def train_epoch_cbm(
    model: nn.Module,
    loader: DataLoader,
    criterion: CBMLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    concept_luts: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    use_amp: bool = True,
) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    loss_accum: Dict[str, float] = {}
    shape_lut, colour_lut, border_lut, symcat_lut = concept_luts

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        target_shape = shape_lut[labels]
        target_colour = colour_lut[labels]
        target_border = border_lut[labels]
        target_symcat = symcat_lut[labels]

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss, loss_dict = criterion(
                outputs, labels, target_shape, target_colour, target_border, target_symcat
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = images.size(0)
        total_loss += loss.item() * bs
        for k, v in loss_dict.items():
            loss_accum[k] = loss_accum.get(k, 0.0) + v * bs

    n_samples = len(loader.dataset)
    avg_loss_dict = {k: v / n_samples for k, v in loss_accum.items()}
    return total_loss / n_samples, avg_loss_dict


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_cbm: bool = False,
) -> Dict[str, float]:
    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []

    s_preds, s_targets = [], []
    c_preds, c_targets = [], []
    b_preds, b_targets = [], []
    y_preds, y_targets = [], []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"]

        if is_cbm:
            outputs = model(images)
            preds = torch.argmax(outputs["logits"], dim=1).cpu().numpy()
            s_p = torch.argmax(outputs["pred_shape"], dim=1).cpu().numpy()
            c_p = torch.argmax(outputs["pred_colour"], dim=1).cpu().numpy()
            b_p = torch.argmax(outputs["pred_border"], dim=1).cpu().numpy()
            y_p = torch.argmax(outputs["pred_symcat"], dim=1).cpu().numpy()

            s_preds.extend(s_p)
            s_targets.extend(batch["shape"].numpy())
            c_preds.extend(c_p)
            c_targets.extend(batch["colour"].numpy())
            b_preds.extend(b_p)
            b_targets.extend(batch["border"].numpy())
            y_preds.extend(y_p)
            y_targets.extend(batch["symbol"].numpy())
        else:
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_targets.extend(labels.numpy())

    top1 = float(accuracy_score(all_targets, all_preds))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))

    metrics = {
        "top1_acc": top1,
        "macro_f1": macro_f1,
    }

    if is_cbm:
        s_acc = float(accuracy_score(s_targets, s_preds))
        c_acc = float(accuracy_score(c_targets, c_preds))
        b_acc = float(accuracy_score(b_targets, b_preds))
        y_acc = float(accuracy_score(y_targets, y_preds))
        mean_cpt = (s_acc + c_acc + b_acc + y_acc) / 4.0

        exact_4 = np.mean(
            (np.array(s_preds) == np.array(s_targets))
            & (np.array(c_preds) == np.array(c_targets))
            & (np.array(b_preds) == np.array(b_targets))
            & (np.array(y_preds) == np.array(y_targets))
        )

        metrics.update({
            "shape_acc": s_acc,
            "colour_acc": c_acc,
            "border_acc": b_acc,
            "symbol_acc": y_acc,
            "mean_concept_acc": mean_cpt,
            "exact_4_concept_match": float(exact_4),
        })

    return metrics


# ==============================================================================
# Main Orchestration Routine
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 2 In-Domain Model Training")
    parser.add_argument("--dataset", type=str, required=True, choices=["belgium", "china"],
                        help="Target dataset: 'belgium' (data/data_v2) or 'china' (data/data_v3)")
    parser.add_argument("--model-type", type=str, required=True, choices=["baseline", "freq_cbm", "cbm"],
                        help="Model architecture: 'baseline', 'freq_cbm', or 'cbm'")
    parser.add_argument("--epochs", type=int, default=30, help="Total training epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=0.05, help="Initial learning rate (default: 0.05)")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed (default: 42)")
    parser.add_argument("--warmup-epochs", type=int, default=3, help="Linear warmup epochs (default: 3)")
    parser.add_argument("--output-dir", type=str, default="", help="Custom output directory")
    parser.add_argument("--no-amp", action="store_true", help="Disable Automatic Mixed Precision")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"

    # Dataset resolution
    if args.dataset == "belgium":
        data_root = PROJECT_ROOT / "data" / "data_v2"
        num_classes = 62
        dataset_name = "BelgiumTSC"
    else:
        data_root = PROJECT_ROOT / "data" / "data_v3"
        num_classes = 58
        dataset_name = "Chinese TSRD"

    train_manifest = data_root / "metadata" / "train.csv"
    val_manifest = data_root / "metadata" / "val.csv"
    test_manifest = data_root / "metadata" / "test.csv"
    concept_csv = data_root / "concepts" / "class_concepts.csv"

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "checkpoints" / f"phase2_{args.dataset}_{args.model_type}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Phase 2 In-Domain Training: {args.model_type.upper()} on {dataset_name}")
    print(f"Device: {device} | Mixed Precision: {use_amp} | Num Classes: {num_classes}")
    print(f"Data Root: {data_root}")
    print(f"Output Directory: {output_dir}")
    print("=" * 80)

    # Datasets & Loaders
    use_freq_drop = (args.model_type == "freq_cbm")
    train_dataset = InDomainTSRDataset(
        data_root=data_root,
        manifest_path=train_manifest,
        concept_csv=concept_csv,
        is_training=True,
        use_frequency_dropout=use_freq_drop,
    )
    val_dataset = InDomainTSRDataset(
        data_root=data_root,
        manifest_path=val_manifest,
        concept_csv=concept_csv,
        is_training=False,
    )
    test_dataset = InDomainTSRDataset(
        data_root=data_root,
        manifest_path=test_manifest,
        concept_csv=concept_csv,
        is_training=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2 if sys.platform != "win32" else 0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2 if sys.platform != "win32" else 0,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2 if sys.platform != "win32" else 0,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Samples: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # Model Initialization
    is_cbm = (args.model_type in ["freq_cbm", "cbm"])
    if is_cbm:
        model = ConceptBottleneckModel(
            num_classes=num_classes,
            pretrained=True,
            zero_init_residual=True,
            bottleneck_mode="residual",
        ).to(device)
        concept_luts = get_dataset_concept_luts(concept_csv, num_classes, device)
        criterion = CBMLoss(lambda_c=1.0, lambda_aux=0.5)
    else:
        model = BaselineResNet18(
            num_classes=num_classes,
            pretrained=True,
            zero_init_residual=True,
        ).to(device)
        criterion = nn.CrossEntropyLoss()
        concept_luts = None

    # Optimizer (Decoupled No-Bias-Decay)
    param_groups = get_parameter_groups(model, weight_decay=1e-4)
    optimizer = torch.optim.SGD(param_groups, lr=args.lr, momentum=0.9, nesterov=True)

    cosine_epochs = max(1, args.epochs - args.warmup_epochs)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_macro_f1 = -1.0
    history: List[Dict[str, Any]] = []
    checkpoint_path = output_dir / f"best_{args.model_type}_{args.dataset}.pt"
    start_time = time.time()

    # Training Loop
    for epoch in range(1, args.epochs + 1):
        # Warmup vs Cosine Annealing
        if epoch <= args.warmup_epochs:
            warmup_lr = args.lr * (epoch / args.warmup_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr
            curr_lr = warmup_lr
        else:
            cosine_scheduler.step()
            curr_lr = optimizer.param_groups[0]["lr"]

        epoch_start = time.time()
        if is_cbm:
            train_loss, loss_dict = train_epoch_cbm(
                model, train_loader, criterion, optimizer, scaler, concept_luts, device, use_amp
            )
        else:
            train_loss = train_epoch_baseline(
                model, train_loader, criterion, optimizer, scaler, device, use_amp
            )
            loss_dict = {"task_ce": train_loss}

        # Validation
        val_metrics = evaluate_model(model, val_loader, device, is_cbm=is_cbm)
        epoch_dur = time.time() - epoch_start

        # Record History
        rec = {
            "epoch": epoch,
            "lr": curr_lr,
            "train_loss": train_loss,
            "val_top1": val_metrics["top1_acc"],
            "val_macro_f1": val_metrics["macro_f1"],
            "duration_sec": epoch_dur,
        }
        rec.update(loss_dict)
        if is_cbm:
            rec["val_mean_concept_acc"] = val_metrics.get("mean_concept_acc", 0.0)
            rec["val_exact_4_match"] = val_metrics.get("exact_4_concept_match", 0.0)

        history.append(rec)

        # Checkpoint selection on Macro-F1
        is_best = val_metrics["macro_f1"] > best_val_macro_f1
        if is_best:
            best_val_macro_f1 = val_metrics["macro_f1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": best_val_macro_f1,
                "val_top1": val_metrics["top1_acc"],
                "num_classes": num_classes,
                "dataset": args.dataset,
                "model_type": args.model_type,
            }, checkpoint_path)

        cpt_str = f" | Val Cpt: {val_metrics.get('mean_concept_acc', 0.0)*100:.1f}%" if is_cbm else ""
        best_str = " [BEST]" if is_best else ""
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | LR: {curr_lr:.5f} | Train Loss: {train_loss:.4f} | "
              f"Val Acc: {val_metrics['top1_acc']*100:.2f}% | Val F1: {val_metrics['macro_f1']*100:.2f}%"
              f"{cpt_str} ({epoch_dur:.1f}s){best_str}")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} mins. Peak Val Macro-F1: {best_val_macro_f1*100:.2f}%")

    # Load best checkpoint for clean test evaluation
    print(f"\nEvaluating Best Checkpoint ({checkpoint_path.name}) on Held-Out Test Set...")
    best_ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_metrics = evaluate_model(model, test_loader, device, is_cbm=is_cbm)

    print(f"Test Top-1 Accuracy: {test_metrics['top1_acc']*100:.2f}%")
    print(f"Test Macro-F1:      {test_metrics['macro_f1']*100:.2f}%")
    if is_cbm:
        print(f"Test Mean Concept Acc: {test_metrics['mean_concept_acc']*100:.2f}%")
        print(f"Test 4-Cpt Exact Match:{test_metrics['exact_4_concept_match']*100:.2f}%")

    # Save summary & history
    summary = {
        "dataset": args.dataset,
        "model_type": args.model_type,
        "num_classes": num_classes,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "best_epoch": best_ckpt["epoch"],
        "val_peak_macro_f1": best_val_macro_f1,
        "test_top1_acc": test_metrics["top1_acc"],
        "test_macro_f1": test_metrics["macro_f1"],
        "total_train_time_sec": total_time,
        "checkpoint_file": str(checkpoint_path),
    }
    summary.update({f"test_{k}": v for k, v in test_metrics.items()})

    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "epochs": history}, f, indent=2)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Results logged to: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

