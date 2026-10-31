"""Phase 1: Cross-Dataset Zero-Shot & Linear Probe Evaluation Across Normalization Regimes
======================================================================================
Evaluates German GTSRB-trained models on foreign road sign benchmarks:
  - BelgiumTSC (data_v2: 62 classes, 2,534 test images)
  - Chinese TSRD (data_v3: 58 classes, 1,994 test images)

Evaluates the 2x2 Factorial Matrix per Dataset:
  1. Baseline ResNet-18 (Static BN)
  2. Baseline ResNet-18 (Test-Time BN-Adaptation / Shift)
  3. Concept Bottleneck Model (Static BN)
  4. Concept Bottleneck Model (Test-Time BN-Adaptation / Shift)

Core Hypotheses Tested:
  - Generality vs. Memorization: Do learned physical concepts (shape, color, border, symbol)
    transfer across international borders zero-shot?
  - Covariate Shift Recovery: Does test-time BN adaptation rescue cross-border domain shift
    induced by foreign camera sensors, lighting, and environmental styles?
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.models.baseline_cnn import BaselineResNet18
from src.models.cbm import ConceptBottleneckModel
from src.models.concept_encoder import (
    BORDER_OFFSET,
    BOTTLENECK_DIM,
    COLOUR_OFFSET,
    N_BORDER,
    N_COLOUR,
    N_SHAPE,
    N_SYMCAT,
    SHAPE_OFFSET,
    SYMCAT_OFFSET,
    SHAPE_VOCAB,
    COLOUR_VOCAB,
    BORDER_VOCAB,
    SYMCAT_VOCAB,
)

def encode_concept_vector(shape: str, colour: str, border: str, symcat: str) -> np.ndarray:
    """Encodes 4 concept strings into a 22-dimensional concatenated one-hot vector."""
    vec = np.zeros(BOTTLENECK_DIM, dtype=np.float32)
    vec[SHAPE_OFFSET + SHAPE_VOCAB[shape]] = 1.0
    vec[COLOUR_OFFSET + COLOUR_VOCAB[colour]] = 1.0
    vec[BORDER_OFFSET + BORDER_VOCAB[border]] = 1.0
    vec[SYMCAT_OFFSET + SYMCAT_VOCAB[symcat]] = 1.0
    return vec

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ==============================================================================
# 1. Dataset & DataLoader for Pre-Rendered Canonical Images
# ==============================================================================

class CanonicalTSRDataset(Dataset):
    """Loads standardized 64x64 letterboxed PNG images and concept metadata."""

    def __init__(self, data_root: Path, manifest_path: Path):
        self.data_root = data_root
        self.df = pd.read_csv(manifest_path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        # Precompute concept target integers
        self.shapes = [SHAPE_VOCAB[s] for s in self.df["shape"]]
        self.colours = [COLOUR_VOCAB[c] for c in self.df["colour_profile"]]
        self.borders = [BORDER_VOCAB[b] for b in self.df["border_type"]]
        self.symbols = [SYMCAT_VOCAB[s] for s in self.df["symbol_category"]]
        self.class_ids = self.df["class_id"].values.astype(np.int64)
        self.paths = self.df["canonical_image_path"].values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rel_path = self.paths[idx]
        full_path = self.data_root / rel_path
        with Image.open(full_path) as img:
            img = img.convert("RGB")
            tensor = self.transform(img)

        return {
            "image": tensor,
            "class_id": self.class_ids[idx],
            "shape": self.shapes[idx],
            "colour": self.colours[idx],
            "border": self.borders[idx],
            "symbol": self.symbols[idx],
        }


# ==============================================================================
# 2. Batch Normalization Mode Control
# ==============================================================================

def set_bn_mode(model: nn.Module, mode: str):
    """
    mode: 'static' -> freeze running stats (m.eval(), track_running_stats=True)
          'adapt'  -> compute pure batch statistics (m.train(), track_running_stats=False)
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if mode == "adapt":
                m.train()
                m.track_running_stats = False
            else:  # static
                m.eval()
                m.track_running_stats = True


# ==============================================================================
# 3. Class Concept Signature Matrix (for Zero-Shot Ontology Matching)
# ==============================================================================

def build_ontology_signatures(concept_csv: Path) -> np.ndarray:
    """Builds an [N_classes, 22] one-hot concept signature matrix."""
    df = pd.read_csv(concept_csv)
    n_classes = len(df)
    signatures = np.zeros((n_classes, BOTTLENECK_DIM), dtype=np.float32)

    for idx, row in df.iterrows():
        c_id = int(row["class_id"])
        vec = encode_concept_vector(
            shape=row["shape"],
            colour=row["colour_profile"],
            border=row["border_type"],
            symcat=row["symbol_category"],
        )
        signatures[c_id] = vec

    return signatures


# ==============================================================================
# 4. Feature Extraction & Concept Prediction Engine
# ==============================================================================

@torch.no_grad()
def extract_dataset_representations(
    model: nn.Module,
    loader: DataLoader,
    is_cbm: bool,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Runs forward pass to collect features, concept predictions, and ground-truth targets."""
    features_list = []
    class_ids_list = []

    # CBM specific accumulators
    shape_preds, colour_preds, border_preds, symbol_preds = [], [], [], []
    shape_targets, colour_targets, border_targets, symbol_targets = [], [], [], []
    c_hat_list = []

    for batch in loader:
        images = batch["image"].to(device)
        class_ids = batch["class_id"].numpy()
        class_ids_list.append(class_ids)

        if is_cbm:
            feat = model.extract_features(images)
            l_s, l_c, l_b, l_sym, c_hat = model.predict_concepts_from_features(feat)

            features_list.append(feat.cpu().numpy())
            c_hat_list.append(c_hat.cpu().numpy())

            shape_preds.append(l_s.argmax(dim=-1).cpu().numpy())
            colour_preds.append(l_c.argmax(dim=-1).cpu().numpy())
            border_preds.append(l_b.argmax(dim=-1).cpu().numpy())
            symbol_preds.append(l_sym.argmax(dim=-1).cpu().numpy())

            shape_targets.append(batch["shape"].numpy())
            colour_targets.append(batch["colour"].numpy())
            border_targets.append(batch["border"].numpy())
            symbol_targets.append(batch["symbol"].numpy())
        else:
            feat = model.extract_features(images)
            features_list.append(feat.cpu().numpy())

    res = {
        "features": np.concatenate(features_list, axis=0),
        "class_ids": np.concatenate(class_ids_list, axis=0),
    }

    if is_cbm:
        res.update({
            "c_hat": np.concatenate(c_hat_list, axis=0),
            "shape_pred": np.concatenate(shape_preds, axis=0),
            "colour_pred": np.concatenate(colour_preds, axis=0),
            "border_pred": np.concatenate(border_preds, axis=0),
            "symbol_pred": np.concatenate(symbol_preds, axis=0),
            "shape_tgt": np.concatenate(shape_targets, axis=0),
            "colour_tgt": np.concatenate(colour_targets, axis=0),
            "border_tgt": np.concatenate(border_targets, axis=0),
            "symbol_tgt": np.concatenate(symbol_targets, axis=0),
        })

    return res


# ==============================================================================
# 5. Experiment Evaluation Pipeline
# ==============================================================================

def run_experiment(
    model: nn.Module,
    is_cbm: bool,
    train_loader: DataLoader,
    test_loader: DataLoader,
    ontology_signatures: np.ndarray,
    bn_mode: str,
    device: torch.device,
) -> dict[str, Any]:
    """Runs a single experiment (model × bn_mode) evaluating representation and concepts."""
    # 1. Set Normalization Mode
    set_bn_mode(model, bn_mode)

    # 2. Extract Training Set Representations
    train_out = extract_dataset_representations(model, train_loader, is_cbm=is_cbm, device=device)
    X_train, y_train = train_out["features"], train_out["class_ids"]

    # 3. Extract Test Set Representations
    test_out = extract_dataset_representations(model, test_loader, is_cbm=is_cbm, device=device)
    X_test, y_test = test_out["features"], test_out["class_ids"]

    # 4. Fit Linear Classifier Probe on Frozen Features
    clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", multi_class="multinomial", random_state=42)
    clf.fit(X_train, y_train)
    y_pred_probe = clf.predict(X_test)

    probe_acc = float(accuracy_score(y_test, y_pred_probe))
    probe_macro_f1 = float(f1_score(y_test, y_pred_probe, average="macro", zero_division=0))

    results: dict[str, Any] = {
        "linear_probe_accuracy": probe_acc,
        "linear_probe_macro_f1": probe_macro_f1,
    }

    # 5. Evaluate Concept Generality (CBM Only)
    if is_cbm:
        # Per-axis Concept Accuracies & Macro-F1
        s_acc = float(accuracy_score(test_out["shape_tgt"], test_out["shape_pred"]))
        s_f1 = float(f1_score(test_out["shape_tgt"], test_out["shape_pred"], average="macro", zero_division=0))

        c_acc = float(accuracy_score(test_out["colour_tgt"], test_out["colour_pred"]))
        c_f1 = float(f1_score(test_out["colour_tgt"], test_out["colour_pred"], average="macro", zero_division=0))

        b_acc = float(accuracy_score(test_out["border_tgt"], test_out["border_pred"]))
        b_f1 = float(f1_score(test_out["border_tgt"], test_out["border_pred"], average="macro", zero_division=0))

        sym_acc = float(accuracy_score(test_out["symbol_tgt"], test_out["symbol_pred"]))
        sym_f1 = float(f1_score(test_out["symbol_tgt"], test_out["symbol_pred"], average="macro", zero_division=0))

        # Exact Match (all 4 concepts simultaneously correct)
        all_exact = (
            (test_out["shape_tgt"] == test_out["shape_pred"])
            & (test_out["colour_tgt"] == test_out["colour_pred"])
            & (test_out["border_tgt"] == test_out["border_pred"])
            & (test_out["symbol_tgt"] == test_out["symbol_pred"])
        )
        exact_rate = float(np.mean(all_exact))

        mean_concept_acc = float(np.mean([s_acc, c_acc, b_acc, sym_acc]))
        mean_concept_f1 = float(np.mean([s_f1, c_f1, b_f1, sym_f1]))

        # Zero-Shot Ontology Matching
        c_hat = test_out["c_hat"]  # [N_test, 22]
        # Cosine / dot-product similarity against ontology signatures: [N_test, N_classes]
        sims = np.dot(c_hat, ontology_signatures.T)
        y_pred_zeroshot = np.argmax(sims, axis=1)

        zs_acc = float(accuracy_score(y_test, y_pred_zeroshot))
        zs_f1 = float(f1_score(y_test, y_pred_zeroshot, average="macro", zero_division=0))

        results.update({
            "concept_shape_acc": s_acc,
            "concept_shape_f1": s_f1,
            "concept_colour_acc": c_acc,
            "concept_colour_f1": c_f1,
            "concept_border_acc": b_acc,
            "concept_border_f1": b_f1,
            "concept_symbol_acc": sym_acc,
            "concept_symbol_f1": sym_f1,
            "concept_mean_acc": mean_concept_acc,
            "concept_mean_macro_f1": mean_concept_f1,
            "concept_exact_match_rate": exact_rate,
            "zero_shot_ontology_acc": zs_acc,
            "zero_shot_ontology_f1": zs_f1,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Cross-Dataset Evaluation on Belgium and China.")
    parser.add_argument("--baseline-ckpt", type=str, default="reports/baseline_resnet18/best_baseline_resnet18.pt")
    parser.add_argument("--cbm-ckpt", type=str, default="reports/cbm_resnet18/best_cbm_resnet18.pt")
    parser.add_argument("--output-dir", type=str, default="reports/cross_dataset_eval")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 1: CROSS-DATASET ZERO-SHOT & NORMALIZATION ADAPTATION BENCHMARK")
    print(f"Device        : {device}")
    print(f"Baseline Ckpt : {args.baseline_ckpt}")
    print(f"CBM Ckpt      : {args.cbm_ckpt}")
    print(f"Output Root   : {out_dir}")
    print("=" * 80)

    # Load Baseline Model
    baseline_model = BaselineResNet18(num_classes=43, pretrained=False)
    b_ckpt = torch.load(args.baseline_ckpt, map_location=device, weights_only=False)
    baseline_model.load_state_dict(b_ckpt["model_state_dict"])
    baseline_model.to(device)

    # Load CBM Model
    cbm_model = ConceptBottleneckModel(num_classes=43, pretrained=False, bottleneck_mode="residual")
    c_ckpt = torch.load(args.cbm_ckpt, map_location=device, weights_only=False)
    cbm_model.load_state_dict(c_ckpt["model_state_dict"])
    cbm_model.to(device)

    datasets_to_eval = [
        ("Belgium (BelgiumTSC)", Path("data/data_v2"), 62),
        ("China (TSRD)", Path("data/data_v3"), 58),
    ]

    all_results: dict[str, Any] = {}

    for name, root, n_classes in datasets_to_eval:
        print(f"\n>>> BENCHMARKING ON: {name.upper()} ({n_classes} Classes)")
        train_csv = root / "metadata/train.csv"
        test_csv = root / "metadata/test.csv"
        concept_csv = root / "concepts/class_concepts.csv"

        train_ds = CanonicalTSRDataset(root, train_csv)
        test_ds = CanonicalTSRDataset(root, test_csv)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        ontology_sigs = build_ontology_signatures(concept_csv)
        ds_results: dict[str, Any] = {}

        # 1. Baseline ResNet-18 (Static BN)
        print("  - Running Exp 1: Baseline (Static BN)...")
        ds_results["baseline_static"] = run_experiment(
            baseline_model, is_cbm=False,
            train_loader=train_loader, test_loader=test_loader,
            ontology_signatures=ontology_sigs, bn_mode="static", device=device
        )
        print(f"    Linear Probe Acc: {ds_results['baseline_static']['linear_probe_accuracy']*100:.2f}% | Macro-F1: {ds_results['baseline_static']['linear_probe_macro_f1']*100:.2f}%")

        # 2. Baseline ResNet-18 (BN-Adapt / Shift)
        print("  - Running Exp 2: Baseline (BN-Adapt / Shift)...")
        ds_results["baseline_bn_adapt"] = run_experiment(
            baseline_model, is_cbm=False,
            train_loader=train_loader, test_loader=test_loader,
            ontology_signatures=ontology_sigs, bn_mode="adapt", device=device
        )
        print(f"    Linear Probe Acc: {ds_results['baseline_bn_adapt']['linear_probe_accuracy']*100:.2f}% | Macro-F1: {ds_results['baseline_bn_adapt']['linear_probe_macro_f1']*100:.2f}%")

        # 3. CBM (Static BN)
        print("  - Running Exp 3: CBM (Static BN)...")
        ds_results["cbm_static"] = run_experiment(
            cbm_model, is_cbm=True,
            train_loader=train_loader, test_loader=test_loader,
            ontology_signatures=ontology_sigs, bn_mode="static", device=device
        )
        print(f"    Linear Probe Acc: {ds_results['cbm_static']['linear_probe_accuracy']*100:.2f}% | Macro-F1: {ds_results['cbm_static']['linear_probe_macro_f1']*100:.2f}%")
        print(f"    Mean Concept Acc: {ds_results['cbm_static']['concept_mean_acc']*100:.2f}% | Exact Match: {ds_results['cbm_static']['concept_exact_match_rate']*100:.2f}% | Zero-Shot Acc: {ds_results['cbm_static']['zero_shot_ontology_acc']*100:.2f}%")

        # 4. CBM (BN-Adapt / Shift)
        print("  - Running Exp 4: CBM (BN-Adapt / Shift)...")
        ds_results["cbm_bn_adapt"] = run_experiment(
            cbm_model, is_cbm=True,
            train_loader=train_loader, test_loader=test_loader,
            ontology_signatures=ontology_sigs, bn_mode="adapt", device=device
        )
        print(f"    Linear Probe Acc: {ds_results['cbm_bn_adapt']['linear_probe_accuracy']*100:.2f}% | Macro-F1: {ds_results['cbm_bn_adapt']['linear_probe_macro_f1']*100:.2f}%")
        print(f"    Mean Concept Acc: {ds_results['cbm_bn_adapt']['concept_mean_acc']*100:.2f}% | Exact Match: {ds_results['cbm_bn_adapt']['concept_exact_match_rate']*100:.2f}% | Zero-Shot Acc: {ds_results['cbm_bn_adapt']['zero_shot_ontology_acc']*100:.2f}%")

        all_results[name] = ds_results

    # Save Results JSON
    summary_json_path = out_dir / "phase1_cross_dataset_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved raw summary JSON: {summary_json_path}")

    # Generate Markdown Summary
    md_lines = [
        "# Phase 1: Cross-Dataset Zero-Shot & Normalization Adaptation Benchmark",
        "",
        "Empirical evaluation of German GTSRB-trained models evaluated across international road sign benchmarks.",
        "",
        "## 1. Downstream Linear Probe Representation Performance",
        "",
        "| Dataset | Paradigm | Normalization Regime | Linear Probe Top-1 Acc | Linear Probe Macro-F1 |",
        "| :--- | :--- | :--- | :---: | :---: |",
    ]

    for name, res in all_results.items():
        md_lines.append(f"| **{name}** | Baseline ResNet-18 | Static BN (Frozen German stats) | `{res['baseline_static']['linear_probe_accuracy']*100:.2f}%` | `{res['baseline_static']['linear_probe_macro_f1']*100:.2f}%` |")
        md_lines.append(f"| **{name}** | Baseline ResNet-18 | BN-Adapt (Test-time shift) | `{res['baseline_bn_adapt']['linear_probe_accuracy']*100:.2f}%` | `{res['baseline_bn_adapt']['linear_probe_macro_f1']*100:.2f}%` |")
        md_lines.append(f"| **{name}** | Concept Bottleneck Model | Static BN (Frozen German stats) | `{res['cbm_static']['linear_probe_accuracy']*100:.2f}%` | `{res['cbm_static']['linear_probe_macro_f1']*100:.2f}%` |")
        md_lines.append(f"| **{name}** | Concept Bottleneck Model | BN-Adapt (Test-time shift) | `{res['cbm_bn_adapt']['linear_probe_accuracy']*100:.2f}%` | `{res['cbm_bn_adapt']['linear_probe_macro_f1']*100:.2f}%` |")

    md_lines.extend([
        "",
        "## 2. Concept Bottleneck Model: Zero-Shot Perceptual Generality",
        "",
        "Evaluation of CBM's 4-axis concept heads (trained in Germany) predicting physical concepts across Belgium and China with **zero retraining**:",
        "",
        "| Dataset | Normalization | Shape Acc | Colour Acc | Border Acc | Symbol Acc | Mean Concept Acc | Exact 4-Concept Match | Zero-Shot Class Acc |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for name, res in all_results.items():
        cs = res["cbm_static"]
        ca = res["cbm_bn_adapt"]
        md_lines.append(
            f"| **{name}** | Static BN | `{cs['concept_shape_acc']*100:.1f}%` | `{cs['concept_colour_acc']*100:.1f}%` | "
            f"`{cs['concept_border_acc']*100:.1f}%` | `{cs['concept_symbol_acc']*100:.1f}%` | "
            f"**`{cs['concept_mean_acc']*100:.2f}%`** | `{cs['concept_exact_match_rate']*100:.2f}%` | `{cs['zero_shot_ontology_acc']*100:.2f}%` |"
        )
        md_lines.append(
            f"| **{name}** | BN-Adapt (Shift) | `{ca['concept_shape_acc']*100:.1f}%` | `{ca['concept_colour_acc']*100:.1f}%` | "
            f"`{ca['concept_border_acc']*100:.1f}%` | `{ca['concept_symbol_acc']*100:.1f}%` | "
            f"**`{ca['concept_mean_acc']*100:.2f}%`** | `{ca['concept_exact_match_rate']*100:.2f}%` | `{ca['zero_shot_ontology_acc']*100:.2f}%` |"
        )

    summary_md_path = out_dir / "phase1_cross_dataset_summary.md"
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Saved summary Markdown report: {summary_md_path}")
    print("\nPhase 1 cross-dataset benchmark completed successfully!")


if __name__ == "__main__":
    main()
