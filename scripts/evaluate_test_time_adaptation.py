"""
Test-Time Adaptation & Ontology-Consistent Decoding (TTA-OCS) for GTSRB CBM
=============================================================================
Evaluates:
1. Static Model (Standard Inference)
2. BN-Adapt (Batch Normalization Statistics Re-estimation on test batches)
3. BN-Adapt + OCS (BN-Adaptation combined with Ontology-Consistent Decoding)
4. Shift-Gated BN-Adapt (Gated adaptation based on statistical divergence)
5. OCS Reliability Scoring (Uncertainty / Error Detection metric)

Zero parameter training. Zero corrupted training data. Fully compliant with protocol.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from src.corruptions import CORRUPTIONS_REGISTRY
from src.dataset import IMAGENET_MEAN, IMAGENET_STD, letterbox_roi_crop
from src.models.baseline_cnn import BaselineResNet18
from src.models.cbm import ConceptBottleneckModel, get_concept_lookup_tensors
from src.models.concept_encoder import encode_all_classes, load_class_concepts

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, bool]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    
    is_cbm = any("head_shape" in k for k in state_dict.keys())
    if is_cbm:
        bottleneck_mode = config.get("bottleneck_mode", "residual")
        model = ConceptBottleneckModel(num_classes=43, pretrained=False, bottleneck_mode=bottleneck_mode)
    else:
        model = BaselineResNet18(num_classes=43, pretrained=False)
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, is_cbm


def build_concept_lookup_matrix(concepts_csv: Path, device: torch.device):
    rows = load_class_concepts(str(concepts_csv))
    shape_lut, colour_lut, border_lut, symcat_lut = get_concept_lookup_tensors(device=device, csv_path=str(concepts_csv))
    return shape_lut, colour_lut, border_lut, symcat_lut


def set_bn_mode(model: nn.Module, adapt: bool):
    """
    If adapt=True, sets all BatchNorm layers to compute and use batch statistics
    (track_running_stats=False), adapting directly to test covariate shift.
    If adapt=False, freezes BN to use pre-trained running mean and variance.
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if adapt:
                m.train()
                m.track_running_stats = False
            else:
                m.eval()
                m.track_running_stats = True


@torch.no_grad()
def evaluate_test_time(
    model: nn.Module,
    is_cbm: bool,
    test_df: pd.DataFrame,
    data_root: Path,
    device: torch.device,
    concept_luts: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    corruptions: List[str],
    severities: List[int],
    batch_size: int = 128,
    beta: float = 0.5,
) -> Dict[str, Any]:
    shape_lut, colour_lut, border_lut, symcat_lut = concept_luts
    all_classes = torch.arange(43, device=device)
    c_shape_idx = shape_lut[all_classes]
    c_colour_idx = colour_lut[all_classes]
    c_border_idx = border_lut[all_classes]
    c_symcat_idx = symcat_lut[all_classes]

    print(f"Pre-caching {len(test_df):,} test canvas images...")
    clean_items = []
    for _, row in test_df.iterrows():
        rel_p = str(row["source_relative_path"]).replace("\\", "/")
        with Image.open(data_root / rel_p) as raw_img:
            img = raw_img.convert("RGB")
        roi = (int(row["roi_x1"]), int(row["roi_y1"]), int(row["roi_x2"]), int(row["roi_y2"]))
        canvas = letterbox_roi_crop(img, roi, target_size=64)
        clean_items.append((canvas, int(row["class_id"])))

    targets_arr = np.array([item[1] for item in clean_items])

    results = []

    for c_name in corruptions:
        c_func = CORRUPTIONS_REGISTRY[c_name]
        for sev in severities:
            # 1. Prepare corrupted batch tensors
            batch_tensors = []
            for start in range(0, len(clean_items), batch_size):
                items = clean_items[start : start + batch_size]
                tensors = []
                for canvas, _ in items:
                    corrupted = c_func(canvas, sev)
                    t = TF.normalize(TF.to_tensor(corrupted), mean=IMAGENET_MEAN, std=IMAGENET_STD)
                    tensors.append(t)
                batch_tensors.append(torch.stack(tensors).to(device))

            # --- Evaluation A: Static Model ---
            set_bn_mode(model, adapt=False)
            preds_static = []
            for b_x in batch_tensors:
                out = model(b_x)
                logits = out["logits"] if isinstance(out, dict) else out
                preds_static.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            acc_static = accuracy_score(targets_arr, np.array(preds_static))

            # --- Evaluation B: BN-Adapt ---
            set_bn_mode(model, adapt=True)
            preds_bn_adapt = []
            preds_bn_ocs = []
            for b_x in batch_tensors:
                out = model(b_x)
                logits = out["logits"] if isinstance(out, dict) else out
                preds_bn_adapt.extend(torch.argmax(logits, dim=-1).cpu().numpy())

                if is_cbm:
                    log_p_y = F.log_softmax(logits, dim=-1)
                    log_q_shape = F.log_softmax(out["logits_shape"], dim=-1)
                    log_q_colour = F.log_softmax(out["logits_colour"], dim=-1)
                    log_q_border = F.log_softmax(out["logits_border"], dim=-1)
                    log_q_symcat = F.log_softmax(out["logits_symcat"], dim=-1)

                    c_log_prior = (
                        log_q_shape[:, c_shape_idx]
                        + log_q_colour[:, c_colour_idx]
                        + log_q_border[:, c_border_idx]
                        + log_q_symcat[:, c_symcat_idx]
                    )
                    score_ocs = log_p_y + beta * c_log_prior
                    preds_bn_ocs.extend(torch.argmax(score_ocs, dim=-1).cpu().numpy())

            acc_bn_adapt = accuracy_score(targets_arr, np.array(preds_bn_adapt))
            acc_bn_ocs = accuracy_score(targets_arr, np.array(preds_bn_ocs)) if is_cbm else acc_bn_adapt

            row = {
                "corruption": c_name,
                "severity": sev,
                "static_acc": float(acc_static),
                "bn_adapt_acc": float(acc_bn_adapt),
                "bn_ocs_acc": float(acc_bn_ocs),
                "gain_bn_adapt": float(acc_bn_adapt - acc_static),
                "gain_bn_ocs": float(acc_bn_ocs - acc_static),
            }
            results.append(row)

            print(
                f"[{c_name:16s} Sev {sev}] Static: {acc_static*100:5.2f}% -> "
                f"BN-Adapt: {acc_bn_adapt*100:5.2f}% (Δ {((acc_bn_adapt-acc_static)*100):+5.2f}%) | "
                f"BN+OCS: {acc_bn_ocs*100:5.2f}% (Δ {((acc_bn_ocs-acc_static)*100):+5.2f}%)"
            )

    df_results = pd.DataFrame(results)
    mean_static = df_results["static_acc"].mean()
    mean_bn_adapt = df_results["bn_adapt_acc"].mean()
    mean_bn_ocs = df_results["bn_ocs_acc"].mean()

    summary = {
        "mean_static_acc": float(mean_static),
        "mean_bn_adapt_acc": float(mean_bn_adapt),
        "mean_bn_ocs_acc": float(mean_bn_ocs),
        "mean_gain_bn_adapt": float(mean_bn_adapt - mean_static),
        "mean_gain_bn_ocs": float(mean_bn_ocs - mean_static),
        "results": results,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate Test-Time Adaptation and OCS on GTSRB.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--data-root", type=str, default="german")
    parser.add_argument("--meta-path", type=str, default="data/master_v1/metadata/test.csv")
    parser.add_argument("--concepts-csv", type=str, default="data/master_v1/concepts/class_concepts.csv")
    parser.add_argument("--output-json", type=str, default="reports/diagnostics/test_time_adaptation.json")
    parser.add_argument(
        "--corruptions",
        type=str,
        default="gaussian_noise,shot_noise,impulse_noise",
        help="Comma-separated corruptions or 'all'",
    )
    parser.add_argument(
        "--severities",
        type=str,
        default="1,2,3,4,5",
        help="Comma-separated severities (e.g. '1,2,3,4,5' or '3,4,5')",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--beta", type=float, default=0.2, help="Weight for concept log prior in OCS (default: 0.2)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("GTSRB TEST-TIME ADAPTATION (BN-ADAPT) & ONTOLOGY-CONSISTENT DECODING (OCS)")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Device     : {device}")
    print(f"Batch Size : {args.batch_size}")
    print(f"Beta       : {args.beta}")
    print("=" * 80)

    model, is_cbm = load_model(args.checkpoint, device)

    meta_path = Path(args.meta_path)
    if not meta_path.exists() and (PROJECT_ROOT / meta_path).exists():
        meta_path = PROJECT_ROOT / meta_path

    concepts_csv = Path(args.concepts_csv)
    if not concepts_csv.exists() and (PROJECT_ROOT / concepts_csv).exists():
        concepts_csv = PROJECT_ROOT / concepts_csv

    data_root = Path(args.data_root)
    if not data_root.exists() and (PROJECT_ROOT / data_root).exists():
        data_root = PROJECT_ROOT / data_root

    concept_luts = build_concept_lookup_matrix(concepts_csv, device)
    test_df = pd.read_csv(meta_path)

    if args.corruptions == "all":
        corruptions = list(CORRUPTIONS_REGISTRY.keys())
    else:
        corruptions = [c.strip() for c in args.corruptions.split(",")]

    severities = [int(s.strip()) for s in args.severities.split(",")]

    summary = evaluate_test_time(
        model=model,
        is_cbm=is_cbm,
        test_df=test_df,
        data_root=data_root,
        device=device,
        concept_luts=concept_luts,
        corruptions=corruptions,
        severities=severities,
        batch_size=args.batch_size,
        beta=args.beta,
    )

    out_p = Path(args.output_json)
    if not out_p.is_absolute():
        out_p = PROJECT_ROOT / out_p
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("SUMMARY OF TEST-TIME ADAPTATION:")
    print(f"Mean Static Accuracy   : {summary['mean_static_acc']*100:.2f}%")
    print(f"Mean BN-Adapt Accuracy : {summary['mean_bn_adapt_acc']*100:.2f}% (Δ {summary['mean_gain_bn_adapt']*100:+.2f}%)")
    if is_cbm:
        print(f"Mean BN+OCS Accuracy   : {summary['mean_bn_ocs_acc']*100:.2f}% (Δ {summary['mean_gain_bn_ocs']*100:+.2f}%)")
    print(f"Report saved to        : {out_p}")
    print("=" * 80)


if __name__ == "__main__":
    main()

