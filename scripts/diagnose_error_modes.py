"""
Diagnostic: Error Mode Breakdown for GTSRB CBM under Corruption
===============================================================
Measures the distribution of misclassifications under corruption:
1. Within-Group Errors: y_pred != y_true, but Concept(y_pred) == Concept(y_true)
   (e.g., Speed 30 confused with Speed 50, or Pedestrian confused with Road Work).
   -> Invariant under ontology-consistent decoding.
2. Cross-Group Errors: y_pred != y_true, AND Concept(y_pred) != Concept(y_true)
   (e.g., Speed 30 confused with Turn Right or Priority Road).
   -> Can be rescued by Ontology-Consistent Decoding (OCS).

Also evaluates the baseline Ontology-Consistent Decoding score:
   score(y|x) = log p(y|x) + beta * sum_k log q_k(c_k(y)|x)
"""

from __future__ import annotations

import argparse
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
from src.models.cbm import ConceptBottleneckModel, get_concept_lookup_tensors
from src.models.concept_encoder import encode_all_classes, load_class_concepts


def load_cbm_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    bottleneck_mode = config.get("bottleneck_mode", "residual")

    model = ConceptBottleneckModel(num_classes=43, pretrained=False, bottleneck_mode=bottleneck_mode)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def build_concept_lookup_matrix(concepts_csv: Path, device: torch.device):
    """
    Builds lookup tables mapping class_id (0..42) to:
    - shape index (0..5)
    - colour index (0..4)
    - border index (0..3)
    - symcat index (0..6)
    """
    df = load_class_concepts(concepts_csv)
    shape_lut, colour_lut, border_lut, symcat_lut = get_concept_lookup_tensors(df, device)
    
    # Also build a string tuple signature for exact within/cross group check
    tuple_signatures = {}
    for _, row in df.iterrows():
        cid = int(row["class_id"])
        sig = (row["shape"], row["colour_profile"], row["border_type"], row["symbol_category"])
        tuple_signatures[cid] = sig

    return (shape_lut, colour_lut, border_lut, symcat_lut), tuple_signatures


@torch.no_grad()
def run_diagnostic(
    model: nn.Module,
    test_df: pd.DataFrame,
    data_root: Path,
    device: torch.device,
    concept_luts: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    tuple_signatures: Dict[int, Tuple[str, str, str, str]],
    corruptions_to_test: List[str],
    severities: List[int],
    beta: float = 1.0,
    batch_size: int = 128,
) -> Dict[str, Any]:
    shape_lut, colour_lut, border_lut, symcat_lut = concept_luts
    
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
    
    # Pre-map all 43 classes to their concept indices [43]
    all_classes = torch.arange(43, device=device)
    c_shape_idx = shape_lut[all_classes]   # [43]
    c_colour_idx = colour_lut[all_classes] # [43]
    c_border_idx = border_lut[all_classes] # [43]
    c_symcat_idx = symcat_lut[all_classes] # [43]

    results_by_condition = {}
    total_errors_all = 0
    within_errors_all = 0
    cross_errors_all = 0
    rescued_by_ocs_all = 0
    harmed_by_ocs_all = 0

    for c_name in corruptions_to_test:
        c_func = CORRUPTIONS_REGISTRY[c_name]
        for sev in severities:
            condition_key = f"{c_name}_sev{sev}"
            
            raw_preds: List[int] = []
            ocs_preds: List[int] = []

            for start in range(0, len(clean_items), batch_size):
                batch_items = clean_items[start : start + batch_size]
                tensors = []
                for canvas, _ in batch_items:
                    corrupted = c_func(canvas, sev)
                    t = TF.normalize(TF.to_tensor(corrupted), mean=IMAGENET_MEAN, std=IMAGENET_STD)
                    tensors.append(t)

                batch_t = torch.stack(tensors).to(device)
                outputs = model(batch_t)
                
                logits = outputs["logits"]  # [B, 43]
                p_y = F.log_softmax(logits, dim=-1) # [B, 43]

                # Concept log probabilities
                log_q_shape = F.log_softmax(outputs["logits_shape"], dim=-1)   # [B, 6]
                log_q_colour = F.log_softmax(outputs["logits_colour"], dim=-1) # [B, 5]
                log_q_border = F.log_softmax(outputs["logits_border"], dim=-1) # [B, 4]
                log_q_symcat = F.log_softmax(outputs["logits_symcat"], dim=-1) # [B, 7]

                # Compute concept log prior for each of the 43 candidate classes [B, 43]
                concept_log_prior = (
                    log_q_shape[:, c_shape_idx]
                    + log_q_colour[:, c_colour_idx]
                    + log_q_border[:, c_border_idx]
                    + log_q_symcat[:, c_symcat_idx]
                )

                # Standard Prediction
                batch_raw_preds = torch.argmax(logits, dim=-1).cpu().numpy()
                raw_preds.extend(batch_raw_preds)

                # Ontology-Consistent Decoding Prediction
                ocs_score = p_y + beta * concept_log_prior
                batch_ocs_preds = torch.argmax(ocs_score, dim=-1).cpu().numpy()
                ocs_preds.extend(batch_ocs_preds)

            raw_preds_arr = np.array(raw_preds)
            ocs_preds_arr = np.array(ocs_preds)

            raw_acc = accuracy_score(targets_arr, raw_preds_arr)
            ocs_acc = accuracy_score(targets_arr, ocs_preds_arr)

            # Analyze error breakdown
            error_mask = (raw_preds_arr != targets_arr)
            n_errors = int(np.sum(error_mask))
            
            n_within = 0
            n_cross = 0
            rescued_by_ocs = 0
            harmed_by_ocs = 0

            for idx in np.where(error_mask)[0]:
                y_true = int(targets_arr[idx])
                y_pred = int(raw_preds_arr[idx])
                y_ocs = int(ocs_preds_arr[idx])

                sig_true = tuple_signatures[y_true]
                sig_pred = tuple_signatures[y_pred]

                if sig_true == sig_pred:
                    n_within += 1
                else:
                    n_cross += 1

                if y_ocs == y_true:
                    rescued_by_ocs += 1

            for idx in np.where(~error_mask)[0]:
                y_true = int(targets_arr[idx])
                y_ocs = int(ocs_preds_arr[idx])
                if y_ocs != y_true:
                    harmed_by_ocs += 1

            within_ratio = (n_within / n_errors) * 100 if n_errors > 0 else 0.0
            cross_ratio = (n_cross / n_errors) * 100 if n_errors > 0 else 0.0

            results_by_condition[condition_key] = {
                "raw_accuracy": float(raw_acc),
                "ocs_accuracy": float(ocs_acc),
                "delta_accuracy": float(ocs_acc - raw_acc),
                "total_errors": n_errors,
                "within_group_errors": n_within,
                "cross_group_errors": n_cross,
                "within_ratio_percent": float(within_ratio),
                "cross_ratio_percent": float(cross_ratio),
                "rescued_samples": rescued_by_ocs,
                "harmed_samples": harmed_by_ocs,
            }

            total_errors_all += n_errors
            within_errors_all += n_within
            cross_errors_all += n_cross
            rescued_by_ocs_all += rescued_by_ocs
            harmed_by_ocs_all += harmed_by_ocs

            print(
                f"[{condition_key:22s}] Acc: {raw_acc*100:5.2f}% -> OCS: {ocs_acc*100:5.2f}% "
                f"(Δ {((ocs_acc-raw_acc)*100):+4.2f}%) | "
                f"Errors: {n_errors:4d} | Within: {within_ratio:5.1f}% | Cross: {cross_ratio:5.1f}% | "
                f"Rescued: {rescued_by_ocs:3d} | Harmed: {harmed_by_ocs:3d}"
            )

    overall_within_ratio = (within_errors_all / total_errors_all) * 100 if total_errors_all > 0 else 0.0
    overall_cross_ratio = (cross_errors_all / total_errors_all) * 100 if total_errors_all > 0 else 0.0

    summary = {
        "total_errors_evaluated": total_errors_all,
        "within_group_errors": within_errors_all,
        "cross_group_errors": cross_errors_all,
        "overall_within_ratio_percent": float(overall_within_ratio),
        "overall_cross_ratio_percent": float(overall_cross_ratio),
        "total_rescued_by_ocs": rescued_by_ocs_all,
        "total_harmed_by_ocs": harmed_by_ocs_all,
        "net_gain_samples": rescued_by_ocs_all - harmed_by_ocs_all,
        "per_condition": results_by_condition,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Diagnose CBM error modes and test Ontology-Consistent Decoding.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_cbm_resnet18.pt")
    parser.add_argument("--data-root", type=str, default="german", help="Path to raw GTSRB dataset")
    parser.add_argument("--meta-path", type=str, default="data/master_v1/metadata/test.csv", help="Path to test.csv")
    parser.add_argument(
        "--concepts-csv",
        type=str,
        default="data/master_v1/concepts/class_concepts.csv",
        help="Path to class_concepts.csv",
    )
    parser.add_argument("--output-json", type=str, default="reports/diagnostics/error_modes_diagnostic.json")
    parser.add_argument(
        "--corruptions",
        type=str,
        default="gaussian_noise,shot_noise,impulse_noise",
        help="Comma-separated corruptions to evaluate (or 'all')",
    )
    parser.add_argument(
        "--severities",
        type=str,
        default="3,4,5",
        help="Comma-separated severities to evaluate (e.g., '3,4,5' or '1,2,3,4,5')",
    )
    parser.add_argument("--beta", type=float, default=1.0, help="Weight for concept log prior in OCS decoding")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("GTSRB CBM ERROR MODE DIAGNOSTIC & ONTOLOGY-CONSISTENT DECODING (OCS)")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Device     : {device}")
    print(f"Beta       : {args.beta}")
    print("=" * 80)

    model = load_cbm_model(args.checkpoint, device)
    concept_luts, tuple_signatures = build_concept_lookup_matrix(Path(args.concepts_csv), device)
    test_df = pd.read_csv(args.meta_path)

    if args.corruptions == "all":
        corruptions_to_test = list(CORRUPTIONS_REGISTRY.keys())
    else:
        corruptions_to_test = [c.strip() for c in args.corruptions.split(",")]

    severities = [int(s.strip()) for s in args.severities.split(",")]

    summary = run_diagnostic(
        model=model,
        test_df=test_df,
        data_root=Path(args.data_root),
        device=device,
        concept_luts=concept_luts,
        tuple_signatures=tuple_signatures,
        corruptions_to_test=corruptions_to_test,
        severities=severities,
        beta=args.beta,
        batch_size=args.batch_size,
    )

    out_p = Path(args.output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY ACROSS EVALUATED CONDITIONS:")
    print(f"Total Errors Evaluated : {summary['total_errors_evaluated']:,}")
    print(f"Within-Group Errors    : {summary['within_group_errors']:,} ({summary['overall_within_ratio_percent']:.1f}%)")
    print(f"Cross-Group Errors     : {summary['cross_group_errors']:,} ({summary['overall_cross_ratio_percent']:.1f}%)")
    print(f"Samples Rescued by OCS : {summary['total_rescued_by_ocs']:,}")
    print(f"Samples Harmed by OCS  : {summary['total_harmed_by_ocs']:,}")
    print(f"Net Gain (OCS vs Raw)  : {summary['net_gain_samples']:+,} samples")
    print(f"Report saved to        : {out_p}")
    print("=" * 80)


if __name__ == "__main__":
    main()
