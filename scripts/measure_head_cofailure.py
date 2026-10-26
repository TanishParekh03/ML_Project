"""
Measure Head Co-Failure Rate (Class Head vs Concept Heads)
===========================================================
Addresses Reviewer Point 7:
Directly tests whether the class head and concept heads fail together on the
exact same images under corruption (proving or refuting the shared-backbone
synchronous failure mechanism).

Computes:
- Contingency table:
  * N_00: Class Wrong AND Concept Tuple Wrong (Synchronous Failure)
  * N_01: Class Wrong AND Concept Tuple Correct (Decodable / Rescuable)
  * N_10: Class Correct AND Concept Tuple Wrong (Robust Class / Fragile Concept)
  * N_11: Class Correct AND Concept Tuple Correct (Clean Concordance)
- P(Concept Tuple Wrong | Class Wrong): The exact co-failure conditional probability.
- Chi-square test of independence between Class correctness and Concept correctness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.stats import chi2_contingency
from tqdm import tqdm

from src.corruptions import CORRUPTIONS_REGISTRY
from src.dataset import IMAGENET_MEAN, IMAGENET_STD, letterbox_roi_crop
from src.models.cbm import ConceptBottleneckModel, get_concept_lookup_tensors
from src.models.concept_encoder import load_class_concepts

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_cbm(checkpoint_path: str, device: torch.device) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    bottleneck_mode = config.get("bottleneck_mode", "residual")
    model = ConceptBottleneckModel(num_classes=43, pretrained=False, bottleneck_mode=bottleneck_mode)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def measure_cofailure(
    model: nn.Module,
    test_df: pd.DataFrame,
    data_root: Path,
    device: torch.device,
    concept_luts: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    corruptions: List[str],
    severities: List[int],
    batch_size: int = 128,
) -> Dict[str, Any]:
    shape_lut, colour_lut, border_lut, symcat_lut = concept_luts
    all_classes = torch.arange(43, device=device)
    gt_s = shape_lut[all_classes]
    gt_c = colour_lut[all_classes]
    gt_b = border_lut[all_classes]
    gt_m = symcat_lut[all_classes]

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

    total_N00 = 0  # Class Wrong & Concept Wrong
    total_N01 = 0  # Class Wrong & Concept Correct
    total_N10 = 0  # Class Correct & Concept Wrong
    total_N11 = 0  # Class Correct & Concept Correct

    per_condition_stats = []

    for c_name in corruptions:
        c_fn = CORRUPTIONS_REGISTRY[c_name]
        for sev in severities:
            preds_class = []
            preds_concept_tuple_correct = []

            for start in range(0, len(clean_items), batch_size):
                items = clean_items[start : start + batch_size]
                tensors = []
                b_targets = []
                for canvas, t_class in items:
                    corrupted = c_fn(canvas, sev)
                    t = TF.normalize(TF.to_tensor(corrupted), mean=IMAGENET_MEAN, std=IMAGENET_STD)
                    tensors.append(t)
                    b_targets.append(t_class)

                b_x = torch.stack(tensors).to(device)
                b_y = torch.tensor(b_targets, device=device)

                out = model(b_x)
                class_pred = torch.argmax(out["logits"], dim=-1)

                # Concept predictions
                pred_s = torch.argmax(out["logits_shape"], dim=-1)
                pred_c = torch.argmax(out["logits_colour"], dim=-1)
                pred_b = torch.argmax(out["logits_border"], dim=-1)
                pred_m = torch.argmax(out["logits_symcat"], dim=-1)

                true_s = gt_s[b_y]
                true_c = gt_c[b_y]
                true_b = gt_b[b_y]
                true_m = gt_m[b_y]

                tuple_correct = (pred_s == true_s) & (pred_c == true_c) & (pred_b == true_b) & (pred_m == true_m)

                preds_class.extend(class_pred.cpu().numpy())
                preds_concept_tuple_correct.extend(tuple_correct.cpu().numpy())

            y_class_pred = np.array(preds_class)
            cpt_correct = np.array(preds_concept_tuple_correct)
            class_correct = (y_class_pred == targets_arr)

            N00 = int(np.sum((~class_correct) & (~cpt_correct)))
            N01 = int(np.sum((~class_correct) & cpt_correct))
            N10 = int(np.sum(class_correct & (~cpt_correct)))
            N11 = int(np.sum(class_correct & cpt_correct))

            total_N00 += N00
            total_N01 += N01
            total_N10 += N10
            total_N11 += N11

            n_class_wrong = N00 + N01
            p_cpt_wrong_given_class_wrong = (N00 / n_class_wrong * 100) if n_class_wrong > 0 else 0.0

            per_condition_stats.append({
                "corruption": c_name,
                "severity": sev,
                "class_acc": float(np.mean(class_correct)),
                "concept_tuple_acc": float(np.mean(cpt_correct)),
                "N00_both_wrong": N00,
                "N01_class_wrong_cpt_correct": N01,
                "N10_class_correct_cpt_wrong": N10,
                "N11_both_correct": N11,
                "cofailure_rate_percent": float(p_cpt_wrong_given_class_wrong),
            })

            print(
                f"[{c_name:16s} S{sev}] Class Acc: {np.mean(class_correct)*100:5.2f}% | "
                f"Tuple Acc: {np.mean(cpt_correct)*100:5.2f}% | "
                f"Both Wrong: {N00:4d} | Class Wrong but Cpt Correct: {N01:4d} | "
                f"Co-Failure Rate: {p_cpt_wrong_given_class_wrong:5.1f}%"
            )

    table = np.array([[total_N00, total_N01], [total_N10, total_N11]])
    chi2, p_val, _, _ = chi2_contingency(table)
    overall_class_wrong = total_N00 + total_N01
    overall_cofailure_rate = (total_N00 / overall_class_wrong * 100) if overall_class_wrong > 0 else 0.0

    summary = {
        "contingency_table": {
            "N00_both_wrong": total_N00,
            "N01_class_wrong_concept_correct": total_N01,
            "N10_class_correct_concept_wrong": total_N10,
            "N11_both_correct": total_N11,
        },
        "overall_cofailure_rate_percent": float(overall_cofailure_rate),
        "chi2_statistic": float(chi2),
        "chi2_p_value": float(p_val),
        "per_condition": per_condition_stats,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Measure Head Co-Failure Rate (Class vs Concept).")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="german")
    parser.add_argument("--meta-path", type=str, default="data/master_v1/metadata/test.csv")
    parser.add_argument("--concepts-csv", type=str, default="data/master_v1/concepts/class_concepts.csv")
    parser.add_argument("--output-json", type=str, default="reports/diagnostics/head_cofailure_analysis.json")
    parser.add_argument("--corruptions", type=str, default="gaussian_noise,shot_noise,impulse_noise")
    parser.add_argument("--severities", type=str, default="3,4,5")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta_path = Path(args.meta_path)
    if not meta_path.exists() and (PROJECT_ROOT / meta_path).exists():
        meta_path = PROJECT_ROOT / meta_path

    concepts_csv = Path(args.concepts_csv)
    if not concepts_csv.exists() and (PROJECT_ROOT / concepts_csv).exists():
        concepts_csv = PROJECT_ROOT / concepts_csv

    data_root = Path(args.data_root)
    if not data_root.exists() and (PROJECT_ROOT / data_root).exists():
        data_root = PROJECT_ROOT / data_root

    model = load_cbm(args.checkpoint, device)
    concept_luts = get_concept_lookup_tensors(device=device, csv_path=str(concepts_csv))
    test_df = pd.read_csv(meta_path)

    corruptions = [c.strip() for c in args.corruptions.split(",")]
    severities = [int(s.strip()) for s in args.severities.split(",")]

    summary = measure_cofailure(
        model=model,
        test_df=test_df,
        data_root=data_root,
        device=device,
        concept_luts=concept_luts,
        corruptions=corruptions,
        severities=severities,
        batch_size=args.batch_size,
    )

    out_p = Path(args.output_json)
    if not out_p.is_absolute():
        out_p = PROJECT_ROOT / out_p
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("CLASS VS CONCEPT HEAD CO-FAILURE ANALYSIS:")
    ct = summary["contingency_table"]
    print(f"  Both Wrong (N00)                  : {ct['N00_both_wrong']:,}")
    print(f"  Class Wrong, Concept Correct (N01): {ct['N01_class_wrong_concept_correct']:,}")
    print(f"  Class Correct, Concept Wrong (N10): {ct['N10_class_correct_concept_wrong']:,}")
    print(f"  Both Correct (N11)                : {ct['N11_both_correct']:,}")
    print(f"  P(Concept Wrong | Class Wrong)    : {summary['overall_cofailure_rate_percent']:.2f}%")
    print(f"  Chi-square Statistic (p-value)   : {summary['chi2_statistic']:.1f} (p = {summary['chi2_p_value']:.2e})")
    print(f"Report saved to: {out_p}")
    print("=" * 80)


if __name__ == "__main__":
    main()

