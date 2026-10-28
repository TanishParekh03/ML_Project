"""
Empirical Evaluation of Test-Time Counterfactual Concept Interventions (intervene())
==================================================================================
Evaluates:
- Mode 0: No Intervention (Standard Predicted Concepts c_hat)
- Mode 1: Single-Axis Clamping (Shape only clamped to ground-truth c*)
- Mode 2: Single-Axis Clamping (Colour only clamped to ground-truth c*)
- Mode 3: Single-Axis Clamping (Border only clamped to ground-truth c*)
- Mode 4: Single-Axis Clamping (Symbol Category only clamped to ground-truth c*)
- Mode 5: Macroscopic V2I Prior Clamping (Shape + Colour + Border clamped to c*)
- Mode 6: Full 4-Axis Concept Clamping (All 22 concept dims clamped to c*)

Evaluates Modes 0-6 under both:
1. Static BN
2. Shift-Gated BN-Adapt

Partitions errors into:
- Within-Group Errors: C(y_pred) == C(y_true)
- Cross-Group Errors:  C(y_pred) != C(y_true)

Calculates:
- Intervention Recovery Rate (IR): (Acc_after - Acc_before) / (1 - Acc_before)
- Concept Degradation Rate (CDR) for Shape, Colour, Border, Symbol Category under Static vs BN-Adapt.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Fix Windows console encoding for Unicode symbols
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.corruptions import CORRUPTIONS_REGISTRY
from src.dataset import IMAGENET_MEAN, IMAGENET_STD, letterbox_roi_crop
from src.models.cbm import ConceptBottleneckModel, get_concept_lookup_tensors


def set_bn_mode(model: nn.Module, adapt: bool):
    """Sets BN layers to train/adapt (track_running_stats=False) or eval (frozen stats)."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if adapt:
                m.train()
                m.track_running_stats = False
            else:
                m.eval()
                m.track_running_stats = True


def compute_layer_divergence(model: nn.Module, batch_x: torch.Tensor) -> float:
    """Computes standardized layer divergence D across BatchNorm layers."""
    divergences = []
    handles = []

    def hook_fn(module, input):
        x = input[0]
        if isinstance(module, nn.BatchNorm2d):
            b_mean = x.mean(dim=[0, 2, 3])
        elif isinstance(module, nn.BatchNorm1d):
            b_mean = x.mean(dim=0)
        else:
            return
        s_mean = module.running_mean
        s_var = module.running_var
        if s_mean is not None and s_var is not None:
            diff_norm = torch.norm(b_mean - s_mean, p=2)
            denom = torch.sqrt(torch.sum(s_var) + 1e-7)
            d = (diff_norm / denom).item()
            divergences.append(d)

    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            handles.append(m.register_forward_pre_hook(hook_fn))

    with torch.no_grad():
        _ = model(batch_x)

    for h in handles:
        h.remove()

    return float(np.mean(divergences)) if len(divergences) > 0 else 0.0


def construct_intervened_vector(
    c_hat: torch.Tensor,
    mode: int,
    t_shape: torch.Tensor,
    t_colour: torch.Tensor,
    t_border: torch.Tensor,
    t_symcat: torch.Tensor,
) -> torch.Tensor | None:
    """Constructs intervened 22-dimensional concept vector for Modes 0 to 6."""
    if mode == 0:
        return None  # Standard model prediction

    c_vec = c_hat.clone()
    oh_shape = F.one_hot(t_shape, 6).float()
    oh_colour = F.one_hot(t_colour, 5).float()
    oh_border = F.one_hot(t_border, 4).float()
    oh_symcat = F.one_hot(t_symcat, 7).float()

    if mode == 1:  # Shape only
        c_vec[:, 0:6] = oh_shape
    elif mode == 2:  # Colour only
        c_vec[:, 6:11] = oh_colour
    elif mode == 3:  # Border only
        c_vec[:, 11:15] = oh_border
    elif mode == 4:  # Symbol only
        c_vec[:, 15:22] = oh_symcat
    elif mode == 5:  # Macroscopic V2I prior (Shape + Colour + Border)
        c_vec[:, 0:6] = oh_shape
        c_vec[:, 6:11] = oh_colour
        c_vec[:, 11:15] = oh_border
    elif mode == 6:  # Full 4-axis concept clamping
        c_vec = torch.cat([oh_shape, oh_colour, oh_border, oh_symcat], dim=-1)

    return c_vec


@torch.no_grad()
def run_intervention_benchmark(
    checkpoint_path: str,
    meta_path: str = "data/master_v1/metadata/test.csv",
    data_root: str = "german",
    output_path: str = "reports/diagnostics/concept_intervention_benchmark.json",
    batch_size: int = 128,
    tau_shift: float = 0.10,
    quick_limit: int | None = None,
    device_str: str | None = None,
) -> Dict[str, Any]:
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    meta_p = Path(meta_path).resolve()
    data_p = Path(data_root).resolve()

    test_df = pd.read_csv(meta_p)
    if quick_limit is not None and quick_limit < len(test_df):
        print(f"Applying quick limit: {quick_limit} samples")
        test_df = test_df.sample(n=quick_limit, random_state=42).reset_index(drop=True)

    total_samples = len(test_df)
    print("=" * 80)
    print("CONCEPT INTERVENTION BENCHMARK (intervene() Modes 0 to 6)")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Device     : {device} | Test samples: {total_samples:,} | Batch size: {batch_size}")
    print(f"Output     : {out_file}")
    print("=" * 80)

    # 1. Load Model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    bottleneck_mode = config.get("bottleneck_mode", "residual")

    model = ConceptBottleneckModel(num_classes=43, pretrained=False, bottleneck_mode=bottleneck_mode)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 2. Concept Lookups
    shape_lut, colour_lut, border_lut, symcat_lut = get_concept_lookup_tensors(device=device)

    # 3. Pre-cache images
    print(f"Pre-caching {total_samples:,} images into memory...")
    clean_items = []
    for _, row in test_df.iterrows():
        rel_p = str(row["source_relative_path"]).replace("\\", "/")
        with Image.open(data_p / rel_p) as raw_img:
            img = raw_img.convert("RGB")
        roi = (int(row["roi_x1"]), int(row["roi_y1"]), int(row["roi_x2"]), int(row["roi_y2"]))
        canvas = letterbox_roi_crop(img, roi, target_size=64)
        clean_items.append((canvas, int(row["class_id"])))

    targets_class = np.array([item[1] for item in clean_items], dtype=np.int64)
    targets_class_t = torch.tensor(targets_class, device=device)
    targets_shape = shape_lut[targets_class_t].cpu().numpy()
    targets_colour = colour_lut[targets_class_t].cpu().numpy()
    targets_border = border_lut[targets_class_t].cpu().numpy()
    targets_symcat = symcat_lut[targets_class_t].cpu().numpy()

    # 4. Clean Evaluation Baseline for CDR
    clean_batch_tensors = []
    for start in range(0, total_samples, batch_size):
        items = clean_items[start : start + batch_size]
        tensors = [TF.normalize(TF.to_tensor(c), mean=IMAGENET_MEAN, std=IMAGENET_STD) for c, _ in items]
        clean_batch_tensors.append(torch.stack(tensors).to(device))

    set_bn_mode(model, adapt=False)
    clean_preds_class = []
    clean_preds_shape = []
    clean_preds_colour = []
    clean_preds_border = []
    clean_preds_symcat = []

    for bx in clean_batch_tensors:
        out = model(bx)
        clean_preds_class.extend(torch.argmax(out["logits"], dim=-1).cpu().numpy())
        clean_preds_shape.extend(torch.argmax(out["logits_shape"], dim=-1).cpu().numpy())
        clean_preds_colour.extend(torch.argmax(out["logits_colour"], dim=-1).cpu().numpy())
        clean_preds_border.extend(torch.argmax(out["logits_border"], dim=-1).cpu().numpy())
        clean_preds_symcat.extend(torch.argmax(out["logits_symcat"], dim=-1).cpu().numpy())

    clean_f1_shape = f1_score(targets_shape, np.array(clean_preds_shape), average="macro", zero_division=0)
    clean_f1_colour = f1_score(targets_colour, np.array(clean_preds_colour), average="macro", zero_division=0)
    clean_f1_border = f1_score(targets_border, np.array(clean_preds_border), average="macro", zero_division=0)
    clean_f1_symcat = f1_score(targets_symcat, np.array(clean_preds_symcat), average="macro", zero_division=0)
    clean_acc_mode0 = accuracy_score(targets_class, np.array(clean_preds_class))

    print(f"Clean Acc (Mode 0) : {clean_acc_mode0 * 100:.2f}%")
    print(f"Clean Concept F1   : Shape {clean_f1_shape*100:.1f}% | Colour {clean_f1_colour*100:.1f}% | Border {clean_f1_border*100:.1f}% | Symbol {clean_f1_symcat*100:.1f}%")

    # 5. Sweep over Severities 3, 4, 5 across all 15 Corruptions
    corruptions_list = [
        "gaussian_noise", "shot_noise", "impulse_noise",
        "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
        "snow", "frost", "fog", "brightness",
        "contrast", "elastic_transform", "pixelate", "jpeg_compression"
    ]
    severities_eval = [3, 4, 5]

    mode_names = [
        "Mode 0 (No Intervention)",
        "Mode 1 (Shape Clamped)",
        "Mode 2 (Colour Clamped)",
        "Mode 3 (Border Clamped)",
        "Mode 4 (Symbol Clamped)",
        "Mode 5 (Macroscopic V2I Clamped)",
        "Mode 6 (Full 4-Axis Clamped)",
    ]

    intervention_results = []
    cdr_records = []

    print("\n--- Evaluating Modes 0 to 6 across Severities 3, 4, 5 ---")
    pbar = tqdm(total=len(corruptions_list) * len(severities_eval), desc="Intervention Sweep")

    for c_name in corruptions_list:
        c_func = CORRUPTIONS_REGISTRY[c_name]
        for sev in severities_eval:
            # Prepare corrupted batches
            batch_tensors = []
            for start in range(0, total_samples, batch_size):
                items = clean_items[start : start + batch_size]
                tensors = [TF.normalize(TF.to_tensor(c_func(c, sev)), mean=IMAGENET_MEAN, std=IMAGENET_STD) for c, _ in items]
                batch_tensors.append(torch.stack(tensors).to(device))

            # Compute standardized divergence
            set_bn_mode(model, adapt=False)
            d_val = compute_layer_divergence(model, batch_tensors[0])
            use_adaptation = d_val >= tau_shift

            # Run for both Static BN and Shift-Gated BN-Adapt
            for adapt_flag in [False, True]:
                adapt_tag = "shift_gated_bn" if adapt_flag else "static_bn"
                if adapt_flag:
                    set_bn_mode(model, adapt=use_adaptation)
                else:
                    set_bn_mode(model, adapt=False)

                # First pass: collect predicted concepts c_hat and un-intervened logits
                preds_by_mode: Dict[int, List[int]] = {m: [] for m in range(7)}
                corrupt_preds_shape = []
                corrupt_preds_colour = []
                corrupt_preds_border = []
                corrupt_preds_symcat = []

                batch_offset = 0
                for bx in batch_tensors:
                    bs = bx.size(0)
                    t_lbl = targets_class_t[batch_offset : batch_offset + bs]
                    sh_gt = shape_lut[t_lbl]
                    col_gt = colour_lut[t_lbl]
                    bor_gt = border_lut[t_lbl]
                    sym_gt = symcat_lut[t_lbl]

                    # Standard forward to get c_hat and per-axis predictions
                    base_out = model(bx)
                    c_hat = base_out["c_hat"]
                    preds_by_mode[0].extend(torch.argmax(base_out["logits"], dim=-1).cpu().numpy())

                    corrupt_preds_shape.extend(torch.argmax(base_out["logits_shape"], dim=-1).cpu().numpy())
                    corrupt_preds_colour.extend(torch.argmax(base_out["logits_colour"], dim=-1).cpu().numpy())
                    corrupt_preds_border.extend(torch.argmax(base_out["logits_border"], dim=-1).cpu().numpy())
                    corrupt_preds_symcat.extend(torch.argmax(base_out["logits_symcat"], dim=-1).cpu().numpy())

                    # Forward passes for Modes 1..6
                    for m in range(1, 7):
                        c_inv = construct_intervened_vector(c_hat, m, sh_gt, col_gt, bor_gt, sym_gt)
                        out_inv = model(bx, c_oracle=c_inv)
                        preds_by_mode[m].extend(torch.argmax(out_inv["logits"], dim=-1).cpu().numpy())

                    batch_offset += bs

                # Accuracies for Modes 0..6
                acc_by_mode = {}
                for m in range(7):
                    acc_m = float(accuracy_score(targets_class, np.array(preds_by_mode[m])))
                    acc_by_mode[f"mode_{m}_acc"] = acc_m

                # Error decomposition for Mode 0 vs Mode 6
                mode0_preds = np.array(preds_by_mode[0])
                mode6_preds = np.array(preds_by_mode[6])

                pred_shapes = shape_lut[torch.tensor(mode0_preds, device=device)].cpu().numpy()
                pred_colours = colour_lut[torch.tensor(mode0_preds, device=device)].cpu().numpy()
                pred_borders = border_lut[torch.tensor(mode0_preds, device=device)].cpu().numpy()
                pred_symcats = symcat_lut[torch.tensor(mode0_preds, device=device)].cpu().numpy()

                errors_mask = (mode0_preds != targets_class)
                total_errors = int(np.sum(errors_mask))

                if total_errors > 0:
                    same_shape = (pred_shapes == targets_shape)
                    same_colour = (pred_colours == targets_colour)
                    same_border = (pred_borders == targets_border)
                    same_symcat = (pred_symcats == targets_symcat)
                    same_concept_tuple = same_shape & same_colour & same_border & same_symcat

                    within_group_mask = errors_mask & same_concept_tuple
                    cross_group_mask = errors_mask & (~same_concept_tuple)

                    within_group_count = int(np.sum(within_group_mask))
                    cross_group_count = int(np.sum(cross_group_mask))

                    # Rescued by Mode 6
                    rescued_mask = errors_mask & (mode6_preds == targets_class)
                    rescued_total = int(np.sum(rescued_mask))
                    rescued_within = int(np.sum(rescued_mask & within_group_mask))
                    rescued_cross = int(np.sum(rescued_mask & cross_group_mask))
                else:
                    within_group_count = 0
                    cross_group_count = 0
                    rescued_total = 0
                    rescued_within = 0
                    rescued_cross = 0

                # Intervention Recovery Rate: (Acc_after - Acc_before) / (1 - Acc_before)
                acc_m0 = acc_by_mode["mode_0_acc"]
                acc_m6 = acc_by_mode["mode_6_acc"]
                ir_rate = float((acc_m6 - acc_m0) / (1.0 - acc_m0)) if acc_m0 < 1.0 else 0.0

                # Concept F1 & CDR calculations
                f1_sh = float(f1_score(targets_shape, np.array(corrupt_preds_shape), average="macro", zero_division=0))
                f1_col = float(f1_score(targets_colour, np.array(corrupt_preds_colour), average="macro", zero_division=0))
                f1_bor = float(f1_score(targets_border, np.array(corrupt_preds_border), average="macro", zero_division=0))
                f1_sym = float(f1_score(targets_symcat, np.array(corrupt_preds_symcat), average="macro", zero_division=0))

                cdr_sh = float((clean_f1_shape - f1_sh) / clean_f1_shape) if clean_f1_shape > 0 else 0.0
                cdr_col = float((clean_f1_colour - f1_col) / clean_f1_colour) if clean_f1_colour > 0 else 0.0
                cdr_bor = float((clean_f1_border - f1_bor) / clean_f1_border) if clean_f1_border > 0 else 0.0
                cdr_sym = float((clean_f1_symcat - f1_sym) / clean_f1_symcat) if clean_f1_symcat > 0 else 0.0

                rec = {
                    "corruption": c_name,
                    "severity": sev,
                    "adaptation": adapt_tag,
                    "divergence": float(d_val),
                    **acc_by_mode,
                    "intervention_recovery_rate": ir_rate,
                    "total_errors": total_errors,
                    "within_group_errors": within_group_count,
                    "cross_group_errors": cross_group_count,
                    "within_group_pct": float(within_group_count / total_errors) if total_errors > 0 else 0.0,
                    "cross_group_pct": float(cross_group_count / total_errors) if total_errors > 0 else 0.0,
                    "rescued_total": rescued_total,
                    "rescued_within": rescued_within,
                    "rescued_cross": rescued_cross,
                    "cdr": {
                        "shape": cdr_sh,
                        "colour": cdr_col,
                        "border": cdr_bor,
                        "symbol": cdr_sym,
                    },
                    "concept_f1": {
                        "shape": f1_sh,
                        "colour": f1_col,
                        "border": f1_bor,
                        "symbol": f1_sym,
                    }
                }
                intervention_results.append(rec)

            pbar.update(1)

    pbar.close()

    # Aggregate summaries across all evaluated conditions
    df_int = pd.DataFrame(intervention_results)
    summary_by_adapt = {}
    for adapt_tag in ["static_bn", "shift_gated_bn"]:
        sub_df = df_int[df_int["adaptation"] == adapt_tag]
        summary_by_adapt[adapt_tag] = {
            "mean_mode_0_acc": float(sub_df["mode_0_acc"].mean()),
            "mean_mode_1_acc": float(sub_df["mode_1_acc"].mean()),
            "mean_mode_2_acc": float(sub_df["mode_2_acc"].mean()),
            "mean_mode_3_acc": float(sub_df["mode_3_acc"].mean()),
            "mean_mode_4_acc": float(sub_df["mode_4_acc"].mean()),
            "mean_mode_5_acc": float(sub_df["mode_5_acc"].mean()),
            "mean_mode_6_acc": float(sub_df["mode_6_acc"].mean()),
            "mean_intervention_recovery_rate": float(sub_df["intervention_recovery_rate"].mean()),
            "mean_within_group_pct": float(sub_df["within_group_pct"].mean()),
            "mean_cross_group_pct": float(sub_df["cross_group_pct"].mean()),
            "mean_cdr_shape": float(np.mean([r["cdr"]["shape"] for _, r in sub_df.iterrows()])),
            "mean_cdr_colour": float(np.mean([r["cdr"]["colour"] for _, r in sub_df.iterrows()])),
            "mean_cdr_border": float(np.mean([r["cdr"]["border"] for _, r in sub_df.iterrows()])),
            "mean_cdr_symbol": float(np.mean([r["cdr"]["symbol"] for _, r in sub_df.iterrows()])),
        }

    final_report = {
        "checkpoint": str(checkpoint_path),
        "total_test_samples": total_samples,
        "clean_baseline": {
            "clean_accuracy": clean_acc_mode0,
            "concept_f1": {
                "shape": clean_f1_shape,
                "colour": clean_f1_colour,
                "border": clean_f1_border,
                "symbol": clean_f1_symcat,
            }
        },
        "summary": summary_by_adapt,
        "condition_details": intervention_results,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    print(f"\nSaved concept intervention benchmark to {out_file}")

    print("\n" + "=" * 80)
    print("CONCEPT INTERVENTION SUMMARY HIGHLIGHTS (Across Severity 3-5)")
    for adapt_tag in ["static_bn", "shift_gated_bn"]:
        s = summary_by_adapt[adapt_tag]
        print(f"\n--- {adapt_tag.upper()} ---")
        print(f"Mode 0 (Un-intervened)   : {s['mean_mode_0_acc']*100:.2f}%")
        print(f"Mode 1 (Shape Clamped)    : {s['mean_mode_1_acc']*100:.2f}% (diff {(s['mean_mode_1_acc'] - s['mean_mode_0_acc'])*100:+.2f}%)")
        print(f"Mode 2 (Colour Clamped)   : {s['mean_mode_2_acc']*100:.2f}% (diff {(s['mean_mode_2_acc'] - s['mean_mode_0_acc'])*100:+.2f}%)")
        print(f"Mode 3 (Border Clamped)   : {s['mean_mode_3_acc']*100:.2f}% (diff {(s['mean_mode_3_acc'] - s['mean_mode_0_acc'])*100:+.2f}%)")
        print(f"Mode 4 (Symbol Clamped)   : {s['mean_mode_4_acc']*100:.2f}% (diff {(s['mean_mode_4_acc'] - s['mean_mode_0_acc'])*100:+.2f}%)")
        print(f"Mode 5 (Macro V2I Prior)  : {s['mean_mode_5_acc']*100:.2f}% (diff {(s['mean_mode_5_acc'] - s['mean_mode_0_acc'])*100:+.2f}%)")
        print(f"Mode 6 (Full 4-Axis)      : {s['mean_mode_6_acc']*100:.2f}% (diff {(s['mean_mode_6_acc'] - s['mean_mode_0_acc'])*100:+.2f}%)")
        print(f"Intervention Recovery (IR): {s['mean_intervention_recovery_rate']*100:.2f}%")
        print(f"Within-Group Error Share  : {s['mean_within_group_pct']*100:.2f}% | Cross-Group: {s['mean_cross_group_pct']*100:.2f}%")
        print(f"Mean CDR Hierarchy        : Shape ({s['mean_cdr_shape']*100:.1f}%) < Colour ({s['mean_cdr_colour']*100:.1f}%) < Border ({s['mean_cdr_border']*100:.1f}%) < Symbol ({s['mean_cdr_symbol']*100:.1f}%)")
    print("=" * 80)

    return final_report


def main():
    parser = argparse.ArgumentParser(description="Evaluate Test-Time Concept Interventions on CBM.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_cbm_resnet18.pt")
    parser.add_argument("--meta-path", type=str, default="data/master_v1/metadata/test.csv", help="Path to test.csv")
    parser.add_argument("--data-root", type=str, default="german", help="Path to raw GTSRB folder")
    parser.add_argument("--output", type=str, default="reports/diagnostics/concept_intervention_benchmark.json", help="Output JSON path")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--tau-shift", type=float, default=0.10, help="Shift-gating threshold")
    parser.add_argument("--quick-limit", type=int, default=None, help="Sample limit for quick local test")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu' or 'cuda')")
    args = parser.parse_args()

    run_intervention_benchmark(
        checkpoint_path=args.checkpoint,
        meta_path=args.meta_path,
        data_root=args.data_root,
        output_path=args.output,
        batch_size=args.batch_size,
        tau_shift=args.tau_shift,
        quick_limit=args.quick_limit,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
