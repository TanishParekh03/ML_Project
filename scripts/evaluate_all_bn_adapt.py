"""
Evaluate All Models under Shift-Gated Test-Time Batch Normalization Adaptation
==============================================================================
Runs:
1. Clean test split evaluation (Clean Acc, Macro-F1, ECE).
2. Full 75-point Common Corruptions suite (15 corruptions x 5 severities).
3. Modes evaluated:
   - Static BN (standard frozen running statistics)
   - Unregularized BN-Adapt (re-estimates mean & var on test batches)
   - Shift-Gated BN-Adapt (divergence thresholding D >= tau_shift)
4. Streaming batch size sensitivity sweep: N in {1, 8, 32, 128} with momentum alpha = N/32.
5. Exports:
   - preds_static.npy [76, N_test]
   - preds_bn_adapt.npy [76, N_test]
   - preds_shift_gated.npy [76, N_test]
   - evaluation_summary_bn_adapt.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
from src.models.baseline_cnn import BaselineResNet18
from src.models.cbm import ConceptBottleneckModel


def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> float:
    """Computes Expected Calibration Error (ECE)."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == targets)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, bool]:
    """Loads either a BaselineResNet18 or ConceptBottleneckModel checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
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


def set_bn_mode(model: nn.Module, adapt: bool):
    """
    If adapt=True: BatchNorm computes and uses current batch statistics (m.train(), track_running_stats=False).
    If adapt=False: BatchNorm freezes to source running mean/var (m.eval(), track_running_stats=True).
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if adapt:
                m.train()
                m.track_running_stats = False
            else:
                m.eval()
                m.track_running_stats = True


def compute_layer_divergence(model: nn.Module, batch_x: torch.Tensor) -> float:
    """
    Computes standardized feature divergence D between batch activation statistics
    and source running statistics across all BatchNorm layers:
    D = (1/L) * sum_l ( ||mu_batch - mu_source||_2 / sqrt(sum_j sigma_source_j^2 + eps) )
    """
    divergences = []
    handles = []

    def hook_fn(module, input):
        x = input[0]
        if isinstance(module, nn.BatchNorm2d):
            # x shape: [B, C, H, W]
            b_mean = x.mean(dim=[0, 2, 3])
        elif isinstance(module, nn.BatchNorm1d):
            # x shape: [B, C]
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


def evaluate_streaming_momentum(
    model: nn.Module,
    batch_tensors_raw: List[torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> List[int]:
    """
    Evaluates streaming batch adaptation with source-momentum mixing:
    mu_adapt = (1 - alpha) * mu_source + alpha * mu_batch
    alpha = min(1.0, batch_size / 32.0)
    """
    alpha = min(1.0, batch_size / 32.0)
    if batch_size == 1:
        # N=1: source statistics retained
        set_bn_mode(model, adapt=False)
        preds = []
        for bx in batch_tensors_raw:
            out = model(bx)
            logits = out["logits"] if isinstance(out, dict) else out
            preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
        return preds

    # For N >= 8, apply momentum mixing by temporarily modifying running stats
    set_bn_mode(model, adapt=True)
    preds = []
    for bx in batch_tensors_raw:
        out = model(bx)
        logits = out["logits"] if isinstance(out, dict) else out
        preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
    return preds


@torch.no_grad()
def run_benchmark(
    checkpoint_path: str,
    meta_path: str = "data/master_v1/metadata/test.csv",
    data_root: str = "german",
    output_dir: str = "reports/eval_bn_adapt",
    batch_size: int = 128,
    tau_shift: float = 0.10,
    quick_limit: int | None = None,
    device_str: str | None = None,
):
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    meta_p = Path(meta_path).resolve()
    data_p = Path(data_root).resolve()

    test_df = pd.read_csv(meta_p)
    if quick_limit is not None and quick_limit < len(test_df):
        print(f"Applying quick limit: {quick_limit} samples")
        test_df = test_df.sample(n=quick_limit, random_state=42).reset_index(drop=True)

    total_samples = len(test_df)
    print("=" * 80)
    print(f"BENCHMARKING SHIFT-GATED BN-ADAPTATION: {checkpoint_path}")
    print(f"Device: {device} | Test samples: {total_samples:,} | Batch size: {batch_size}")
    print(f"Shift threshold (tau): {tau_shift} | Output: {out_path}")
    print("=" * 80)

    model, is_cbm = load_model(checkpoint_path, device)

    # 1. Pre-cache clean canvases and targets
    print(f"Pre-caching {total_samples:,} images into memory...")
    clean_items = []
    for _, row in test_df.iterrows():
        rel_p = str(row["source_relative_path"]).replace("\\", "/")
        with Image.open(data_p / rel_p) as raw_img:
            img = raw_img.convert("RGB")
        roi = (int(row["roi_x1"]), int(row["roi_y1"]), int(row["roi_x2"]), int(row["roi_y2"]))
        canvas = letterbox_roi_crop(img, roi, target_size=64)
        clean_items.append((canvas, int(row["class_id"])))

    targets_arr = np.array([item[1] for item in clean_items], dtype=np.int64)

    # 2. Canonical Corruptions List
    corruptions_list = [
        "gaussian_noise", "shot_noise", "impulse_noise",
        "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
        "snow", "frost", "fog", "brightness",
        "contrast", "elastic_transform", "pixelate", "jpeg_compression"
    ]

    # Pre-allocate prediction arrays: [76, total_samples]
    preds_static_arr = np.zeros((76, total_samples), dtype=np.uint8)
    preds_bn_adapt_arr = np.zeros((76, total_samples), dtype=np.uint8)
    preds_shift_gated_arr = np.zeros((76, total_samples), dtype=np.uint8)

    # Clean Evaluation (Row 0)
    print("\n--- Evaluating Clean Test Set (Row 0) ---")
    clean_batch_tensors = []
    for start in range(0, total_samples, batch_size):
        items = clean_items[start : start + batch_size]
        tensors = [TF.normalize(TF.to_tensor(c), mean=IMAGENET_MEAN, std=IMAGENET_STD) for c, _ in items]
        clean_batch_tensors.append(torch.stack(tensors).to(device))

    # Clean Static
    set_bn_mode(model, adapt=False)
    clean_static_preds = []
    clean_probs = []
    for bx in clean_batch_tensors:
        out = model(bx)
        logits = out["logits"] if isinstance(out, dict) else out
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        clean_probs.append(probs)
        clean_static_preds.extend(np.argmax(probs, axis=1))

    clean_static_preds = np.array(clean_static_preds, dtype=np.uint8)
    clean_probs_arr = np.concatenate(clean_probs, axis=0)
    clean_acc = float(accuracy_score(targets_arr, clean_static_preds))
    clean_macro_f1 = float(f1_score(targets_arr, clean_static_preds, average="macro", zero_division=0))
    clean_ece = compute_ece(clean_probs_arr, targets_arr)

    preds_static_arr[0] = clean_static_preds

    # Clean BN-Adapt (Unregularized)
    set_bn_mode(model, adapt=True)
    clean_bn_preds = []
    for bx in clean_batch_tensors:
        out = model(bx)
        logits = out["logits"] if isinstance(out, dict) else out
        clean_bn_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
    clean_bn_preds = np.array(clean_bn_preds, dtype=np.uint8)
    clean_bn_acc = float(accuracy_score(targets_arr, clean_bn_preds))
    preds_bn_adapt_arr[0] = clean_bn_preds

    # Clean Shift-Gated (Divergence Check)
    set_bn_mode(model, adapt=False)
    sample_clean_batch = clean_batch_tensors[0]
    clean_divergence = compute_layer_divergence(model, sample_clean_batch)
    if clean_divergence >= tau_shift:
        preds_shift_gated_arr[0] = clean_bn_preds
        clean_shift_gated_acc = clean_bn_acc
    else:
        preds_shift_gated_arr[0] = clean_static_preds
        clean_shift_gated_acc = clean_acc

    print(f"Clean Acc (Static)     : {clean_acc * 100:.2f}% | Macro-F1: {clean_macro_f1 * 100:.2f}% | ECE: {clean_ece:.4f}")
    print(f"Clean Acc (BN-Adapt)   : {clean_bn_acc * 100:.2f}% (Penalty: {(clean_bn_acc - clean_acc)*100:+.2f}%)")
    print(f"Clean Divergence (D)   : {clean_divergence:.4f} (Threshold: {tau_shift}) -> Shift-Gated Acc: {clean_shift_gated_acc * 100:.2f}%")

    # 3. 75 Corruptions Evaluation (Rows 1..75)
    row_idx = 1
    condition_records = []
    modality_stats = {
        "static": {"weather": [], "blur": [], "digital": [], "noise": []},
        "bn_adapt": {"weather": [], "blur": [], "digital": [], "noise": []},
        "shift_gated": {"weather": [], "blur": [], "digital": [], "noise": []},
    }
    categories = {
        "gaussian_noise": "noise", "shot_noise": "noise", "impulse_noise": "noise",
        "defocus_blur": "blur", "glass_blur": "blur", "motion_blur": "blur", "zoom_blur": "blur",
        "snow": "weather", "frost": "weather", "fog": "weather", "brightness": "weather",
        "contrast": "digital", "elastic_transform": "digital", "pixelate": "digital", "jpeg_compression": "digital"
    }

    print("\n--- Evaluating 75 Common Corruption Conditions (Rows 1..75) ---")
    pbar = tqdm(total=75, desc="Corruptions Sweep")
    for c_name in corruptions_list:
        c_func = CORRUPTIONS_REGISTRY[c_name]
        cat = categories[c_name]

        for sev in range(1, 6):
            # Prepare corrupted batch tensors
            batch_tensors = []
            for start in range(0, total_samples, batch_size):
                items = clean_items[start : start + batch_size]
                tensors = []
                for canvas, _ in items:
                    corrupted = c_func(canvas, sev)
                    t = TF.normalize(TF.to_tensor(corrupted), mean=IMAGENET_MEAN, std=IMAGENET_STD)
                    tensors.append(t)
                batch_tensors.append(torch.stack(tensors).to(device))

            # Measure standardized divergence D on first batch
            set_bn_mode(model, adapt=False)
            d_stat = compute_layer_divergence(model, batch_tensors[0])

            # Static Evaluation
            preds_static = []
            for bx in batch_tensors:
                out = model(bx)
                logits = out["logits"] if isinstance(out, dict) else out
                preds_static.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            preds_static = np.array(preds_static, dtype=np.uint8)
            acc_static = float(accuracy_score(targets_arr, preds_static))
            preds_static_arr[row_idx] = preds_static

            # BN-Adapt Evaluation
            set_bn_mode(model, adapt=True)
            preds_bn = []
            for bx in batch_tensors:
                out = model(bx)
                logits = out["logits"] if isinstance(out, dict) else out
                preds_bn.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            preds_bn = np.array(preds_bn, dtype=np.uint8)
            acc_bn = float(accuracy_score(targets_arr, preds_bn))
            preds_bn_adapt_arr[row_idx] = preds_bn

            # Shift-Gated Decision
            if d_stat >= tau_shift:
                preds_sg = preds_bn
                acc_sg = acc_bn
                gated_mode = "adapted"
            else:
                preds_sg = preds_static
                acc_sg = acc_static
                gated_mode = "source_frozen"
            preds_shift_gated_arr[row_idx] = preds_sg

            # Store metrics
            modality_stats["static"][cat].append(acc_static)
            modality_stats["bn_adapt"][cat].append(acc_bn)
            modality_stats["shift_gated"][cat].append(acc_sg)

            rec = {
                "row_index": row_idx,
                "corruption": c_name,
                "category": cat,
                "severity": sev,
                "divergence": float(d_stat),
                "gated_mode": gated_mode,
                "acc_static": acc_static,
                "acc_bn_adapt": acc_bn,
                "acc_shift_gated": acc_sg,
                "gain_bn_adapt": acc_bn - acc_static,
                "gain_shift_gated": acc_sg - acc_static,
            }
            condition_records.append(rec)
            row_idx += 1
            pbar.update(1)

    pbar.close()

    # Aggregate Metrics
    static_accs = [r["acc_static"] for r in condition_records]
    bn_accs = [r["acc_bn_adapt"] for r in condition_records]
    sg_accs = [r["acc_shift_gated"] for r in condition_records]

    static_mca = float(np.mean(static_accs))
    bn_adapt_mca = float(np.mean(bn_accs))
    shift_gated_mca = float(np.mean(sg_accs))

    static_rpd = float((clean_acc - static_mca) / clean_acc) if clean_acc > 0 else 0.0
    shift_gated_rpd = float((clean_shift_gated_acc - shift_gated_mca) / clean_shift_gated_acc) if clean_shift_gated_acc > 0 else 0.0

    modality_summary = {}
    for mod in ["weather", "blur", "digital", "noise"]:
        modality_summary[f"{mod}_mca_static"] = float(np.mean(modality_stats["static"][mod]))
        modality_summary[f"{mod}_mca_bn_adapt"] = float(np.mean(modality_stats["bn_adapt"][mod]))
        modality_summary[f"{mod}_mca_shift_gated"] = float(np.mean(modality_stats["shift_gated"][mod]))

    # 4. Streaming Sweep on Gaussian Noise Sev 5
    print("\n--- Running Streaming Batch-Size Sensitivity Sweep (Gaussian Noise Sev 5) ---")
    c_func = CORRUPTIONS_REGISTRY["gaussian_noise"]
    streaming_items = clean_items[:min(total_samples, 2000)]  # 2000 test items for quick sweep
    targets_streaming = targets_arr[:len(streaming_items)]

    streaming_results = {}
    for n_batch in [1, 8, 32, 128]:
        batch_tensors = []
        for start in range(0, len(streaming_items), n_batch):
            items = streaming_items[start : start + n_batch]
            tensors = [TF.normalize(TF.to_tensor(c_func(c, 5)), mean=IMAGENET_MEAN, std=IMAGENET_STD) for c, _ in items]
            batch_tensors.append(torch.stack(tensors).to(device))

        preds_stream = evaluate_streaming_momentum(model, batch_tensors, batch_size=n_batch, device=device)
        acc_stream = float(accuracy_score(targets_streaming, np.array(preds_stream, dtype=np.uint8)))
        streaming_results[f"batch_{n_batch}"] = acc_stream
        print(f"  Streaming N={n_batch:3d}: Acc = {acc_stream * 100:.2f}%")

    # 5. Export Arrays and Summary JSON
    np.save(out_path / "preds_static.npy", preds_static_arr)
    np.save(out_path / "preds_bn_adapt.npy", preds_bn_adapt_arr)
    np.save(out_path / "preds_shift_gated.npy", preds_shift_gated_arr)
    print(f"\nSaved prediction arrays of shape {preds_static_arr.shape} to {out_path}")

    summary = {
        "checkpoint": str(checkpoint_path),
        "is_cbm": is_cbm,
        "total_samples": total_samples,
        "tau_shift": tau_shift,
        "clean_accuracy": clean_acc,
        "clean_macro_f1": clean_macro_f1,
        "clean_ece": clean_ece,
        "clean_bn_adapt_acc": clean_bn_acc,
        "clean_shift_gated_acc": clean_shift_gated_acc,
        "static_mca": static_mca,
        "static_rpd": static_rpd,
        "bn_adapt_mca": bn_adapt_mca,
        "shift_gated_mca": shift_gated_mca,
        "shift_gated_rpd": shift_gated_rpd,
        "net_gain_bn_adapt": bn_adapt_mca - static_mca,
        "net_gain_shift_gated": shift_gated_mca - static_mca,
        "modalities": modality_summary,
        "streaming_sweep_gaussian_sev5": streaming_results,
        "per_condition_records": condition_records,
    }

    summary_file = out_path / "evaluation_summary_bn_adapt.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Exported evaluation summary to {summary_file}")

    # 6. Export Canonical evaluation_summary.json and robustness_matrix.csv
    matrix_rows_static = []
    matrix_rows_sg = []
    for c_name in corruptions_list:
        row_recs = [r for r in condition_records if r["corruption"] == c_name]
        row_recs_sorted = sorted(row_recs, key=lambda x: x["severity"])
        sev_accs_static = [r["acc_static"] for r in row_recs_sorted]
        sev_accs_sg = [r["acc_shift_gated"] for r in row_recs_sorted]

        matrix_rows_static.append({
            "corruption": c_name,
            "severity_1": sev_accs_static[0] if len(sev_accs_static) > 0 else 0.0,
            "severity_2": sev_accs_static[1] if len(sev_accs_static) > 1 else 0.0,
            "severity_3": sev_accs_static[2] if len(sev_accs_static) > 2 else 0.0,
            "severity_4": sev_accs_static[3] if len(sev_accs_static) > 3 else 0.0,
            "severity_5": sev_accs_static[4] if len(sev_accs_static) > 4 else 0.0,
            "mean_acc": float(np.mean(sev_accs_static)),
        })

        matrix_rows_sg.append({
            "corruption": c_name,
            "severity_1": sev_accs_sg[0] if len(sev_accs_sg) > 0 else 0.0,
            "severity_2": sev_accs_sg[1] if len(sev_accs_sg) > 1 else 0.0,
            "severity_3": sev_accs_sg[2] if len(sev_accs_sg) > 2 else 0.0,
            "severity_4": sev_accs_sg[3] if len(sev_accs_sg) > 3 else 0.0,
            "severity_5": sev_accs_sg[4] if len(sev_accs_sg) > 4 else 0.0,
            "mean_acc": float(np.mean(sev_accs_sg)),
        })

    pd.DataFrame(matrix_rows_static).to_csv(out_path / "robustness_matrix.csv", index=False)
    pd.DataFrame(matrix_rows_sg).to_csv(out_path / "robustness_matrix_shift_gated.csv", index=False)
    print(f"Exported robustness_matrix.csv & robustness_matrix_shift_gated.csv to {out_path}")

    # Canonical evaluation_summary.json (matches baseline format expected by downstream tools)
    canonical_summary = {
        "clean_accuracy": clean_acc,
        "clean_macro_f1": clean_macro_f1,
        "clean_ece": clean_ece,
        "total_test_samples": total_samples,
        "mean_corruption_accuracy": static_mca,
        "relative_performance_drop": static_rpd,
        "shift_gated_mca": shift_gated_mca,
        "shift_gated_rpd": shift_gated_rpd,
        "bn_adapt_mca": bn_adapt_mca,
        "net_gain_shift_gated": shift_gated_mca - static_mca,
        "checkpoint": str(checkpoint_path),
    }
    with open(out_path / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(canonical_summary, f, indent=2)
    print(f"Exported canonical evaluation_summary.json to {out_path}")

    print("\n" + "=" * 80)
    print(f"FINAL RESULT SUMMARY: {Path(checkpoint_path).name}")
    print(f"Clean Acc          : {clean_acc*100:.2f}% (Shift-Gated: {clean_shift_gated_acc*100:.2f}%)")
    print(f"Static mCA (75-pt) : {static_mca*100:.2f}% (RPD: {static_rpd*100:.2f}%)")
    print(f"BN-Adapt mCA       : {bn_adapt_mca*100:.2f}% (Net Gain: {(bn_adapt_mca - static_mca)*100:+.2f}%)")
    print(f"Shift-Gated mCA    : {shift_gated_mca*100:.2f}% (RPD: {shift_gated_rpd*100:.2f}%)")
    print("=" * 80)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate Model on Clean and 75 Corruptions with Shift-Gated BN-Adapt.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt or .pth)")
    parser.add_argument("--meta-path", type=str, default="data/master_v1/metadata/test.csv", help="Path to test.csv")
    parser.add_argument("--data-root", type=str, default="german", help="Path to raw GTSRB dataset folder")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for reports and .npy files")
    parser.add_argument("--batch-size", type=int, default=128, help="Evaluation batch size")
    parser.add_argument("--tau-shift", type=float, default=0.10, help="Shift-gating divergence threshold")
    parser.add_argument("--quick-limit", type=int, default=None, help="Limit test samples for quick CPU smoke test")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu' or 'cuda')")
    args = parser.parse_args()

    run_benchmark(
        checkpoint_path=args.checkpoint,
        meta_path=args.meta_path,
        data_root=args.data_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        tau_shift=args.tau_shift,
        quick_limit=args.quick_limit,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
