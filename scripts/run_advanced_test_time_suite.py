"""
Advanced Test-Time Robustness & OCS Diagnostic Suite
====================================================
Implements the 5 reviewer recommendations:
1. Breakdown of within-group errors by class (quantifies speed-limit digit concentration).
2. OCS as an Error Detector (AUROC of OCS vs Max-Softmax Probability for failure detection).
3. Uncertainty-Gated & CDR-Weighted Decoding (rerank only when MSP < tau, downweight fragile symbol head).
4. BN-Adaptation on test batches (restoring backbone features under covariate shift).
5. Combined BN-Adapt + Gated-OCS.

Zero training on noise. Zero extra parameters.
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
from sklearn.metrics import accuracy_score, roc_auc_score
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


def build_concept_lookup(concepts_csv: Path, device: torch.device):
    rows = load_class_concepts(str(concepts_csv))
    shape_lut, colour_lut, border_lut, symcat_lut = get_concept_lookup_tensors(device=device, csv_path=str(concepts_csv))
    class_names = {int(r["class_id"]): r["class_name"] for r in rows}
    tuple_sigs = {
        int(r["class_id"]): (r["shape"], r["colour_profile"], r["border_type"], r["symbol_category"])
        for r in rows
    }
    return (shape_lut, colour_lut, border_lut, symcat_lut), class_names, tuple_sigs


def set_bn_mode(model: nn.Module, adapt: bool):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if adapt:
                m.train()
                m.track_running_stats = False
            else:
                m.eval()
                m.track_running_stats = True


@torch.no_grad()
def run_full_suite(
    model: nn.Module,
    is_cbm: bool,
    test_df: pd.DataFrame,
    data_root: Path,
    device: torch.device,
    concept_luts: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    class_names: Dict[int, str],
    tuple_sigs: Dict[int, Tuple[str, str, str, str]],
    corruptions: List[str],
    severities: List[int],
    batch_size: int = 128,
    tau_uncertainty: float = 0.70,
    beta: float = 0.3,
    weights: Tuple[float, float, float, float] = (1.0, 0.8, 0.8, 0.2),  # CDR-informed weights
) -> Dict[str, Any]:
    w_shape, w_colour, w_border, w_symcat = weights
    shape_lut, colour_lut, border_lut, symcat_lut = concept_luts
    all_classes = torch.arange(43, device=device)
    c_s = shape_lut[all_classes]
    c_c = colour_lut[all_classes]
    c_b = border_lut[all_classes]
    c_m = symcat_lut[all_classes]

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

    condition_results = []
    all_msp_scores = []
    all_ocs_scores = []
    all_correct_indicators = []
    within_error_class_counts: Dict[int, int] = {c: 0 for c in range(43)}

    for c_name in corruptions:
        c_func = CORRUPTIONS_REGISTRY[c_name]
        for sev in severities:
            # Prepare corrupted batch tensors
            batch_tensors = []
            for start in range(0, len(clean_items), batch_size):
                items = clean_items[start : start + batch_size]
                tensors = []
                for canvas, _ in items:
                    corrupted = c_func(canvas, sev)
                    t = TF.normalize(TF.to_tensor(corrupted), mean=IMAGENET_MEAN, std=IMAGENET_STD)
                    tensors.append(t)
                batch_tensors.append(torch.stack(tensors).to(device))

            # -------------------------------------------------------------
            # PASS 1: Static Model Inference (Raw & Gated-OCS)
            # -------------------------------------------------------------
            set_bn_mode(model, adapt=False)
            preds_raw = []
            preds_gated_ocs = []
            batch_msp = []
            batch_ocs_rel = []

            for b_x in batch_tensors:
                out = model(b_x)
                logits = out["logits"] if isinstance(out, dict) else out
                probs = F.softmax(logits, dim=-1)
                log_p_y = F.log_softmax(logits, dim=-1)

                max_p, raw_pred = torch.max(probs, dim=-1)
                preds_raw.extend(raw_pred.cpu().numpy())
                batch_msp.extend(max_p.cpu().numpy())

                if is_cbm:
                    log_q_s = F.log_softmax(out["logits_shape"], dim=-1)
                    log_q_c = F.log_softmax(out["logits_colour"], dim=-1)
                    log_q_b = F.log_softmax(out["logits_border"], dim=-1)
                    log_q_m = F.log_softmax(out["logits_symcat"], dim=-1)

                    # Weighted concept prior (Shape & Border weighted high, Symbol weighted low)
                    weighted_c_prior = (
                        w_shape * log_q_s[:, c_s]
                        + w_colour * log_q_c[:, c_c]
                        + w_border * log_q_b[:, c_b]
                        + w_symcat * log_q_m[:, c_m]
                    )

                    # Uncertainty-Gated Reranking: Only rerank when max_p < tau_uncertainty!
                    score_rerank = log_p_y + beta * weighted_c_prior
                    rerank_pred = torch.argmax(score_rerank, dim=-1)

                    gate_mask = max_p < tau_uncertainty
                    final_pred = torch.where(gate_mask, rerank_pred, raw_pred)
                    preds_gated_ocs.extend(final_pred.cpu().numpy())

                    # OCS Reliability Score: expected consistency E_{p(y)} [log q(c(y))]
                    # Measures whether predicted class distribution aligns with concept evidence
                    sample_ocs = torch.sum(probs * weighted_c_prior, dim=-1)
                    batch_ocs_rel.extend(sample_ocs.cpu().numpy())
                else:
                    preds_gated_ocs.extend(raw_pred.cpu().numpy())

            preds_raw_arr = np.array(preds_raw)
            preds_gated_arr = np.array(preds_gated_ocs)
            acc_raw = accuracy_score(targets_arr, preds_raw_arr)
            acc_gated_ocs = accuracy_score(targets_arr, preds_gated_arr)

            # Record error breakdown & within-group class frequency
            err_mask = (preds_raw_arr != targets_arr)
            for idx in np.where(err_mask)[0]:
                yt = targets_arr[idx]
                yp = preds_raw_arr[idx]
                if tuple_sigs[yt] == tuple_sigs[yp]:
                    within_error_class_counts[yt] += 1

            # Accumulate reliability detection metrics
            correct_mask = (preds_raw_arr == targets_arr).astype(int)
            all_correct_indicators.extend(correct_mask)
            all_msp_scores.extend(batch_msp)
            if is_cbm:
                all_ocs_scores.extend(batch_ocs_rel)

            # -------------------------------------------------------------
            # PASS 2: BN-Adaptation Inference (Test-Time Covariate Shift Fix)
            # -------------------------------------------------------------
            set_bn_mode(model, adapt=True)
            preds_bn = []
            preds_bn_gated_ocs = []

            for b_x in batch_tensors:
                out = model(b_x)
                logits = out["logits"] if isinstance(out, dict) else out
                probs = F.softmax(logits, dim=-1)
                log_p_y = F.log_softmax(logits, dim=-1)

                max_p, bn_pred = torch.max(probs, dim=-1)
                preds_bn.extend(bn_pred.cpu().numpy())

                if is_cbm:
                    log_q_s = F.log_softmax(out["logits_shape"], dim=-1)
                    log_q_c = F.log_softmax(out["logits_colour"], dim=-1)
                    log_q_b = F.log_softmax(out["logits_border"], dim=-1)
                    log_q_m = F.log_softmax(out["logits_symcat"], dim=-1)

                    weighted_c_prior = (
                        w_shape * log_q_s[:, c_s]
                        + w_colour * log_q_c[:, c_c]
                        + w_border * log_q_b[:, c_b]
                        + w_symcat * log_q_m[:, c_m]
                    )
                    score_rerank = log_p_y + beta * weighted_c_prior
                    rerank_pred = torch.argmax(score_rerank, dim=-1)

                    gate_mask = max_p < tau_uncertainty
                    final_bn_pred = torch.where(gate_mask, rerank_pred, bn_pred)
                    preds_bn_gated_ocs.extend(final_bn_pred.cpu().numpy())
                else:
                    preds_bn_gated_ocs.extend(bn_pred.cpu().numpy())

            acc_bn = accuracy_score(targets_arr, np.array(preds_bn))
            acc_bn_gated = accuracy_score(targets_arr, np.array(preds_bn_gated_ocs))

            row_info = {
                "corruption": c_name,
                "severity": sev,
                "acc_raw": float(acc_raw),
                "acc_gated_ocs": float(acc_gated_ocs),
                "acc_bn_adapt": float(acc_bn),
                "acc_bn_gated_ocs": float(acc_bn_gated),
                "gain_gated_ocs": float(acc_gated_ocs - acc_raw),
                "gain_bn_adapt": float(acc_bn - acc_raw),
                "gain_total": float(acc_bn_gated - acc_raw),
            }
            condition_results.append(row_info)

            print(
                f"[{c_name:16s} S{sev}] Raw: {acc_raw*100:5.2f}% | "
                f"Gated-OCS: {acc_gated_ocs*100:5.2f}% ({((acc_gated_ocs-acc_raw)*100):+5.2f}%) | "
                f"BN-Adapt: {acc_bn*100:5.2f}% ({((acc_bn-acc_raw)*100):+5.2f}%) | "
                f"BN+Gated-OCS: {acc_bn_gated*100:5.2f}% ({((acc_bn_gated-acc_raw)*100):+5.2f}%)"
            )

    # -------------------------------------------------------------
    # AUROC for Correctness / Error Detection
    # -------------------------------------------------------------
    y_true_corr = np.array(all_correct_indicators)
    y_msp = np.array(all_msp_scores)
    auroc_msp = float(roc_auc_score(y_true_corr, y_msp))

    auroc_ocs = 0.0
    auroc_hybrid = 0.0
    if is_cbm and len(all_ocs_scores) == len(y_true_corr):
        y_ocs = np.array(all_ocs_scores)
        # Normalize OCS for hybrid fusion
        y_ocs_norm = (y_ocs - np.mean(y_ocs)) / (np.std(y_ocs) + 1e-7)
        y_msp_norm = (y_msp - np.mean(y_msp)) / (np.std(y_msp) + 1e-7)
        y_hybrid = y_msp_norm + 0.5 * y_ocs_norm

        auroc_ocs = float(roc_auc_score(y_true_corr, y_ocs))
        auroc_hybrid = float(roc_auc_score(y_true_corr, y_hybrid))

    # Top within-group error classes
    top_within_classes = sorted(within_error_class_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    top_within_list = [
        {"class_id": cid, "class_name": class_names.get(cid, f"Class {cid}"), "count": cnt}
        for cid, cnt in top_within_classes
    ]

    df_summary = pd.DataFrame(condition_results)
    mean_raw = float(df_summary["acc_raw"].mean())
    mean_gated = float(df_summary["acc_gated_ocs"].mean())
    mean_bn = float(df_summary["acc_bn_adapt"].mean())
    mean_bn_gated = float(df_summary["acc_bn_gated_ocs"].mean())

    suite_summary = {
        "mean_raw_acc": mean_raw,
        "mean_gated_ocs_acc": mean_gated,
        "mean_bn_adapt_acc": mean_bn,
        "mean_bn_gated_ocs_acc": mean_bn_gated,
        "mean_gain_bn_adapt": mean_bn - mean_raw,
        "mean_gain_bn_gated_ocs": mean_bn_gated - mean_raw,
        "auroc_error_detection": {
            "auroc_msp": auroc_msp,
            "auroc_ocs": auroc_ocs,
            "auroc_hybrid_msp_plus_ocs": auroc_hybrid,
        },
        "top_within_group_error_classes": top_within_list,
        "per_condition": condition_results,
    }
    return suite_summary


def main():
    parser = argparse.ArgumentParser(description="Advanced Test-Time Robustness Suite.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--data-root", type=str, default="german")
    parser.add_argument("--meta-path", type=str, default="data/master_v1/metadata/test.csv")
    parser.add_argument("--concepts-csv", type=str, default="data/master_v1/concepts/class_concepts.csv")
    parser.add_argument("--output-json", type=str, default="reports/diagnostics/advanced_test_time_suite.json")
    parser.add_argument(
        "--corruptions",
        type=str,
        default="gaussian_noise,shot_noise,impulse_noise",
        help="Comma-separated corruptions or 'all'",
    )
    parser.add_argument(
        "--severities",
        type=str,
        default="3,4,5",
        help="Comma-separated severities (e.g. '3,4,5' or '1,2,3,4,5')",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--tau-uncertainty", type=float, default=0.70, help="MSP gating threshold (default: 0.70)")
    parser.add_argument("--beta", type=float, default=0.3, help="Weight for concept log prior in OCS (default: 0.3)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 85)
    print("ADVANCED TEST-TIME ROBUSTNESS & ONTOLOGY RELIABILITY SUITE")
    print(f"Checkpoint       : {args.checkpoint}")
    print(f"Device           : {device}")
    print(f"Uncertainty Tau  : {args.tau_uncertainty}")
    print(f"Decoding Beta    : {args.beta}")
    print("=" * 85)

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

    concept_luts, class_names, tuple_sigs = build_concept_lookup(concepts_csv, device)
    test_df = pd.read_csv(meta_path)

    if args.corruptions == "all":
        corruptions = list(CORRUPTIONS_REGISTRY.keys())
    else:
        corruptions = [c.strip() for c in args.corruptions.split(",")]

    severities = [int(s.strip()) for s in args.severities.split(",")]

    summary = run_full_suite(
        model=model,
        is_cbm=is_cbm,
        test_df=test_df,
        data_root=data_root,
        device=device,
        concept_luts=concept_luts,
        class_names=class_names,
        tuple_sigs=tuple_sigs,
        corruptions=corruptions,
        severities=severities,
        batch_size=args.batch_size,
        tau_uncertainty=args.tau_uncertainty,
        beta=args.beta,
    )

    out_p = Path(args.output_json)
    if not out_p.is_absolute():
        out_p = PROJECT_ROOT / out_p
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 85)
    print("FINAL SUITE SUMMARY:")
    print(f"Mean Raw Accuracy       : {summary['mean_raw_acc']*100:.2f}%")
    print(f"Mean Gated-OCS Accuracy : {summary['mean_gated_ocs_acc']*100:.2f}% (Δ {((summary['mean_gated_ocs_acc']-summary['mean_raw_acc'])*100):+.2f}%)")
    print(f"Mean BN-Adapt Accuracy  : {summary['mean_bn_adapt_acc']*100:.2f}% (Δ {summary['mean_gain_bn_adapt']*100:+.2f}%)")
    print(f"Mean BN+Gated-OCS Acc   : {summary['mean_bn_gated_ocs_acc']*100:.2f}% (Δ {summary['mean_gain_bn_gated_ocs']*100:+.2f}%)")
    print("-" * 85)
    print("ERROR DETECTION AUROC (Correct vs Misclassified):")
    auroc = summary["auroc_error_detection"]
    print(f"  Max-Softmax (MSP) AUROC  : {auroc['auroc_msp']:.4f}")
    if is_cbm:
        print(f"  Concept Consistency (OCS): {auroc['auroc_ocs']:.4f}")
        print(f"  Hybrid (MSP + OCS) AUROC : {auroc['auroc_hybrid_msp_plus_ocs']:.4f}")
    print("-" * 85)
    print("TOP 5 WITHIN-GROUP ERROR CLASSES (Confusions inside same concept group):")
    for item in summary["top_within_group_error_classes"][:5]:
        print(f"  Class {item['class_id']:2d} ({item['class_name']:30s}): {item['count']:,} errors")
    print(f"\nReport saved to: {out_p}")
    print("=" * 85)


if __name__ == "__main__":
    main()
