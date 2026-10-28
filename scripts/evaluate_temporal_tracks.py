"""
Temporal Video-Track Inference & Concept Smoothing Across 30-Frame Sequences
============================================================================
Evaluates:
1. Single-Frame Static Inference (frame-by-frame)
2. Temporal Concept & Feature EMA Smoothing:
     c_bar_t = gamma * c_bar_{t-1} + (1 - gamma) * c_hat_t
     h_bar_t = gamma * h_bar_{t-1} + (1 - gamma) * h_t
   for gamma in {0.5, 0.7, 0.85}
3. Online Streaming BN-Adapt (Batch N=8, momentum alpha=N/32) + Temporal EMA

Evaluates on:
data/master_v1/metadata/val.csv (261 tracks, 30 frames per track, 7,830 images total).
"""

from __future__ import annotations

import argparse
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


@torch.no_grad()
def evaluate_temporal(
    checkpoint_path: str,
    val_meta_path: str = "data/master_v1/metadata/val.csv",
    data_root: str = "german",
    output_path: str = "reports/diagnostics/temporal_track_smoothing.json",
    limit_tracks: int | None = None,
    eval_corruptions: List[str] | None = None,
    severities: List[int] | None = None,
    gammas: List[float] = [0.5, 0.7, 0.85],
    device_str: str | None = None,
) -> Dict[str, Any]:
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    val_df = pd.read_csv(val_meta_path)
    data_p = Path(data_root).resolve()

    if eval_corruptions is None:
        eval_corruptions = ["gaussian_noise", "shot_noise", "impulse_noise", "fog", "defocus_blur"]
    if severities is None:
        severities = [3, 4, 5]

    # Group into tracks sorted chronologically
    track_groups = []
    for group_id, grp in val_df.groupby("group_id"):
        grp_sorted = grp.sort_values("image_id").reset_index(drop=True)
        track_groups.append((group_id, grp_sorted))

    if limit_tracks is not None and limit_tracks < len(track_groups):
        print(f"Limiting to first {limit_tracks} tracks (out of {len(track_groups)})")
        track_groups = track_groups[:limit_tracks]

    print("=" * 80)
    print("TEMPORAL VIDEO-TRACK CONCEPT SMOOTHING BENCHMARK")
    print(f"Checkpoint    : {checkpoint_path}")
    print(f"Device        : {device} | Total Tracks: {len(track_groups)} (30 frames/track)")
    print(f"Corruptions   : {eval_corruptions}")
    print(f"Severities    : {severities}")
    print(f"EMA gammas    : {gammas}")
    print(f"Output        : {out_file}")
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

    # 3. Pre-load track images
    print(f"Pre-caching {len(track_groups)} tracks into memory...")
    cached_tracks = []
    for gid, grp in track_groups:
        frames_items = []
        c_id = int(grp.iloc[0]["class_id"])
        for _, row in grp.iterrows():
            rel_p = str(row["source_relative_path"]).replace("\\", "/")
            with Image.open(data_p / rel_p) as raw_img:
                img = raw_img.convert("RGB")
            roi = (int(row["roi_x1"]), int(row["roi_y1"]), int(row["roi_x2"]), int(row["roi_y2"]))
            canvas = letterbox_roi_crop(img, roi, target_size=64)
            frames_items.append(canvas)
        cached_tracks.append((gid, c_id, frames_items))

    # Evaluate across specified corruptions and severities
    results_records = []
    total_eval_points = len(eval_corruptions) * len(severities)
    pbar = tqdm(total=total_eval_points, desc="Temporal Sweep")

    for c_name in eval_corruptions:
        c_func = CORRUPTIONS_REGISTRY[c_name]
        for sev in severities:
            # Metrics accumulators
            acc_single_frames = []
            acc_track_terminal = []  # accuracy at frame 30
            acc_track_majority = []  # majority vote across 30 frames
            acc_track_mean_logits = []  # argmax of mean logits

            ema_frame_accs = {g: [] for g in gammas}
            ema_track_terminal = {g: [] for g in gammas}

            streaming_frame_accs = []
            streaming_track_terminal = []

            for gid, class_gt, frames in cached_tracks:
                # Corrupt frames independently
                corrupt_tensors = []
                for f_pil in frames:
                    c_im = c_func(f_pil, sev)
                    t = TF.normalize(TF.to_tensor(c_im), mean=IMAGENET_MEAN, std=IMAGENET_STD)
                    corrupt_tensors.append(t)

                batch_30 = torch.stack(corrupt_tensors).to(device)  # [30, 3, 64, 64]

                # --- Regime 1: Single Frame Static ---
                set_bn_mode(model, adapt=False)
                out_static = model(batch_30)
                logits_static = out_static["logits"]  # [30, 43]
                preds_static = torch.argmax(logits_static, dim=-1).cpu().numpy()  # [30]

                acc_single_frames.extend((preds_static == class_gt).astype(int).tolist())
                acc_track_terminal.append(int(preds_static[-1] == class_gt))

                # Majority vote
                vals, counts = np.unique(preds_static, return_counts=True)
                maj_pred = vals[np.argmax(counts)]
                acc_track_majority.append(int(maj_pred == class_gt))

                # Mean logits
                mean_l = logits_static.mean(dim=0).argmax().item()
                acc_track_mean_logits.append(int(mean_l == class_gt))

                # --- Regime 2: Temporal Concept & Logit EMA Smoothing ---
                c_hat_seq = out_static["c_hat"]  # [30, 22]
                aux_seq = out_static["logits_aux"]  # [30, 43]

                for g in gammas:
                    # Recursive EMA filter
                    c_bar = c_hat_seq[0].clone()
                    aux_bar = aux_seq[0].clone()
                    ema_preds = [preds_static[0]]

                    for t in range(1, 30):
                        c_bar = g * c_bar + (1.0 - g) * c_hat_seq[t]
                        aux_bar = g * aux_bar + (1.0 - g) * aux_seq[t]
                        # Compute smoothed class prediction: class_head(c_bar) + aux_bar
                        pred_t = torch.argmax(model.class_head(c_bar.unsqueeze(0)) + aux_bar.unsqueeze(0), dim=-1).item()
                        ema_preds.append(pred_t)

                    ema_frame_accs[g].extend((np.array(ema_preds) == class_gt).astype(int).tolist())
                    ema_track_terminal[g].append(int(ema_preds[-1] == class_gt))

                # --- Regime 3: Online Streaming BN-Adapt (N=8) + EMA (gamma=0.7) ---
                # Split 30 frames into chunks of 8 (last chunk has 6 frames)
                set_bn_mode(model, adapt=True)
                streaming_preds = []
                for ch_start in range(0, 30, 8):
                    ch_x = batch_30[ch_start : min(30, ch_start + 8)]
                    ch_out = model(ch_x)
                    streaming_preds.extend(torch.argmax(ch_out["logits"], dim=-1).cpu().numpy().tolist())

                streaming_frame_accs.extend((np.array(streaming_preds) == class_gt).astype(int).tolist())
                streaming_track_terminal.append(int(streaming_preds[-1] == class_gt))

            rec = {
                "corruption": c_name,
                "severity": sev,
                "single_frame_acc": float(np.mean(acc_single_frames)),
                "terminal_frame_acc": float(np.mean(acc_track_terminal)),
                "majority_vote_acc": float(np.mean(acc_track_majority)),
                "mean_logits_acc": float(np.mean(acc_track_mean_logits)),
                "ema_frame_acc": {f"gamma_{g}": float(np.mean(ema_frame_accs[g])) for g in gammas},
                "ema_terminal_acc": {f"gamma_{g}": float(np.mean(ema_track_terminal[g])) for g in gammas},
                "streaming_bn_frame_acc": float(np.mean(streaming_frame_accs)),
                "streaming_bn_terminal_acc": float(np.mean(streaming_track_terminal)),
            }
            results_records.append(rec)
            pbar.update(1)

    pbar.close()

    df_res = pd.DataFrame(results_records)
    summary = {
        "checkpoint": str(checkpoint_path),
        "total_tracks": len(cached_tracks),
        "total_frames_evaluated": len(cached_tracks) * 30,
        "mean_single_frame_acc": float(df_res["single_frame_acc"].mean()),
        "mean_terminal_frame_acc": float(df_res["terminal_frame_acc"].mean()),
        "mean_majority_vote_acc": float(df_res["majority_vote_acc"].mean()),
        "mean_mean_logits_acc": float(df_res["mean_logits_acc"].mean()),
        "mean_ema_frame_acc": {f"gamma_{g}": float(np.mean([r[f'gamma_{g}'] for r in df_res['ema_frame_acc']])) for g in gammas},
        "mean_ema_terminal_acc": {f"gamma_{g}": float(np.mean([r[f'gamma_{g}'] for r in df_res['ema_terminal_acc']])) for g in gammas},
        "mean_streaming_bn_frame_acc": float(df_res["streaming_bn_frame_acc"].mean()),
        "mean_streaming_bn_terminal_acc": float(df_res["streaming_bn_terminal_acc"].mean()),
        "condition_details": results_records,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved temporal evaluation report to {out_file}")

    print("\n" + "=" * 80)
    print("TEMPORAL TRACK EVALUATION SUMMARY HIGHLIGHTS")
    print(f"Single-Frame Accuracy          : {summary['mean_single_frame_acc']*100:.2f}%")
    print(f"Terminal Frame (t=30) Accuracy : {summary['mean_terminal_frame_acc']*100:.2f}%")
    print(f"Track Majority Vote Accuracy   : {summary['mean_majority_vote_acc']*100:.2f}% (Gain: {(summary['mean_majority_vote_acc'] - summary['mean_single_frame_acc'])*100:+.2f}%)")
    print(f"Track Mean Logits Accuracy     : {summary['mean_mean_logits_acc']*100:.2f}% (Gain: {(summary['mean_mean_logits_acc'] - summary['mean_single_frame_acc'])*100:+.2f}%)")
    for g in gammas:
        ema_f = summary['mean_ema_frame_acc'][f'gamma_{g}']
        print(f"EMA Smoothing (gamma={g})      : Frame Acc = {ema_f*100:.2f}% (Gain: {(ema_f - summary['mean_single_frame_acc'])*100:+.2f}%)")
    st_f = summary['mean_streaming_bn_frame_acc']
    print(f"Streaming BN-Adapt (N=8)       : Frame Acc = {st_f*100:.2f}% (Gain: {(st_f - summary['mean_single_frame_acc'])*100:+.2f}%)")
    print("=" * 80)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate Temporal 30-Frame Concept Smoothing on GTSRB.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_cbm_resnet18.pt")
    parser.add_argument("--val-meta", type=str, default="data/master_v1/metadata/val.csv", help="Path to val.csv")
    parser.add_argument("--data-root", type=str, default="german", help="Path to raw GTSRB folder")
    parser.add_argument("--output", type=str, default="reports/diagnostics/temporal_track_smoothing.json", help="Output JSON path")
    parser.add_argument("--limit-tracks", type=int, default=None, help="Limit number of tracks for quick local test")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu' or 'cuda')")
    args = parser.parse_args()

    evaluate_temporal(
        checkpoint_path=args.checkpoint,
        val_meta_path=args.val_meta,
        data_root=args.data_root,
        output_path=args.output,
        limit_tracks=args.limit_tracks,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()

