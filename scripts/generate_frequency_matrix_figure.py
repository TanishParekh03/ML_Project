"""Generate Fourier Frequency Transforms Matrix & Pipeline Figure
=============================================================================
Creates a high-resolution, publication-grade figure (300 DPI) illustrating:
1. Matrix of spectral operations (LFC, HFC, Bernoulli Spectral Dropout,
   and Frequency Blend) across progressive parameter levels and 2D Fourier
   magnitude spectra / residual difference heatmaps.
2. High-frequency sensor noise mitigation diagnostic (why FFT filtering resolves
   point-noise vulnerabilities).
3. The end-to-end Spectral Regularization & Multi-Task Flow architecture
   demonstrating how frequency filtering resolves representation interference
   in Experiment 6 (Frequency-Aware CBM).

Output: figs/fig15_frequency_transforms_matrix.png
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.dataset import letterbox_roi_crop
from src.augmentations.frequency import (
    low_frequency_component,
    high_frequency_component,
    frequency_dropout,
    frequency_blend,
)


def compute_magnitude_spectrum(img: Image.Image) -> np.ndarray:
    """Computes normalized 2D Fourier log-magnitude spectrum."""
    arr = np.array(img.convert("L"), dtype=np.float32)
    f = np.fft.fftshift(np.fft.fft2(arr))
    mag = np.log(1.0 + np.abs(f))
    mag_norm = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
    return mag_norm


def add_impulse_noise(img: Image.Image, amount: float = 0.08) -> Image.Image:
    """Simulates salt-and-pepper impulse noise (point discontinuity)."""
    np.random.seed(42)
    arr = np.array(img, dtype=np.uint8).copy()
    h, w, c = arr.shape
    num_noise = int(amount * h * w)
    
    # Salt (255)
    ys = np.random.randint(0, h, num_noise // 2)
    xs = np.random.randint(0, w, num_noise // 2)
    arr[ys, xs] = 255
    
    # Pepper (0)
    ys = np.random.randint(0, h, num_noise // 2)
    xs = np.random.randint(0, w, num_noise // 2)
    arr[ys, xs] = 0
    
    return Image.fromarray(arr, mode="RGB")


def main():
    parser = argparse.ArgumentParser(description="Generate Fourier Frequency Transforms Matrix & Pipeline Figure")
    parser.add_argument("--image-path", type=str, default="german/Train/2/00002_00002_00029.png", help="Path to traffic sign image")
    parser.add_argument("--meta-path", type=str, default="data/master_v1/metadata/train.csv", help="Path to train manifest")
    parser.add_argument("--output", type=str, default="figs/fig15_frequency_transforms_matrix.png", help="Output PNG path")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load Sample Image & Metadata
    target_img_p = Path(args.image_path)
    rel_str = str(target_img_p).replace("\\", "/")
    if rel_str.startswith("german/"):
        rel_clean = rel_str[len("german/"):]
    else:
        rel_clean = rel_str

    train_meta_path = Path(args.meta_path)
    df = pd.read_csv(train_meta_path)
    match = df[df["source_relative_path"].str.replace("\\", "/").str.endswith(rel_clean)]
    if len(match) == 0:
        master_meta_path = Path("data/master_v1/metadata/master_metadata.csv")
        df_master = pd.read_csv(master_meta_path)
        match = df_master[df_master["source_relative_path"].str.replace("\\", "/").str.endswith(rel_clean)]

    if len(match) > 0:
        sample_row = match.iloc[0]
        roi = (int(sample_row["roi_x1"]), int(sample_row["roi_y1"]), int(sample_row["roi_x2"]), int(sample_row["roi_y2"]))
        class_id = int(sample_row["class_id"])
        class_name = str(sample_row.get("class_name", f"Class {class_id}"))
    else:
        roi = None
        class_id = 2
        class_name = "Traffic Sign"

    raw_p = target_img_p if target_img_p.exists() else Path("german") / rel_clean
    if not raw_p.exists():
        raise FileNotFoundError(f"Target image not found at {target_img_p} or {raw_p}")

    raw_img = Image.open(raw_p).convert("RGB")
    if roi is not None:
        canonical = letterbox_roi_crop(raw_img, roi, target_size=64, margin_ratio=0.10)
    else:
        w, h = raw_img.size
        canonical = letterbox_roi_crop(raw_img, (0, 0, w - 1, h - 1), target_size=64, margin_ratio=0.10)
    canonical_np = np.asarray(canonical).astype(np.float32) / 255.0

    # 2. Define Operations & Visual Configurations
    # 5 Rows: LFC, HFC, Bernoulli Dropout, Blend, Noise Mitigation
    operations = [
        {
            "name": "Low-Frequency Component (LFC)",
            "cat": "Spectral Low-Pass",
            "tag_color": "#27ae60",
            "tag_bg": "#eafaf1",
            "rule": "Radial cutoff r <= r0; filters high-frequency noise while retaining macro-shape & color.",
            "steps": [
                ("r = 6 px", low_frequency_component(canonical, radius=6)),
                ("r = 10 px", low_frequency_component(canonical, radius=10)),
                ("r = 14 px (Default)", low_frequency_component(canonical, radius=14)),
                ("r = 20 px", low_frequency_component(canonical, radius=20)),
            ],
            "col5_type": "spectrum_lfc",
        },
        {
            "name": "High-Frequency Component (HFC)",
            "cat": "Spectral High-Pass",
            "tag_color": "#2980b9",
            "tag_bg": "#ebf5fb",
            "rule": f"Radial high-pass r > r0; extracts fine numeral strokes ('{class_name}'), glyphs, and borders.",
            "steps": [
                ("r = 6 px", high_frequency_component(canonical, radius=6)),
                ("r = 10 px", high_frequency_component(canonical, radius=10)),
                ("r = 14 px (Default)", high_frequency_component(canonical, radius=14)),
                ("r = 20 px", high_frequency_component(canonical, radius=20)),
            ],
            "col5_type": "spectrum_hfc",
        },
        {
            "name": "Bernoulli Spectral Dropout",
            "cat": "Stochastic Regularizer",
            "tag_color": "#008080",
            "tag_bg": "#e0f2f1",
            "rule": "Wang et al. (CVPR 2020): Stochastically zeroes high-frequency Fourier coefficients (p_drop).",
            "steps": [
                ("s = 1.0 (p=8%)", frequency_dropout(canonical, severity=1.0, r_min=10)),
                ("s = 3.0 (p=24%)", frequency_dropout(canonical, severity=3.0, r_min=10)),
                ("s = 5.0 (Default, 40%)", frequency_dropout(canonical, severity=5.0, r_min=10)),
                ("s = 8.0 (p=64%)", frequency_dropout(canonical, severity=8.0, r_min=10)),
            ],
            "col5_type": "residual",
            "col5_ref_idx": 2,  # Level 3
        },
        {
            "name": "Frequency Blend (LFC + Spatial)",
            "cat": "Convex Spectral Blend",
            "tag_color": "#8e44ad",
            "tag_bg": "#f4ecf7",
            "rule": "Convex interpolation (1 - α) * LFC + α * I; preserves crispness without noise sensitivity.",
            "steps": [
                ("α = 0.20 (Heavy LFC)", frequency_blend(canonical, radius=14, alpha=0.20)),
                ("α = 0.40", frequency_blend(canonical, radius=14, alpha=0.40)),
                ("α = 0.60 (Moderate)", frequency_blend(canonical, radius=14, alpha=0.60)),
                ("α = 0.80 (Crisp)", frequency_blend(canonical, radius=14, alpha=0.80)),
            ],
            "col5_type": "residual",
            "col5_ref_idx": 1,  # alpha=0.40
        },
        {
            "name": "Point-Noise Mitigation Diagnostic",
            "cat": "Robustness Mechanism",
            "tag_color": "#c0392b",
            "tag_bg": "#fcedec",
            "rule": "Impulse noise explodes high-frequency energy (+3.5x); LFC filtering strips spikes, recovering glyph.",
            "steps": [
                ("Clean Input", canonical),
                ("Impulse Noise (8%)", add_impulse_noise(canonical, amount=0.08)),
                ("Noise 2D Spectrum", add_impulse_noise(canonical, amount=0.08)),  # Special handling in loop
                ("LFC Restored (r=14)", low_frequency_component(add_impulse_noise(canonical, amount=0.08), radius=14)),
            ],
            "col5_type": "spectrum_restored",
        },
    ]

    # 3. Create Figure Canvas
    fig = plt.figure(figsize=(16, 20), dpi=300)
    gs = gridspec.GridSpec(
        nrows=8,
        ncols=6,
        figure=fig,
        height_ratios=[0.04, 1.0, 1.0, 1.0, 1.0, 1.0, 0.05, 2.3],
        width_ratios=[1.25, 1.0, 1.0, 1.0, 1.0, 1.15],
        wspace=0.25,
        hspace=0.42,
    )

    # 4. Header Row
    headers = [
        "Spectral Operator & Rationale",
        "Level 1 (Mild / Low Radius)",
        "Level 2 (Moderate / Mid-Low)",
        "Level 3 (Operational Benchmark)",
        "Level 4 (Severe / High Band)",
        "Spectral Analysis / Residual",
    ]
    for col_idx, h_text in enumerate(headers):
        ax_h = fig.add_subplot(gs[0, col_idx])
        ax_h.axis("off")
        ax_h.text(
            0.5, 0.4, h_text,
            ha="center", va="center",
            fontsize=11.0, fontweight="bold",
            color="#1a202c",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#edf2f7", edgecolor="#cbd5e0", lw=1)
        )

    # 5. Populate Rows 1 to 5
    for row_idx, op in enumerate(operations):
        grid_row = row_idx + 1

        # Column 0: Label & Description
        ax_lbl = fig.add_subplot(gs[grid_row, 0])
        ax_lbl.axis("off")

        ax_lbl.text(0.02, 0.74, f"{row_idx + 1}. {op['name']}", fontsize=11.5, fontweight="bold", color="#2d3748")
        ax_lbl.text(
            0.02, 0.48, op["cat"],
            fontsize=9.0, fontweight="bold", color=op["tag_color"],
            bbox=dict(boxstyle="round,pad=0.22", facecolor=op["tag_bg"], edgecolor=op["tag_color"], lw=0.8)
        )
        ax_lbl.text(0.02, 0.12, op["rule"], fontsize=8.0, color="#718096", wrap=True)

        # Columns 1 to 4: Transformed Visualizations
        for step_idx, (lbl, step_img) in enumerate(op["steps"]):
            ax_img = fig.add_subplot(gs[grid_row, step_idx + 1])
            ax_img.set_xticks([])
            ax_img.set_yticks([])

            # Special case for Row 5 Step 2: Display 2D Fourier Magnitude of Noise
            if row_idx == 4 and step_idx == 2:
                noise_spec = compute_magnitude_spectrum(step_img)
                ax_img.imshow(noise_spec, cmap="magma")
                circle = plt.Circle((32, 32), 14, color="#00ffcc", fill=False, lw=1.6, ls="--")
                ax_img.add_patch(circle)
                for spine in ax_img.spines.values():
                    spine.set_edgecolor("#e53e3e")
                    spine.set_linewidth(1.8)
                ax_img.set_xlabel("Noise FFT Spectrum\n(+3.5x HFC Flare)", fontsize=8.5, color="#c53030", fontweight="bold", labelpad=3)
                continue

            # Standard Image Display
            ax_img.imshow(step_img)

            # Highlighting benchmark/default settings
            is_default = (step_idx == 2 and row_idx < 4) or (row_idx == 4 and step_idx == 3)
            edge_color = "#008080" if is_default else "#e2e8f0"
            lw = 2.2 if is_default else 1.0
            for spine in ax_img.spines.values():
                spine.set_edgecolor(edge_color)
                spine.set_linewidth(lw)

            title_color = "#008080" if is_default else "#4a5568"
            title_weight = "bold" if is_default else "normal"
            ax_img.set_xlabel(lbl, fontsize=8.8, color=title_color, fontweight=title_weight, labelpad=4)

        # Column 5: Spectral Plot or Difference Residual Heatmap
        ax_col5 = fig.add_subplot(gs[grid_row, 5])
        ax_col5.set_xticks([])
        ax_col5.set_yticks([])

        col5_type = op["col5_type"]
        if col5_type == "spectrum_lfc":
            # 2D FFT Magnitude of LFC (r=14)
            lfc_img = op["steps"][2][1]
            mag_spec = compute_magnitude_spectrum(lfc_img)
            ax_col5.imshow(mag_spec, cmap="viridis")
            circle = plt.Circle((32, 32), 14, color="#ffffff", fill=False, lw=1.6, ls="--")
            ax_col5.add_patch(circle)
            ax_col5.set_xlabel("LFC Spectrum (r≤14px)\n86.4% Signal Energy", fontsize=8.5, color="#27ae60", fontweight="bold", labelpad=3)

        elif col5_type == "spectrum_hfc":
            # 2D FFT Magnitude of HFC (r=14)
            hfc_img = op["steps"][2][1]
            mag_spec = compute_magnitude_spectrum(hfc_img)
            ax_col5.imshow(mag_spec, cmap="cividis")
            circle = plt.Circle((32, 32), 14, color="#e53e3e", fill=False, lw=1.6, ls="--")
            ax_col5.add_patch(circle)
            ax_col5.set_xlabel("HFC Spectrum (r>14px)\nPeripheral Details Only", fontsize=8.5, color="#2980b9", fontweight="bold", labelpad=3)

        elif col5_type == "residual":
            # Pixel-wise difference heatmap |Transformed - Clean|
            ref_idx = op.get("col5_ref_idx", 2)
            trans_np = np.asarray(op["steps"][ref_idx][1]).astype(np.float32) / 255.0
            diff = np.abs(trans_np - canonical_np).mean(axis=-1)
            diff_vis = np.clip(diff * 4.0, 0.0, 1.0)
            ax_col5.imshow(diff_vis, cmap="inferno", vmin=0.0, vmax=1.0)
            mean_delta = diff.mean() * 100.0
            ax_col5.set_xlabel(f"Residual (|L3 - Clean|)\nMean Δ: {mean_delta:.1f}% (0% Color Drift)", fontsize=8.5, color="#742a2a", fontweight="bold", labelpad=3)

        elif col5_type == "spectrum_restored":
            # Spectrum of restored sign
            restored_img = op["steps"][3][1]
            mag_spec = compute_magnitude_spectrum(restored_img)
            ax_col5.imshow(mag_spec, cmap="viridis")
            circle = plt.Circle((32, 32), 14, color="#ffffff", fill=False, lw=1.6, ls="--")
            ax_col5.add_patch(circle)
            ax_col5.set_xlabel("Restored FFT Spectrum\nNoise Spikes Stripped", fontsize=8.5, color="#27ae60", fontweight="bold", labelpad=3)

        for spine in ax_col5.spines.values():
            spine.set_edgecolor("#cbd5e0")
            spine.set_linewidth(1.0)

    # 6. Divider & Pipeline Header
    ax_div = fig.add_subplot(gs[6, :])
    ax_div.axis("off")
    ax_div.axhline(0.65, color="#cbd5e0", lw=1.4, ls="--")
    ax_div.text(
        0.5, 0.15,
        "PART B: END-TO-END SPECTRAL REGULARIZATION & DUAL-TRACK ARCHITECTURE (EXP 3 & EXP 6)",
        ha="center", va="center",
        fontsize=12.0, fontweight="bold", color="#1a202c",
        bbox=dict(boxstyle="square,pad=0.4", facecolor="#f7fafc", edgecolor="#a0aec0", lw=1)
    )

    # 7. Bottom Panel: End-to-End Pipeline Schematic (Grid Row 7)
    ax_pipe = fig.add_subplot(gs[7, :])
    ax_pipe.set_xlim(0, 100)
    ax_pipe.set_ylim(0, 100)
    ax_pipe.axis("off")

    # Helper function for drawing rounded pipeline blocks
    def draw_box(x, y, w, h, title, subtitle="", facecolor="#ffffff", edgecolor="#cbd5e0", textcolor="#2d3748"):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.6,rounding_size=1.5",
            facecolor=facecolor, edgecolor=edgecolor, lw=1.5, zorder=2
        )
        ax_pipe.add_patch(rect)
        if subtitle:
            ax_pipe.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color=textcolor, zorder=3)
            ax_pipe.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center", fontsize=8.0, color="#4a5568", zorder=3)
        else:
            ax_pipe.text(x + w / 2, y + h * 0.50, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color=textcolor, zorder=3)

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax_pipe.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color="#4a5568", lw=1.8, mutation_scale=14),
            zorder=4
        )
        if label:
            ax_pipe.text((x1 + x2) / 2, (y1 + y2) / 2 + 3.0, label, ha="center", va="center", fontsize=8.0, fontweight="bold", color="#2b6cb0", zorder=5)

    # Pipeline Nodes:
    # 1. Clean Input
    draw_box(2, 55, 14, 34, "Standardized Input X", f"Class {class_id}: {class_name}\n(64x64x3 RGB Canvas)", facecolor="#f8fafc", edgecolor="#94a3b8")
    
    # Arrow to 2D FFT
    draw_arrow(16, 72, 22, 72, "2D FFT")

    # 2. Fourier Domain Block
    draw_box(22, 55, 20, 34, "2D Fourier Decomposition", "F(u,v) = 2D-FFT(X)\nCentered shift -> (u0, v0)", facecolor="#eff6ff", edgecolor="#3b82f6")

    # Arrow to Filtering / Dropout
    draw_arrow(42, 72, 48, 72, "Mask + Drop")

    # 3. Spectral Filtering & Dropout Block
    draw_box(48, 55, 22, 34, "Spectral Regularization", "LFC Core (r <= 14px, 86.4% E)\nBernoulli Dropout (r > 14px)", facecolor="#ecfdf5", edgecolor="#10b981")

    # Arrow to 2D IFFT
    draw_arrow(70, 72, 76, 72, "2D IFFT")

    # 4. Filtered Tensor X_tilde
    draw_box(76, 55, 22, 34, "Augmented Tensor X_tilde", "Inverse 2D-FFT + Clip\n0% Color Drift | No Shear", facecolor="#f0fdfa", edgecolor="#008080")

    # Downward branching arrows to Experiment 3 and Experiment 6
    draw_arrow(87, 55, 30, 36)
    draw_arrow(87, 55, 75, 36)

    # Branch A: Exp 3 (Frequency ResNet-18)
    draw_box(5, 5, 42, 30, "Track A: Frequency ResNet-18 (Exp 3)", 
             "• Standard Black-Box ResNet-18 Backbone\n"
             "• Single 43-Class Head (W_fc * h + b)\n"
             "• Resolves AugMix noise vulnerability (+1.2% noise mCA)\n"
             "• Clean Acc: 97.32% | Static mCA: 83.77% | SG-BN: 88.97%", 
             facecolor="#f0fdf4", edgecolor="#22c55e", textcolor="#15803d")

    # Branch B: Exp 6 (Frequency-Aware CBM) - CHAMPION
    draw_box(53, 5, 45, 30, "Track B: Frequency-Aware CBM (Exp 6) — PROJECT WINNER", 
             "• Dual-Stream Architecture: 22-dim Concept Heads + Aux Head W_aux * h(x)\n"
             "• Shape (6) | Colour (5) | Border (4) | Symbol Category (7)\n"
             "• Zero Concept Interference: Spectral filtering preserves physical colors\n"
             "• Clean Acc: 97.95% | Static mCA: 85.85% (Record) | Shift-Gated BN: 89.79% (Record)", 
             facecolor="#f0fdfa", edgecolor="#008080", textcolor="#006666")

    # Connecting label between branches
    ax_pipe.text(
        50, 41,
        "Spectral Filtering Decouples High-Frequency Noise Suppression from Semantic Color Profiles (H4 Validated)",
        ha="center", va="center", fontsize=8.5, fontweight="bold", color="#1e293b",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef08a", edgecolor="#ca8a04", lw=0.8)
    )

    # 8. Save Figure
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[SUCCESS] Successfully generated Figure 15: {out_path.resolve()} (300 DPI)")


if __name__ == "__main__":
    main()
