"""
Generate GTSRB Common Corruptions Matrix Visualization
======================================================
Produces a publication-grade 15-row x 6-column matrix showing a canonical GTSRB traffic sign
under Clean condition and all 15 Hendrycks & Dietterich corruptions across Severities 1 to 5.

Exports:
1. High-resolution PNG image: `reports/diagnostics/corruption_matrix_sample.png`
2. Copied to conversation artifact directory for inline preview
3. Interactive HTML Generative UI viewer with base64 images and progressive severity scrubbing.
"""

import argparse
import base64
import io
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.corruptions import CORRUPTIONS_REGISTRY
from src.dataset import letterbox_roi_crop

ARTIFACT_DIR = Path(r"C:\Users\Pratham\.gemini\antigravity\brain\06abf734-3f4e-47e6-add3-10fd6083fd0d")

MODALITY_GROUPS = {
    "Noise": ["gaussian_noise", "shot_noise", "impulse_noise"],
    "Blur": ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "Weather": ["snow", "frost", "fog", "brightness", "contrast"],
    "Digital": ["elastic_transform", "pixelate", "jpeg_compression"],
}

MODALITY_COLORS = {
    "Noise": "#ef4444",    # Red
    "Blur": "#3b82f6",     # Blue
    "Weather": "#06b6d4",  # Cyan
    "Digital": "#8b5cf6",  # Purple
}

CORRUPTION_DISPLAY_NAMES = {
    "gaussian_noise": "Gaussian Noise",
    "shot_noise": "Shot Noise",
    "impulse_noise": "Impulse Noise",
    "defocus_blur": "Defocus Blur",
    "glass_blur": "Glass Blur",
    "motion_blur": "Motion Blur",
    "zoom_blur": "Zoom Blur",
    "snow": "Snow",
    "frost": "Frost",
    "fog": "Fog",
    "brightness": "Brightness",
    "contrast": "Contrast",
    "elastic_transform": "Elastic Transform",
    "pixelate": "Pixelate",
    "jpeg_compression": "JPEG Compression",
}


def pil_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def generate_matrix(class_id: int = 1, output_png: str = "reports/diagnostics/corruption_matrix_sample.png"):
    data_root = Path("german")
    meta_path = Path("data/master_v1/metadata/test.csv")
    out_path = Path(output_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(meta_path)
    sample_rows = test_df[test_df["class_id"] == class_id]
    if len(sample_rows) == 0:
        sample = test_df.iloc[0]
    else:
        sample = sample_rows.iloc[0]

    class_name = sample["class_name"]
    img_rel_path = sample["source_relative_path"]
    full_img_path = data_root / img_rel_path

    roi = (int(sample["roi_x1"]), int(sample["roi_y1"]), int(sample["roi_x2"]), int(sample["roi_y2"]))
    raw_img = Image.open(full_img_path).convert("RGB")
    clean_crop = letterbox_roi_crop(raw_img, roi, target_size=64)

    print(f"Generating corruption matrix for: Class {class_id} - '{class_name}' ({sample['image_id']})")

    # Generate all corrupted images
    all_corruptions = []
    for mod, c_list in MODALITY_GROUPS.items():
        for c in c_list:
            all_corruptions.append((mod, c))

    n_rows = len(all_corruptions)  # 15
    n_cols = 6  # Clean, Sev 1, 2, 3, 4, 5

    # Matplotlib High-Res Figure
    fig = plt.figure(figsize=(15, 26), facecolor="#0f172a")  # Dark slate background
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, wspace=0.08, hspace=0.18,
                           left=0.18, right=0.98, top=0.94, bottom=0.02)

    fig.suptitle(
        f"GTSRB Benchmark Common Corruptions Matrix\nClass {class_id}: {class_name} | 15 Corruptions × 5 Severities (64×64 Letterbox)",
        fontsize=16,
        fontweight="bold",
        color="#f8fafc",
        y=0.98,
    )

    col_titles = ["Clean (Ref)", "Severity 1\n(Mild)", "Severity 2\n(Light)", "Severity 3\n(Moderate)", "Severity 4\n(Severe)", "Severity 5\n(Extreme)"]

    base64_data = {
        "class_id": class_id,
        "class_name": class_name,
        "clean": pil_to_base64(clean_crop),
        "corruptions": {},
    }

    current_modality = None

    for row_idx, (modality, c_name) in enumerate(all_corruptions):
        c_func = CORRUPTIONS_REGISTRY[c_name]
        display_name = CORRUPTION_DISPLAY_NAMES[c_name]
        mod_color = MODALITY_COLORS[modality]

        base64_data["corruptions"][c_name] = {
            "name": display_name,
            "modality": modality,
            "color": mod_color,
            "severities": {},
        }

        # Col 0: Clean reference
        ax_clean = fig.add_subplot(gs[row_idx, 0])
        ax_clean.imshow(clean_crop, interpolation="nearest")
        ax_clean.set_xticks([])
        ax_clean.set_yticks([])
        for spine in ax_clean.spines.values():
            spine.set_color("#475569")
            spine.set_linewidth(1.2)

        if row_idx == 0:
            ax_clean.set_title(col_titles[0], color="#cbd5e1", fontsize=11, fontweight="bold", pad=8)

        # Label row on the left of clean column
        row_label = f"[{modality.upper()}]\n{display_name}"
        ax_clean.text(
            -0.15, 0.5, row_label,
            transform=ax_clean.transAxes,
            color=mod_color,
            fontsize=9.5,
            fontweight="bold",
            va="center",
            ha="right",
        )

        # Cols 1 to 5: Severities
        for sev in range(1, 6):
            col_idx = sev
            corrupted_img = c_func(clean_crop, severity=sev)
            base64_data["corruptions"][c_name]["severities"][sev] = pil_to_base64(corrupted_img)

            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.imshow(corrupted_img, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])

            # Highlight border with modality color
            for spine in ax.spines.values():
                spine.set_color(mod_color)
                spine.set_linewidth(1.0 + 0.3 * sev)

            if row_idx == 0:
                ax.set_title(col_titles[col_idx], color="#cbd5e1", fontsize=11, fontweight="bold", pad=8)

    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"High-resolution matrix saved to: {out_path} ({out_path.stat().st_size / (1024*1024):.2f} MB)")

    # Copy to artifact directory if present
    if ARTIFACT_DIR.exists():
        artifact_png = ARTIFACT_DIR / "corruption_matrix_sample.png"
        shutil.copy2(out_path, artifact_png)
        print(f"Copied matrix PNG to artifact directory: {artifact_png}")

    # Build interactive HTML Generative UI
    generate_html_viewer(base64_data)


def generate_html_viewer(data: dict):
    out_html = ARTIFACT_DIR / "corruption_matrix_viewer.html"

    # Convert base64 data to JSON string for JS embedding
    data_json = json.dumps(data)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>GTSRB Common Corruptions Matrix Explorer</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    .pixelated {{ image-rendering: pixelated; }}
  </style>
</head>
<body class="bg-[var(--background)] text-[var(--foreground)] antialiased p-4">
  <div class="max-w-6xl mx-auto bg-[var(--card)] border border-[var(--border)] rounded-2xl shadow-xl overflow-hidden p-6">
    
    <!-- Header -->
    <div class="flex flex-col md:flex-row md:items-center justify-between pb-6 border-b border-[var(--border)] gap-4">
      <div>
        <div class="flex items-center gap-2">
          <span class="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-blue-500/10 text-blue-500 border border-blue-500/20">GTSRB-C Benchmark</span>
          <span class="text-xs text-[var(--muted-foreground)]">15 Corruptions × 5 Severities</span>
        </div>
        <h1 class="text-2xl font-bold mt-1 text-[var(--foreground)]">Common Corruptions Matrix Explorer</h1>
        <p class="text-sm text-[var(--muted-foreground)]">Sample: <strong class="text-[var(--foreground)]">Class {data['class_id']} ({data['class_name']})</strong> (Letterbox ROI 64×64)</p>
      </div>

      <!-- Modality Filter Buttons -->
      <div class="flex flex-wrap gap-2 items-center" id="filter-buttons">
        <button onclick="filterModality('all')" id="btn-all" class="px-3 py-1.5 rounded-lg text-xs font-medium bg-blue-600 text-white shadow-sm transition">All (15)</button>
        <button onclick="filterModality('Noise')" id="btn-noise" class="px-3 py-1.5 rounded-lg text-xs font-medium bg-[var(--card)] hover:bg-[var(--content)] border border-[var(--border)] text-red-400 transition">Noise</button>
        <button onclick="filterModality('Blur')" id="btn-blur" class="px-3 py-1.5 rounded-lg text-xs font-medium bg-[var(--card)] hover:bg-[var(--content)] border border-[var(--border)] text-blue-400 transition">Blur</button>
        <button onclick="filterModality('Weather')" id="btn-weather" class="px-3 py-1.5 rounded-lg text-xs font-medium bg-[var(--card)] hover:bg-[var(--content)] border border-[var(--border)] text-cyan-400 transition">Weather</button>
        <button onclick="filterModality('Digital')" id="btn-digital" class="px-3 py-1.5 rounded-lg text-xs font-medium bg-[var(--card)] hover:bg-[var(--content)] border border-[var(--border)] text-purple-400 transition">Digital</button>
      </div>
    </div>

    <!-- Live Interactive Inspector / Comparison Stage -->
    <div class="my-6 p-4 rounded-xl bg-[var(--background)] border border-[var(--border)] grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
      <!-- Clean Reference Card -->
      <div class="flex flex-col items-center p-3 rounded-lg bg-[var(--card)] border border-[var(--border)]">
        <span class="text-xs font-bold text-emerald-400 mb-2 uppercase tracking-wide">Clean Ground-Truth</span>
        <img id="stage-clean" src="data:image/png;base64,{data['clean']}" class="w-32 h-32 pixelated rounded border-2 border-emerald-500/50 shadow-md">
        <span class="text-xs text-[var(--muted-foreground)] mt-2">64×64 Baseline</span>
      </div>

      <!-- Slider & Controls -->
      <div class="flex flex-col items-center justify-center text-center px-4">
        <span id="inspector-name" class="text-lg font-bold text-[var(--foreground)]">Defocus Blur</span>
        <span id="inspector-modality" class="text-xs px-2 py-0.5 rounded font-semibold bg-blue-500/20 text-blue-400 my-1">BLUR MODALITY</span>
        
        <div class="w-full mt-4">
          <div class="flex justify-between text-xs font-medium text-[var(--muted-foreground)] mb-1">
            <span>Sev 1 (Mild)</span>
            <span id="severity-label" class="text-amber-400 font-bold">Severity 3 (Moderate)</span>
            <span>Sev 5 (Extreme)</span>
          </div>
          <input type="range" id="severity-slider" min="1" max="5" value="3" oninput="updateInspectorSeverity(this.value)" class="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500">
        </div>
        <p class="text-xs text-[var(--muted-foreground)] mt-3">Click any card below to load into inspector</p>
      </div>

      <!-- Corrupted Inspector View -->
      <div class="flex flex-col items-center p-3 rounded-lg bg-[var(--card)] border border-[var(--border)]">
        <span id="stage-corrupt-title" class="text-xs font-bold text-amber-400 mb-2 uppercase tracking-wide">Defocus Blur (Sev 3)</span>
        <img id="stage-corrupt" class="w-32 h-32 pixelated rounded border-2 border-amber-500/50 shadow-md">
        <span id="stage-corrupt-desc" class="text-xs text-[var(--muted-foreground)] mt-2">Simulated camera defocus</span>
      </div>
    </div>

    <!-- Matrix Grid Table -->
    <div class="overflow-x-auto">
      <table class="w-full text-left border-collapse">
        <thead>
          <tr class="border-b border-[var(--border)] text-xs uppercase tracking-wider text-[var(--muted-foreground)]">
            <th class="py-3 px-3">Corruption / Modality</th>
            <th class="py-3 px-2 text-center">Clean Ref</th>
            <th class="py-3 px-2 text-center">Severity 1</th>
            <th class="py-3 px-2 text-center">Severity 2</th>
            <th class="py-3 px-2 text-center">Severity 3</th>
            <th class="py-3 px-2 text-center">Severity 4</th>
            <th class="py-3 px-2 text-center">Severity 5</th>
          </tr>
        </thead>
        <tbody id="matrix-tbody" class="divide-y divide-[var(--border)] text-sm">
          <!-- Dynamically populated rows -->
        </tbody>
      </table>
    </div>

  </div>

  <script>
    const DATA = {data_json};
    let activeCorruption = "defocus_blur";
    let activeSeverity = 3;
    let currentFilter = "all";

    function updateInspector(cKey, sev) {{
      activeCorruption = cKey;
      activeSeverity = parseInt(sev);
      const cObj = DATA.corruptions[cKey];
      
      document.getElementById("inspector-name").innerText = cObj.name;
      const modEl = document.getElementById("inspector-modality");
      modEl.innerText = cObj.modality.toUpperCase() + " MODALITY";
      modEl.style.color = cObj.color;
      modEl.style.backgroundColor = cObj.color + "22";

      document.getElementById("stage-corrupt-title").innerText = `${{cObj.name}} (Sev ${{activeSeverity}})`;
      document.getElementById("stage-corrupt-title").style.color = cObj.color;
      
      const imgEl = document.getElementById("stage-corrupt");
      imgEl.src = "data:image/png;base64," + cObj.severities[activeSeverity];
      imgEl.style.borderColor = cObj.color;

      document.getElementById("severity-slider").value = activeSeverity;
      const sevNames = ["", "Mild", "Light", "Moderate", "Severe", "Extreme"];
      document.getElementById("severity-label").innerText = `Severity ${{activeSeverity}} (${{sevNames[activeSeverity]}})`;
    }}

    function updateInspectorSeverity(val) {{
      updateInspector(activeCorruption, val);
    }}

    function filterModality(mod) {{
      currentFilter = mod;
      ['all', 'noise', 'blur', 'weather', 'digital'].forEach(id => {{
        const btn = document.getElementById('btn-' + id);
        if (id === mod.toLowerCase()) {{
          btn.className = "px-3 py-1.5 rounded-lg text-xs font-semibold bg-blue-600 text-white shadow-sm transition";
        }} else {{
          btn.className = "px-3 py-1.5 rounded-lg text-xs font-medium bg-[var(--card)] hover:bg-[var(--content)] border border-[var(--border)] text-[var(--muted-foreground)] transition";
        }}
      }});
      renderTable();
    }}

    function renderTable() {{
      const tbody = document.getElementById("matrix-tbody");
      tbody.innerHTML = "";

      for (const [cKey, cObj] of Object.entries(DATA.corruptions)) {{
        if (currentFilter !== "all" && cObj.modality !== currentFilter) continue;

        const tr = document.createElement("tr");
        tr.className = "hover:bg-white/[0.02] transition cursor-pointer";
        tr.onclick = () => updateInspector(cKey, activeSeverity);

        // Name & Badge cell
        const tdName = document.createElement("td");
        tdName.className = "py-3 px-3 whitespace-nowrap";
        tdName.innerHTML = `
          <div class="flex items-center gap-2">
            <span class="w-2.5 h-2.5 rounded-full" style="background-color: ${{cObj.color}}"></span>
            <div>
              <div class="font-semibold text-xs text-[var(--foreground)]">${{cObj.name}}</div>
              <div class="text-[11px] text-[var(--muted-foreground)] uppercase font-mono tracking-tight">${{cObj.modality}}</div>
            </div>
          </div>
        `;
        tr.appendChild(tdName);

        // Clean cell
        const tdClean = document.createElement("td");
        tdClean.className = "py-2 px-2 text-center";
        tdClean.innerHTML = `<img src="data:image/png;base64,${{DATA.clean}}" class="w-12 h-12 mx-auto pixelated rounded border border-slate-700 hover:scale-110 transition shadow-sm">`;
        tr.appendChild(tdClean);

        // Severities 1..5
        for (let s = 1; s <= 5; s++) {{
          const tdSev = document.createElement("td");
          tdSev.className = "py-2 px-2 text-center";
          tdSev.innerHTML = `<img src="data:image/png;base64,${{cObj.severities[s]}}" onclick="event.stopPropagation(); updateInspector('${{cKey}}', ${{s}});" class="w-12 h-12 mx-auto pixelated rounded border hover:scale-110 transition shadow-sm" style="border-color: ${{cObj.color}}">`;
          tr.appendChild(tdSev);
        }}

        tbody.appendChild(tr);
      }}
    }}

    // Initial render
    renderTable();
    updateInspector("defocus_blur", 3);
  </script>
</body>
</html>
"""

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Interactive HTML viewer saved to: {out_html}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-id", type=int, default=1, help="Class ID to visualize (default: 1 for Speed limit 30km/h)")
    parser.add_argument("--output", type=str, default="reports/diagnostics/corruption_matrix_sample.png", help="Output PNG path")
    args = parser.parse_args()
    generate_matrix(args.class_id, args.output)

