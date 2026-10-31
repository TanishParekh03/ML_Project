"""
Package Phase 2 Project for Google Colab Execution
===================================================
Creates a self-contained zip archive containing:
- Full Python source code (src/ and scripts/)
- Standardized datasets (data_v2 Belgium and data_v3 China including pre-rendered 64x64 images)
- Pretrained German checkpoints for Phase 1 cross-evaluation
- Colab Phase 2 interactive notebook
"""

import argparse
import os
import zipfile
from pathlib import Path


def package_phase2(output_zip: str = "phase2_colab_package.zip"):
    root = Path(__file__).resolve().parent
    out_zip_path = root / output_zip

    include_patterns = [
        "src/**/*",
        "scripts/**/*",
        "data/data_v2/metadata/*.csv",
        "data/data_v2/concepts/*",
        "data/data_v2/configs/*",
        "data/data_v2/images/**/*",
        "data/data_v3/metadata/*.csv",
        "data/data_v3/concepts/*",
        "data/data_v3/configs/*",
        "data/data_v3/images/**/*",
        "reports/baseline_resnet18/best_baseline_resnet18.pt",
        "reports/cbm_resnet18/best_cbm_resnet18.pt",
        "reports/cross_dataset_eval/**/*",
        "notebooks/phase2_cross_dataset_robustness_colab.ipynb",
    ]

    exclude_patterns = [
        "__pycache__",
        "*.pyc",
        ".git",
        ".agents",
        "graphify-out",
        "german",
        "belgium",
        "chinese",
    ]

    print(f"Building Phase 2 Colab Archive: {out_zip_path}...")
    file_count = 0
    total_bytes = 0
    added_paths = set()

    with zipfile.ZipFile(out_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for pattern in include_patterns:
            for p in root.glob(pattern):
                if p.is_file():
                    rel_p = p.relative_to(root)
                    rel_str = str(rel_p).replace("\\", "/")

                    # Check exclusions
                    if any(ex in rel_str for ex in exclude_patterns):
                        continue

                    if rel_str not in added_paths:
                        added_paths.add(rel_str)
                        zipf.write(p, rel_str)
                        file_count += 1
                        total_bytes += p.stat().st_size

    archive_size_mb = out_zip_path.stat().st_size / (1024 * 1024)
    raw_size_mb = total_bytes / (1024 * 1024)
    print(f"Archive successfully created: {out_zip_path.name}")
    print(f"Total files: {file_count} | Uncompressed: {raw_size_mb:.2f} MB | Compressed: {archive_size_mb:.2f} MB")
    return out_zip_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package Phase 2 for Colab")
    parser.add_argument("--output", type=str, default="phase2_colab_package.zip", help="Output zip filename")
    args = parser.parse_args()
    package_phase2(args.output)

