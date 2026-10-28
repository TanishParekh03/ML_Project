"""
Package Code for Google Colab Execution
=======================================
Creates a lightweight zip file containing all source code, manifests, and scripts
needed to run training, evaluation, and test-time sweeps on Google Colab T4 GPU.
"""

import argparse
import os
import zipfile
from pathlib import Path


def package_project(output_zip: str = "colab_code_package.zip", include_checkpoints: bool = False):
    root = Path(__file__).resolve().parent
    out_zip_path = root / output_zip

    include_patterns = [
        "src/**/*",
        "scripts/**/*",
        "data/master_v1/metadata/*.csv",
        "data/master_v1/concepts/*.csv",
        "data/master_v1/configs/*",
        "reports/**/*.json",
        "reports/**/*.csv",
        "reports/diagnostics/**/*",
        "train_*.py",
        "evaluate_*.py",
        "corruption_code.py",
        "build_master_dataset.py",
        "notebooks/*.ipynb",
    ]

    if include_checkpoints:
        include_patterns.extend([
            "reports/**/*.pt",
            "reports/**/*.pth",
        ])

    exclude_patterns = [
        "__pycache__",
        "*.pyc",
        ".git",
        ".agents",
        "german/**",  # Raw dataset excluded (mounted from Drive)
        "graphify-out/**",
        "*.zip",
    ]

    print(f"Creating Colab code archive: {out_zip_path}...")
    file_count = 0
    total_bytes = 0
    added_paths = set()

    with zipfile.ZipFile(out_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for pattern in include_patterns:
            for p in root.glob(pattern):
                if p.is_dir():
                    continue
                # Check exclusions
                rel_p = p.relative_to(root)
                rel_str = str(rel_p).replace("\\", "/")

                if rel_str in added_paths:
                    continue

                skip = False
                for exc in exclude_patterns:
                    if exc.endswith("/**") and rel_str.startswith(exc[:-3]):
                        skip = True
                        break
                    elif exc in rel_str:
                        skip = True
                        break

                if not skip:
                    zipf.write(p, arcname=rel_str)
                    added_paths.add(rel_str)
                    file_count += 1
                    total_bytes += p.stat().st_size

    mb = total_bytes / (1024 * 1024)
    zip_mb = out_zip_path.stat().st_size / (1024 * 1024)
    print(f"Successfully packaged {file_count} files ({mb:.2f} MB uncompressed -> {zip_mb:.2f} MB compressed)")
    print(f"Archive saved to: {out_zip_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None, help="Output zip name")
    parser.add_argument("--include-weights", action="store_true", help="Include model weight checkpoints")
    args = parser.parse_args()

    if args.output:
        package_project(args.output, args.include_weights)
    else:
        print("=== Generating Full Package (with All 5 Model Weights) ===")
        package_project("colab_code_package_with_weights.zip", include_checkpoints=True)
        print("\n=== Generating Lightweight Code Package (without Weights) ===")
        package_project("colab_code_package.zip", include_checkpoints=False)

