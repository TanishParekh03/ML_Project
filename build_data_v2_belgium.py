"""Build BelgiumTSC Master Dataset (data_v2)
=========================================
Strictly implements the Master Dataset specification from GTSRB (master_v1):
1. Anti-leakage: Stratified group train/val split (80/20) by (class_id + pole_id)
   guaranteeing ZERO physical pole overlap between train and validation.
2. Official held-out test split: Preserves BelgiumTSC_Testing (2,534 images) strictly untouched.
3. 4-Stage Preprocessing Contract:
   - Context-buffered ROI crop with 10% margin (clamped to image bounds).
   - Proportional bilinear letterbox resize to max dimension 64 (aspect-ratio preserved).
   - Symmetric center zero-padding on a 64x64 black canvas.
4. Schema-complete metadata manifests in data/data_v2/metadata/:
   - train.csv, val.csv, test.csv, master_metadata.csv.
5. Standardized 64x64 PNG images rendered to data/data_v2/images/.
6. Legally grounded concept ontology mapping in data/data_v2/concepts/class_concepts.csv.
7. Cryptographic SHA-256 manifests record in data/data_v2/configs/dataset_version.md.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

BELGIAN_SIGN_CATALOG = {
    'A13': ('Pedestrian crossing', 'triangle_up', 'white_red', 'red_border', 'warning_hazard', 'pedestrian'),
    'A14': ('Children / School zone', 'triangle_up', 'white_red', 'red_border', 'warning_hazard', 'children'),
    'A15': ('Cyclists and moped riders', 'triangle_up', 'white_red', 'red_border', 'warning_hazard', 'cyclist'),
    'A1A': ('Dangerous bend to the left', 'triangle_up', 'white_red', 'red_border', 'warning_hazard', 'curve_left'),
    'A1B': ('Dangerous bend to the right', 'triangle_up', 'white_red', 'red_border', 'warning_hazard', 'curve_right'),
    'A1C': ('Double bend, first to the left', 'triangle_up', 'white_red', 'red_border', 'warning_hazard', 'double_curve_left'),
    'A1D': ('Double bend, first to the right', 'triangle_up', 'white_red', 'red_border', 'warning_hazard', 'double_curve_right'),
    'A23': ('Road works', 'triangle_up', 'white_red', 'red_border', 'warning_hazard', 'road_work'),
    'A25': ('Traffic light signals ahead', 'triangle_up', 'white_red', 'red_border', 'warning_hazard', 'traffic_light'),
    'A29': ('Low-flying aircraft', 'triangle_up', 'white_red', 'red_border', 'warning_hazard', 'aircraft'),
    'A31': ('Crosswinds', 'triangle_up', 'white_red', 'red_border', 'warning_hazard', 'crosswinds'),
    'A33': ('Two-way traffic ahead', 'triangle_up', 'white_red', 'red_border', 'warning_hazard', 'two_way_traffic'),
    'A41': ('Level crossing with barriers', 'triangle_up', 'white_red', 'red_border', 'warning_hazard', 'railway_barrier'),
    'A51': ('General danger / Other hazards', 'triangle_up', 'white_red', 'red_border', 'warning_hazard', 'general_danger'),
    'A7A': ('Steep hill downwards (descent)', 'triangle_up', 'white_red', 'red_border', 'warning_hazard', 'steep_descent'),
    'A7B': ('Steep hill upwards (ascent)', 'triangle_up', 'white_red', 'red_border', 'warning_hazard', 'steep_ascent'),
    'A7C': ('Quay or river bank / Slippery water hazard', 'triangle_up', 'white_red', 'red_border', 'warning_hazard', 'quay_water'),
    'B15A': ('Priority road intersection with minor road', 'triangle_up', 'white_red', 'red_border', 'priority_geometry', 'priority_intersection'),
    'B17': ('Priority to the right intersection', 'triangle_up', 'white_red', 'red_border', 'priority_geometry', 'priority_right'),
    'B1': ('Give way / Yield', 'triangle_down', 'white_red', 'red_border', 'priority_geometry', 'yield'),
    'B19': ('Narrow passage: Give way to oncoming vehicles', 'circle', 'white_red', 'red_border', 'priority_geometry', 'priority_oncoming'),
    'B5': ('Stop and give way', 'octagon', 'red_white', 'white_border', 'priority_geometry', 'stop'),
    'C1': ('No entry for vehicular traffic', 'circle', 'red_white', 'none', 'prohibition_restriction', 'no_entry'),
    'C11': ('No motorcycles', 'circle', 'white_red', 'red_border', 'prohibition_restriction', 'no_motorcycles'),
    'C21': ('Weight limit', 'circle', 'white_red', 'red_border', 'prohibition_restriction', 'weight_limit'),
    'C23': ('No goods vehicles / trucks', 'circle', 'white_red', 'red_border', 'prohibition_restriction', 'no_trucks'),
    'C27': ('Height limit', 'circle', 'white_red', 'red_border', 'prohibition_restriction', 'height_limit'),
    'C29': ('Length limit', 'circle', 'white_red', 'red_border', 'prohibition_restriction', 'length_limit'),
    'C3': ('No vehicular traffic in both directions', 'circle', 'white_red', 'red_border', 'prohibition_restriction', 'no_traffic'),
    'C31LEFT': ('No left turn', 'circle', 'white_red', 'red_border', 'prohibition_restriction', 'no_left_turn'),
    'C31RIGHT': ('No right turn', 'circle', 'white_red', 'red_border', 'prohibition_restriction', 'no_right_turn'),
    'C35': ('No passing / overtaking', 'circle', 'white_red', 'red_border', 'prohibition_restriction', 'no_passing'),
    'C43': ('Speed limit', 'circle', 'white_red', 'red_border', 'speed_numeral', 'speed_numeral'),
    'D10': ('Shared pedestrian and bicycle path', 'circle', 'blue_white', 'white_border', 'mandatory_direction', 'shared_ped_cycle'),
    'D1a': ('Mandatory direction: Straight ahead', 'circle', 'blue_white', 'white_border', 'mandatory_direction', 'straight'),
    'D1b': ('Mandatory direction: Turn left or right', 'circle', 'blue_white', 'white_border', 'mandatory_direction', 'turn_direction'),
    'D3b': ('Pass this side: Keep right or left', 'circle', 'blue_white', 'white_border', 'mandatory_direction', 'pass_side'),
    'D5': ('Compulsory roundabout', 'circle', 'blue_white', 'white_border', 'mandatory_direction', 'roundabout'),
    'D7': ('Compulsory cycle path', 'circle', 'blue_white', 'white_border', 'mandatory_direction', 'cycle_path'),
    'D9': ('Compulsory pedestrian path', 'circle', 'blue_white', 'white_border', 'mandatory_direction', 'pedestrian_path'),
    'E1': ('No parking', 'circle', 'blue_white', 'red_border', 'prohibition_restriction', 'no_parking'),
    'E3': ('No parking and no standing', 'circle', 'blue_white', 'red_border', 'prohibition_restriction', 'no_standing'),
    'E5': ('Parking zone', 'rectangle', 'blue_white', 'white_border', 'none', 'parking_zone'),
    'E7': ('Reserved parking area', 'rectangle', 'blue_white', 'white_border', 'none', 'parking_reserved'),
    'B21': ('Narrow passage: Priority over oncoming vehicles', 'rectangle', 'blue_white', 'white_border', 'priority_geometry', 'priority_over_oncoming'),
    'E9a': ('Parking for passenger cars', 'rectangle', 'blue_white', 'white_border', 'none', 'parking_cars'),
    'E9a_miva': ('Parking for disabled persons', 'rectangle', 'blue_white', 'white_border', 'none', 'parking_disabled'),
    'E9b': ('Parking for goods vehicles / trucks', 'rectangle', 'blue_white', 'white_border', 'none', 'parking_trucks'),
    'E9c': ('Parking for buses', 'rectangle', 'blue_white', 'white_border', 'none', 'parking_buses'),
    'E9d': ('Parking partly on sidewalk', 'rectangle', 'blue_white', 'white_border', 'none', 'parking_sidewalk'),
    'E9e': ('Parking completely on sidewalk', 'rectangle', 'blue_white', 'white_border', 'none', 'parking_full_sidewalk'),
    'F12a': ('Residential area (Woonerf) beginning', 'rectangle', 'blue_white', 'white_border', 'none', 'residential_begin'),
    'F12b': ('Residential area (Woonerf) end', 'rectangle', 'blue_white', 'red_border', 'deregulation_cancel', 'residential_end'),
    'F19': ('One-way traffic street', 'rectangle', 'blue_white', 'white_border', 'mandatory_direction', 'one_way'),
    'F45': ('No through road / Dead end', 'rectangle', 'blue_white', 'white_border', 'none', 'dead_end'),
    'F47': ('Hospital / First aid', 'rectangle', 'blue_white', 'white_border', 'none', 'hospital'),
    'F49': ('Emergency repair station / Breakdown service', 'rectangle', 'blue_white', 'white_border', 'none', 'breakdown_service'),
    'F50': ('Public telephone', 'rectangle', 'blue_white', 'white_border', 'none', 'telephone'),
    'F59': ('Camping site', 'rectangle', 'blue_white', 'white_border', 'none', 'camping'),
    'F87': ('Lane configuration / Expressway exit', 'rectangle', 'blue_white', 'white_border', 'none', 'lane_exit'),
    'B11': ('End of priority road', 'diamond', 'white_black', 'black_border', 'deregulation_cancel', 'end_priority'),
    'B9': ('Priority road', 'diamond', 'yellow_white', 'white_border', 'priority_geometry', 'priority_road'),
}


def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def preprocess_and_save_image(
    src_img_path: Path,
    dest_img_path: Path,
    roi: tuple[int, int, int, int],
    target_size: int = 64,
    margin_ratio: float = 0.10,
) -> None:
    """Crops to ROI + 10% margin, proportionally letterboxes, and zero-pads to target_size square."""
    with Image.open(src_img_path) as img:
        img = img.convert("RGB")
        w_img, h_img = img.size

        x1, y1, x2, y2 = roi
        roi_w = x2 - x1 + 1
        roi_h = y2 - y1 + 1

        margin_x = int(round(margin_ratio * roi_w))
        margin_y = int(round(margin_ratio * roi_h))

        crop_x1 = max(0, x1 - margin_x)
        crop_y1 = max(0, y1 - margin_y)
        crop_x2 = min(w_img - 1, x2 + margin_x)
        crop_y2 = min(h_img - 1, y2 + margin_y)

        cropped = img.crop((crop_x1, crop_y1, crop_x2 + 1, crop_y2 + 1))
        crop_w, crop_h = cropped.size

        scale = min(target_size / crop_w, target_size / crop_h)
        new_w = max(1, int(round(crop_w * scale)))
        new_h = max(1, int(round(crop_h * scale)))

        resized = cropped.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)

        canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
        pad_x = (target_size - new_w) // 2
        pad_y = (target_size - new_h) // 2
        canvas.paste(resized, (pad_x, pad_y))

        dest_img_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(dest_img_path, format="PNG", optimize=True)


def stratified_group_train_val_split(
    train_df: pd.DataFrame, train_ratio: float = 0.80, seed: int = 42
) -> pd.DataFrame:
    """Splits training physical poles into train and validation sets per class.
    Guarantees ZERO physical pole overlap between train and validation.
    """
    rng = np.random.default_rng(seed)
    train_df = train_df.copy()
    train_df["split"] = ""

    for class_id, group in train_df.groupby("ClassId"):
        unique_poles = group["pole_id"].unique()
        shuffled_poles = rng.permutation(unique_poles)
        n_poles = len(shuffled_poles)

        if n_poles > 1:
            n_train_poles = int(round(train_ratio * n_poles))
            n_train_poles = min(max(1, n_train_poles), n_poles - 1)
        else:
            n_train_poles = 1

        train_poles = set(shuffled_poles[:n_train_poles])
        val_poles = set(shuffled_poles[n_train_poles:])

        is_train = group["pole_id"].isin(train_poles)
        is_val = group["pole_id"].isin(val_poles)

        train_df.loc[group[is_train].index, "split"] = "train"
        train_df.loc[group[is_val].index, "split"] = "val"

    return train_df


def build_concepts_table(reduced_set_path: Path, out_path: Path) -> pd.DataFrame:
    """Generates the standardized 62-class concept table for Belgium."""
    with open(reduced_set_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    class_codes = lines[1:]

    records = []
    for c_id, code in enumerate(class_codes):
        info = BELGIAN_SIGN_CATALOG.get(code, (f"Sign {code}", "circle", "white_red", "red_border", "none", "unknown"))
        records.append({
            "class_id": c_id,
            "sign_code": code,
            "class_name": info[0],
            "shape": info[1],
            "colour_profile": info[2],
            "border_type": info[3],
            "symbol_category": info[4],
            "symbol_content": info[5],
            "vienna_convention_category": code[0] if code[0] in ['A', 'B', 'C', 'D', 'E', 'F'] else 'Unknown',
            "belgian_code_reference": f"Belgian Traffic Code Art. 65 ({code})"
        })

    concepts_df = pd.DataFrame(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    concepts_df.to_csv(out_path, index=False)
    print(f"Saved concept table: {out_path} ({len(concepts_df)} classes)")
    return concepts_df


def main():
    parser = argparse.ArgumentParser(description="Construct BelgiumTSC Master Dataset (data_v2).")
    parser.add_argument("--belgium-root", type=str, default="belgium", help="Path to raw belgium folder")
    parser.add_argument("--output-dir", type=str, default="data/data_v2", help="Output directory for data_v2")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible train/val group split")
    parser.add_argument("--metadata-only", action="store_true", help="Generate manifests only without rendering images")
    args = parser.parse_args()

    belgium_root = Path(args.belgium_root).resolve()
    out_dir = Path(args.output_dir).resolve()
    meta_dir = out_dir / "metadata"
    img_dir = out_dir / "images"
    cfg_dir = out_dir / "configs"
    cpt_dir = out_dir / "concepts"

    meta_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cpt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CONSTRUCTING BELGIUM MASTER DATASET (data_v2)")
    print(f"Raw Source  : {belgium_root}")
    print(f"Output Root : {out_dir}")
    print(f"Master Seed : {args.seed}")
    print(f"Render Imgs : {'No (metadata only)' if args.metadata_only else 'Yes (64x64 letterboxed)'}")
    print("=" * 80)

    # 1. Build and save concept table
    reduced_path = belgium_root / "Seqs_poses_annotations (1)" / "reducedSetTS.txt"
    concept_csv = cpt_dir / "class_concepts.csv"
    concepts_df = build_concepts_table(reduced_path, concept_csv)
    concept_map = concepts_df.set_index("class_id").to_dict("index")

    # 2. Load raw Training manifests
    train_dir = belgium_root / "BelgiumTSC_Training" / "Training"
    train_csvs = sorted(glob.glob(str(train_dir / "*" / "GT-*.csv")))
    print(f"Found {len(train_csvs)} training class manifests.")

    train_rows = []
    for p in train_csvs:
        p_path = Path(p)
        class_folder = p_path.parent.name
        df = pd.read_csv(p_path, sep=";")
        df["class_folder"] = class_folder
        extracted = df["Filename"].str.extract(r"^(\d+)_(\d+)\.ppm")
        df["pole_id"] = extracted[0]
        df["view_id"] = extracted[1]
        df["source_relative_path"] = df.apply(
            lambda r: f"BelgiumTSC_Training/Training/{class_folder}/{r['Filename']}", axis=1
        )
        train_rows.append(df)

    raw_train_df = pd.concat(train_rows, ignore_index=True)
    initial_train_count = len(raw_train_df)

    # Deduplicate verbatim duplicate entries from raw manifests (audit standard)
    raw_train_df = raw_train_df.drop_duplicates(
        subset=["source_relative_path", "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2"]
    ).reset_index(drop=True)
    removed_dups = initial_train_count - len(raw_train_df)
    print(f"Loaded {initial_train_count:,} raw training rows. Removed {removed_dups} verbatim duplicate annotations.")
    print(f"Clean unique training samples: {len(raw_train_df):,} across {raw_train_df['ClassId'].nunique()} classes.")

    # 3. Stratified Group Train/Val Split by Pole ID
    train_split_df = stratified_group_train_val_split(raw_train_df, train_ratio=0.80, seed=args.seed)
    n_train = (train_split_df["split"] == "train").sum()
    n_val = (train_split_df["split"] == "val").sum()
    print(f"Pole-stratified split -> Train: {n_train:,} | Validation: {n_val:,} (0 pole leakage)")

    # 4. Load raw Testing manifests
    test_dir = belgium_root / "BelgiumTSC_Testing" / "Testing"
    test_csvs = sorted(glob.glob(str(test_dir / "*" / "GT-*.csv")))
    print(f"Found {len(test_csvs)} testing class manifests.")

    test_rows = []
    for p in test_csvs:
        p_path = Path(p)
        class_folder = p_path.parent.name
        df = pd.read_csv(p_path, sep=";")
        df["class_folder"] = class_folder
        extracted = df["Filename"].str.extract(r"^(\d+)_(\d+)\.ppm")
        df["pole_id"] = extracted[0]
        df["view_id"] = extracted[1]
        df["source_relative_path"] = df.apply(
            lambda r: f"BelgiumTSC_Testing/Testing/{class_folder}/{r['Filename']}", axis=1
        )
        df["split"] = "test"
        test_rows.append(df)

    raw_test_df = pd.concat(test_rows, ignore_index=True)
    print(f"Loaded {len(raw_test_df):,} raw held-out testing samples across {raw_test_df['ClassId'].nunique()} classes.")

    # 5. Build Standardized Metadata Records
    master_records: list[dict[str, Any]] = []

    # Process Train & Val
    for idx, row in train_split_df.iterrows():
        c_id = int(row["ClassId"])
        pole_id = row["pole_id"]
        view_id = row["view_id"]
        split = row["split"]
        c_info = concept_map.get(c_id, {})

        image_id = f"{split}_c{c_id:02d}_p{pole_id}_v{view_id}"
        canonical_rel = f"images/{split}/{image_id}.png"

        master_records.append({
            "image_id": image_id,
            "source_split": "train",
            "split": split,
            "class_id": c_id,
            "class_name": c_info.get("class_name", ""),
            "group_id": f"c{c_id:02d}_p{pole_id}",
            "source_relative_path": row["source_relative_path"],
            "canonical_image_path": canonical_rel,
            "shape": c_info.get("shape", ""),
            "colour_profile": c_info.get("colour_profile", ""),
            "border_type": c_info.get("border_type", ""),
            "symbol_category": c_info.get("symbol_category", ""),
            "symbol_content": c_info.get("symbol_content", ""),
            "roi_x1": int(row["Roi.X1"]),
            "roi_y1": int(row["Roi.Y1"]),
            "roi_x2": int(row["Roi.X2"]),
            "roi_y2": int(row["Roi.Y2"]),
            "roi_coordinate_convention": "inclusive_xyxy_zero_indexed",
            "orig_width": int(row["Width"]),
            "orig_height": int(row["Height"]),
            "preprocessing_version": "v1_roi_margin10_letterbox64",
        })

    # Process Test
    for idx, row in raw_test_df.iterrows():
        c_id = int(row["ClassId"])
        pole_id = row["pole_id"]
        view_id = row["view_id"]
        split = "test"
        c_info = concept_map.get(c_id, {})

        image_id = f"test_c{c_id:02d}_p{pole_id}_v{view_id}_{idx:05d}"
        canonical_rel = f"images/test/{image_id}.png"

        master_records.append({
            "image_id": image_id,
            "source_split": "test",
            "split": split,
            "class_id": c_id,
            "class_name": c_info.get("class_name", ""),
            "group_id": f"c{c_id:02d}_p{pole_id}",
            "source_relative_path": row["source_relative_path"],
            "canonical_image_path": canonical_rel,
            "shape": c_info.get("shape", ""),
            "colour_profile": c_info.get("colour_profile", ""),
            "border_type": c_info.get("border_type", ""),
            "symbol_category": c_info.get("symbol_category", ""),
            "symbol_content": c_info.get("symbol_content", ""),
            "roi_x1": int(row["Roi.X1"]),
            "roi_y1": int(row["Roi.Y1"]),
            "roi_x2": int(row["Roi.X2"]),
            "roi_y2": int(row["Roi.Y2"]),
            "roi_coordinate_convention": "inclusive_xyxy_zero_indexed",
            "orig_width": int(row["Width"]),
            "orig_height": int(row["Height"]),
            "preprocessing_version": "v1_roi_margin10_letterbox64",
        })

    master_df = pd.DataFrame(master_records)

    # 6. Save Manifest CSVs
    train_manifest = meta_dir / "train.csv"
    val_manifest = meta_dir / "val.csv"
    test_manifest = meta_dir / "test.csv"
    master_manifest = meta_dir / "master_metadata.csv"

    master_df[master_df["split"] == "train"].to_csv(train_manifest, index=False)
    master_df[master_df["split"] == "val"].to_csv(val_manifest, index=False)
    master_df[master_df["split"] == "test"].to_csv(test_manifest, index=False)
    master_df.to_csv(master_manifest, index=False)

    print("\nSaved metadata manifests:")
    print(f"  - train.csv           : {len(master_df[master_df['split'] == 'train']):,} rows")
    print(f"  - val.csv             : {len(master_df[master_df['split'] == 'val']):,} rows")
    print(f"  - test.csv            : {len(master_df[master_df['split'] == 'test']):,} rows")
    print(f"  - master_metadata.csv : {len(master_df):,} rows")

    # 7. Render 64x64 Letterboxed Canonical PNG Images
    if not args.metadata_only:
        print("\nRendering canonical 64x64 letterboxed images...")
        pbar = tqdm(master_df.iterrows(), total=len(master_df), desc="Rendering 64x64 PNGs", unit="img")
        for _, r in pbar:
            src_p = belgium_root / r["source_relative_path"]
            dst_p = out_dir / r["canonical_image_path"]
            roi = (r["roi_x1"], r["roi_y1"], r["roi_x2"], r["roi_y2"])
            preprocess_and_save_image(src_p, dst_p, roi, target_size=64, margin_ratio=0.10)
        print("Canonical image generation complete!")

    # 8. Compute Checksums & Write configs
    train_hash = compute_sha256(train_manifest)
    val_hash = compute_sha256(val_manifest)
    test_hash = compute_sha256(test_manifest)
    master_hash = compute_sha256(master_manifest)
    concept_hash = compute_sha256(concept_csv)

    now_utc = datetime.now(timezone.utc).isoformat()
    version_md_content = f"""# BelgiumTSC Master Dataset Specification and Version Record (data_v2)

- **Creation Timestamp (UTC):** {now_utc}
- **Raw Data Source:** `{belgium_root}`
- **Master Random Seed:** `{args.seed}`
- **Split Protocol:** Stratified Group Split by `(ClassId + PoleId)`
- **Preprocessing Version:** `v1_roi_margin10_letterbox64`
  - ROI Crop: 10% contextual margin on all sides (clamped to image dimensions).
  - Resizing: Proportional bilinear scaling preserving aspect ratio.
  - Padding: Symmetric zero-padding to 64x64 black canvas.

## Dataset Partition Summary

| Split | Number of Images | Description |
| :--- | :--- | :--- |
| **Train** | `{len(master_df[master_df['split'] == 'train']):,}` | 80% of physical poles per class for training |
| **Validation** | `{len(master_df[master_df['split'] == 'val']):,}` | 20% of physical poles per class (zero physical pole leakage) |
| **Test** | `{len(master_df[master_df['split'] == 'test']):,}` | Official BelgiumTSC held-out test benchmark (strictly untouched) |
| **Total** | `{len(master_df):,}` | Total annotated samples across 62 classes |

## Manifest SHA-256 Checksums

```text
train.csv:           {train_hash}
val.csv:             {val_hash}
test.csv:            {test_hash}
master_metadata.csv: {master_hash}
class_concepts.csv:  {concept_hash}
```
"""
    with open(cfg_dir / "dataset_version.md", "w", encoding="utf-8") as f:
        f.write(version_md_content)

    preprocessing_yaml_content = f"""# Canonical Preprocessing Configuration (Version 2.0 - BelgiumTSC)
# Frozen scientific contract matching German GTSRB master_v1.

dataset_name: "BelgiumTSC (Belgium Traffic Sign Classification Benchmark)"
preprocessing_version: "v1_roi_margin10_letterbox64"

input_resolution:
  target_height: 64
  target_width: 64
  channels: 3
  color_space: "RGB"

roi_crop:
  enabled: true
  coordinate_convention: "inclusive_xyxy_zero_indexed"
  margin_ratio: 0.10
  clamp_to_image_bounds: true

scaling_and_aspect_ratio:
  policy: "proportional_letterbox"
  interpolation: "bilinear"
  scale_factor_rule: "min(target_w / crop_w, target_h / crop_h)"

padding:
  policy: "symmetric_zero_padding"
  fill_value: [0, 0, 0]
  alignment: "center"

splits:
  source_dataset: "BelgiumTSC (KUL VISICS / ETH Zurich)"
  train_split_policy: "stratified_group_kfold_by_pole"
  train_ratio: 0.80
  val_ratio: 0.20
  split_seed: {args.seed}
  grouping_key: "class_id + '_' + pole_id"
  test_split_policy: "held_out_official_belgiumtsc_test"
"""
    with open(cfg_dir / "preprocessing.yaml", "w", encoding="utf-8") as f:
        f.write(preprocessing_yaml_content)

    # Write ontology.md
    ontology_md = """# Belgian Traffic Sign Physical Concept Ontology (data_v2)

Grounded in the **1968 UNECE Vienna Convention on Road Signs and Signals** and the **Belgian Highway Code (Code de la route / Wegcode)**:

## 1. Shape (6 dimensions)
- `circle`: Regulatory prohibitory (Series C) and mandatory (Series D) signs.
- `triangle_up`: Warning/danger signs (Series A).
- `triangle_down`: Yield / Give way sign (B1).
- `octagon`: Stop sign (B5).
- `diamond`: Priority road signs (B9, B11).
- `rectangle`: Informational / Indication (Series F) and Parking (Series E) signs.

## 2. Colour Profile (5 dimensions)
- `white_red`: White background with red border / red accents.
- `blue_white`: Blue background with white pictograms / white border.
- `yellow_white`: Yellow diamond with white border (Priority road B9).
- `white_black`: White background with black diagonal cancellation (B11).
- `red_white`: Red background with white elements (B5 Stop, C1 No Entry).

## 3. Border Type (4 dimensions)
- `red_border`: Standard warning and prohibitory outer band.
- `white_border`: Inner border for mandatory blue discs, Stop sign, and Priority road.
- `black_border`: Diagonal cancellation stripe.
- `none`: Borderless signs.

## 4. Symbol Category (7 dimensions)
- `warning_hazard`: Hazard warnings (curves, pedestrians, children, road works).
- `priority_geometry`: Right-of-way regulations (Yield, Stop, Priority Road).
- `prohibition_restriction`: Restrictions (No entry, No passing, No trucks, weight limits).
- `speed_numeral`: Speed limits (C43).
- `mandatory_direction`: Obligatory maneuvers (turn directions, roundabout).
- `deregulation_cancel`: End of restriction / End of residential area (B11, F12b).
- `none`: General informational and parking signs.
"""
    with open(cpt_dir / "ontology.md", "w", encoding="utf-8") as f:
        f.write(ontology_md)

    print("\nDataset generation completed successfully!")
    print(f"Data saved to: {out_dir}")


if __name__ == "__main__":
    main()
