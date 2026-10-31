"""Build Chinese TSRD Master Dataset (data_v3)
===========================================
Strictly implements the Master Dataset specification from GTSRB (master_v1)
and BelgiumTSC (data_v2):
1. Anti-leakage: Stratified group train/val split (80/20) by (class_id + instance_id)
   guaranteeing ZERO scene/image-twin leakage between train and validation.
2. Official held-out test split: Preserves TSRD-Test (1,994 images) strictly untouched.
3. 4-Stage Preprocessing Contract:
   - Context-buffered ROI crop with 10% margin (clamped to image bounds).
   - Proportional bilinear letterbox resize to max dimension 64 (aspect-ratio preserved).
   - Symmetric center zero-padding on a 64x64 black canvas.
4. Schema-complete metadata manifests in data/data_v3/metadata/:
   - train.csv, val.csv, test.csv, master_metadata.csv.
5. Standardized 64x64 PNG images rendered to data/data_v3/images/.
6. Legally grounded concept ontology mapping in data/data_v3/concepts/class_concepts.csv
   aligned with Chinese National Standard GB 5768 and the 1968 Vienna Convention.
7. Cryptographic SHA-256 manifests record in data/data_v3/configs/dataset_version.md.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

CHINESE_SIGN_CATALOG: dict[int, tuple[str, str, str, str, str, str, str]] = {
    0: ('Speed limit (5km/h)', '限速5km/h', 'circle', 'white_red', 'red_border', 'speed_numeral', 'speed_5'),
    1: ('Speed limit (15km/h)', '限速15km/h', 'circle', 'white_red', 'red_border', 'speed_numeral', 'speed_15'),
    2: ('Speed limit (30km/h)', '限速30km/h', 'circle', 'white_red', 'red_border', 'speed_numeral', 'speed_30'),
    3: ('Speed limit (40km/h)', '限速40km/h', 'circle', 'white_red', 'red_border', 'speed_numeral', 'speed_40'),
    4: ('Speed limit (50km/h)', '限速50km/h', 'circle', 'white_red', 'red_border', 'speed_numeral', 'speed_50'),
    5: ('Speed limit (60km/h)', '限速60km/h', 'circle', 'white_red', 'red_border', 'speed_numeral', 'speed_60'),
    6: ('Speed limit (70km/h)', '限速70km/h', 'circle', 'white_red', 'red_border', 'speed_numeral', 'speed_70'),
    7: ('Speed limit (80km/h)', '限速80km/h', 'circle', 'white_red', 'red_border', 'speed_numeral', 'speed_80'),
    8: ('No straight ahead and no left turn', '禁止左转和直行', 'circle', 'white_red', 'red_border', 'prohibition_restriction', 'no_straight_left'),
    9: ('No straight ahead and no right turn', '禁止右转和直行', 'circle', 'white_red', 'red_border', 'prohibition_restriction', 'no_straight_right'),
    10: ('No straight ahead', '禁止直行', 'circle', 'white_red', 'red_border', 'prohibition_restriction', 'no_straight'),
    11: ('No left turn', '禁止左转', 'circle', 'white_red', 'red_border', 'prohibition_restriction', 'no_left_turn'),
    12: ('No left and right turn', '禁止左转和右转', 'circle', 'white_red', 'red_border', 'prohibition_restriction', 'no_left_right'),
    13: ('No right turn', '禁止右转', 'circle', 'white_red', 'red_border', 'prohibition_restriction', 'no_right_turn'),
    14: ('No passing / overtaking', '禁止超车', 'circle', 'white_red', 'red_border', 'prohibition_restriction', 'no_passing'),
    15: ('End of no passing / overtaking', '解除禁止超车', 'circle', 'white_black', 'black_border', 'deregulation_cancel', 'end_no_passing'),
    16: ('No U-turn', '禁止掉头', 'circle', 'white_red', 'red_border', 'prohibition_restriction', 'no_u_turn'),
    17: ('No motor vehicles', '禁止机动车驶入', 'circle', 'white_red', 'red_border', 'prohibition_restriction', 'no_motor_vehicles'),
    18: ('No honking / Horn prohibition', '禁止鸣喇叭', 'circle', 'white_red', 'red_border', 'prohibition_restriction', 'no_honking'),
    19: ('End of speed limit (40km/h)', '解除限速40km/h', 'circle', 'white_black', 'black_border', 'deregulation_cancel', 'end_speed_40'),
    20: ('End of speed limit (50km/h)', '解除限速50km/h', 'circle', 'white_black', 'black_border', 'deregulation_cancel', 'end_speed_50'),
    21: ('Mandatory straight and right turn', '只许直行和右转', 'circle', 'blue_white', 'white_border', 'mandatory_direction', 'straight_right'),
    22: ('Mandatory straight ahead', '只许直行', 'circle', 'blue_white', 'white_border', 'mandatory_direction', 'straight'),
    23: ('Mandatory left turn', '只许左转', 'circle', 'blue_white', 'white_border', 'mandatory_direction', 'left_turn'),
    24: ('Mandatory left and right turn', '只许左转和右转', 'circle', 'blue_white', 'white_border', 'mandatory_direction', 'left_right_turn'),
    25: ('Mandatory right turn', '只许右转', 'circle', 'blue_white', 'white_border', 'mandatory_direction', 'right_turn'),
    26: ('Keep left', '靠左行驶', 'circle', 'blue_white', 'white_border', 'mandatory_direction', 'keep_left'),
    27: ('Keep right', '靠右行驶', 'circle', 'blue_white', 'white_border', 'mandatory_direction', 'keep_right'),
    28: ('Compulsory roundabout', '环岛行驶', 'circle', 'blue_white', 'white_border', 'mandatory_direction', 'roundabout'),
    29: ('Passenger cars only lane', '只许小型车行驶', 'circle', 'blue_white', 'white_border', 'mandatory_direction', 'passenger_cars'),
    30: ('Horn sound mandatory', '鸣喇叭', 'circle', 'blue_white', 'white_border', 'mandatory_direction', 'sound_horn'),
    31: ('Motor vehicle lane', '机动车车道', 'circle', 'blue_white', 'white_border', 'mandatory_direction', 'motor_vehicles'),
    32: ('Mandatory U-turn', '只许掉头', 'circle', 'blue_white', 'white_border', 'mandatory_direction', 'u_turn'),
    33: ('Pass obstacle on left', '前方障碍物绕行（左侧）', 'circle', 'blue_white', 'white_border', 'mandatory_direction', 'pass_obstacle_left'),
    34: ('Pass obstacle on right', '前方障碍物绕行（右侧）', 'circle', 'blue_white', 'white_border', 'mandatory_direction', 'pass_obstacle_right'),
    35: ('Traffic signals ahead warning', '注意红绿灯', 'triangle_up', 'yellow_white', 'black_border', 'warning_hazard', 'traffic_light'),
    36: ('Slow down / Give way', '减速慢行', 'triangle_down', 'yellow_white', 'black_border', 'priority_geometry', 'slow_down'),
    37: ('Pedestrian crossing warning', '注意行人', 'triangle_up', 'yellow_white', 'black_border', 'warning_hazard', 'pedestrian'),
    38: ('Children / School warning', '注意儿童', 'triangle_up', 'yellow_white', 'black_border', 'warning_hazard', 'children'),
    39: ('Cyclists warning', '注意自行车', 'triangle_up', 'yellow_white', 'black_border', 'warning_hazard', 'cyclist'),
    40: ('School zone warning', '注意学校', 'triangle_up', 'yellow_white', 'black_border', 'warning_hazard', 'school'),
    41: ('Sharp curve left warning', '注意左急转弯', 'triangle_up', 'yellow_white', 'black_border', 'warning_hazard', 'curve_left'),
    42: ('Sharp curve right warning', '注意右急转弯', 'triangle_up', 'yellow_white', 'black_border', 'warning_hazard', 'curve_right'),
    43: ('Continuous curves warning', '注意连续弯路', 'triangle_up', 'yellow_white', 'black_border', 'warning_hazard', 'double_curve'),
    44: ('Steep ascent warning', '注意上陡坡', 'triangle_up', 'yellow_white', 'black_border', 'warning_hazard', 'steep_ascent'),
    45: ('Steep descent warning', '注意下陡坡', 'triangle_up', 'yellow_white', 'black_border', 'warning_hazard', 'steep_descent'),
    46: ('General danger warning', '注意危险', 'triangle_up', 'yellow_white', 'black_border', 'warning_hazard', 'general_danger'),
    47: ('Village ahead warning', '注意村庄', 'triangle_up', 'yellow_white', 'black_border', 'warning_hazard', 'village'),
    48: ('Falling rocks warning', '注意落石', 'triangle_up', 'yellow_white', 'black_border', 'warning_hazard', 'falling_rocks'),
    49: ('Crosswinds warning', '注意侧风', 'triangle_up', 'yellow_white', 'black_border', 'warning_hazard', 'crosswinds'),
    50: ('Traffic congestion warning', '注意交通拥堵', 'triangle_up', 'yellow_white', 'black_border', 'warning_hazard', 'congestion'),
    51: ('Road work / Construction warning', '注意施工', 'triangle_up', 'yellow_white', 'black_border', 'warning_hazard', 'road_work'),
    52: ('Guarded railway crossing warning', '注意铁道路口', 'triangle_up', 'yellow_white', 'black_border', 'warning_hazard', 'railway_crossing'),
    53: ('Unguarded railway crossing warning', '注意无人看守铁道路口', 'triangle_up', 'yellow_white', 'black_border', 'warning_hazard', 'railway_unguarded'),
    54: ('Other hazards warning', '注意其他危险', 'triangle_up', 'yellow_white', 'black_border', 'warning_hazard', 'other_danger'),
    55: ('No entry in both directions', '禁止通行', 'circle', 'white_red', 'red_border', 'prohibition_restriction', 'no_entry'),
    56: ('No parking', '禁止停车', 'circle', 'blue_white', 'red_border', 'prohibition_restriction', 'no_parking'),
    57: ('No stopping / No waiting', '禁止长时停车', 'circle', 'blue_white', 'red_border', 'prohibition_restriction', 'no_standing'),
}


def get_base_instance(fn: str) -> str:
    """Extracts standard 4-digit instance identifier from TSRD filename.
    Examples:
      '000_0001.png'      -> '0001'
      '000_1_0001.png'    -> '0001'
      '008_0005_j.png'    -> '0005'
      '008_1_0005_1_j.png'-> '0005'
    """
    clean = re.sub(r'(_j)?\.png$', '', fn)
    parts = clean.split('_')
    if len(parts) == 2:
        return parts[1]
    elif len(parts) == 3:
        return parts[2]
    elif len(parts) == 4:
        return parts[2]
    else:
        return parts[-1]


def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def preprocess_image_bytes(
    img_bytes: bytes,
    roi: tuple[int, int, int, int],
    target_size: int = 64,
    margin_ratio: float = 0.10,
) -> Image.Image:
    """Crops to ROI + 10% margin, proportionally letterboxes, and zero-pads to target_size square."""
    with Image.open(io.BytesIO(img_bytes)) as img:
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
        return canvas


def stratified_group_train_val_split(
    train_df: pd.DataFrame, train_ratio: float = 0.80, seed: int = 42
) -> pd.DataFrame:
    """Splits training sign instances into train and validation sets per class.
    Guarantees ZERO instance/twin overlap between train and validation.
    """
    rng = np.random.default_rng(seed)
    train_df = train_df.copy()
    train_df["split"] = ""

    for class_id, group in train_df.groupby("ClassId"):
        unique_groups = group["group_id"].unique()
        shuffled_groups = rng.permutation(unique_groups)
        n_groups = len(shuffled_groups)

        if n_groups > 1:
            n_train_groups = int(round(train_ratio * n_groups))
            n_train_groups = min(max(1, n_train_groups), n_groups - 1)
        else:
            n_train_groups = 1

        train_groups = set(shuffled_groups[:n_train_groups])
        val_groups = set(shuffled_groups[n_train_groups:])

        is_train = group["group_id"].isin(train_groups)
        is_val = group["group_id"].isin(val_groups)

        train_df.loc[group[is_train].index, "split"] = "train"
        train_df.loc[group[is_val].index, "split"] = "val"

    return train_df


def build_concepts_table(out_path: Path) -> pd.DataFrame:
    """Generates the standardized 58-class concept table for Chinese TSRD."""
    records = []
    for c_id in range(58):
        info = CHINESE_SIGN_CATALOG[c_id]
        records.append({
            "class_id": c_id,
            "chinese_name": info[1],
            "class_name": info[0],
            "shape": info[2],
            "colour_profile": info[3],
            "border_type": info[4],
            "symbol_category": info[5],
            "symbol_content": info[6],
            "national_standard": "GB 5768-2009 (Road Traffic Signs and Markings)",
            "vienna_convention_category": "Prohibitory (C)" if info[5] in ["speed_numeral", "prohibition_restriction", "deregulation_cancel"] else ("Mandatory (D)" if info[5] == "mandatory_direction" else "Warning (A)"),
        })

    concepts_df = pd.DataFrame(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    concepts_df.to_csv(out_path, index=False)
    print(f"Saved concept table: {out_path} ({len(concepts_df)} classes)")
    return concepts_df


def main():
    parser = argparse.ArgumentParser(description="Construct Chinese TSRD Master Dataset (data_v3).")
    parser.add_argument("--chinese-root", type=str, default="chinese", help="Path to raw chinese folder")
    parser.add_argument("--output-dir", type=str, default="data/data_v3", help="Output directory for data_v3")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible train/val group split")
    parser.add_argument("--metadata-only", action="store_true", help="Generate manifests only without rendering images")
    args = parser.parse_args()

    chinese_root = Path(args.chinese_root).resolve()
    out_dir = Path(args.output_dir).resolve()
    meta_dir = out_dir / "metadata"
    img_dir = out_dir / "images"
    cfg_dir = out_dir / "configs"
    cpt_dir = out_dir / "concepts"

    meta_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cpt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CONSTRUCTING CHINESE TSRD MASTER DATASET (data_v3)")
    print(f"Raw Source  : {chinese_root}")
    print(f"Output Root : {out_dir}")
    print(f"Master Seed : {args.seed}")
    print(f"Render Imgs : {'No (metadata only)' if args.metadata_only else 'Yes (64x64 letterboxed)'}")
    print("=" * 80)

    # 1. Build and save concept table
    concept_csv = cpt_dir / "class_concepts.csv"
    concepts_df = build_concepts_table(concept_csv)
    concept_map = concepts_df.set_index("class_id").to_dict("index")

    # 2. Parse Raw Training Annotations
    train_annot_zip = chinese_root / "TSRD-Train Annotation.zip"
    with zipfile.ZipFile(train_annot_zip) as z:
        with z.open("TsignRecgTrain4170Annotation.txt") as f:
            train_lines = [l.decode("utf-8", errors="replace").strip() for l in f if l.strip()]

    train_records = []
    for line in train_lines:
        parts = [p.strip() for p in line.split(";") if p.strip()]
        fname = parts[0]
        cid = int(parts[7])
        inst = get_base_instance(fname)
        train_records.append({
            "Filename": fname,
            "Width": int(parts[1]),
            "Height": int(parts[2]),
            "Roi.X1": int(parts[3]),
            "Roi.Y1": int(parts[4]),
            "Roi.X2": int(parts[5]),
            "Roi.Y2": int(parts[6]),
            "ClassId": cid,
            "group_id": f"c{cid:02d}_inst{inst}",
            "source_relative_path": f"tsrd-train.zip/{fname}",
        })

    raw_train_df = pd.DataFrame(train_records)
    print(f"Loaded {len(raw_train_df):,} raw training rows across {raw_train_df['ClassId'].nunique()} classes.")
    print(f"Identified {raw_train_df['group_id'].nunique():,} unique sign instance groups.")

    # 3. Stratified Group Split (80% Train, 20% Val by Instance Group)
    train_split_df = stratified_group_train_val_split(raw_train_df, train_ratio=0.80, seed=args.seed)
    n_train = (train_split_df["split"] == "train").sum()
    n_val = (train_split_df["split"] == "val").sum()
    print(f"Instance-stratified split -> Train: {n_train:,} | Validation: {n_val:,} (0 instance leakage)")

    # 4. Parse Raw Testing Annotations
    test_annot_zip = chinese_root / "TSRD-Test Annotation.zip"
    with zipfile.ZipFile(test_annot_zip) as z:
        with z.open("TsignRecgTest1994Annotation.txt") as f:
            test_lines = [l.decode("utf-8", errors="replace").strip() for l in f if l.strip()]

    test_records = []
    for line in test_lines:
        parts = [p.strip() for p in line.split(";") if p.strip()]
        fname = parts[0]
        cid = int(parts[7])
        inst = get_base_instance(fname)
        test_records.append({
            "Filename": fname,
            "Width": int(parts[1]),
            "Height": int(parts[2]),
            "Roi.X1": int(parts[3]),
            "Roi.Y1": int(parts[4]),
            "Roi.X2": int(parts[5]),
            "Roi.Y2": int(parts[6]),
            "ClassId": cid,
            "group_id": f"c{cid:02d}_inst{inst}",
            "source_relative_path": f"TSRD-Test.zip/{fname}",
            "split": "test",
        })

    raw_test_df = pd.DataFrame(test_records)
    print(f"Loaded {len(raw_test_df):,} raw held-out testing samples across {raw_test_df['ClassId'].nunique()} classes.")

    # 5. Build Standardized Metadata Records
    master_records: list[dict[str, Any]] = []

    # Process Train & Val
    for _, row in train_split_df.iterrows():
        c_id = int(row["ClassId"])
        fname_stem = Path(row["Filename"]).stem
        split = row["split"]
        c_info = concept_map.get(c_id, {})

        image_id = f"{split}_c{c_id:02d}_{fname_stem}"
        canonical_rel = f"images/{split}/{image_id}.png"

        master_records.append({
            "image_id": image_id,
            "source_split": "train",
            "split": split,
            "class_id": c_id,
            "class_name": c_info.get("class_name", ""),
            "group_id": row["group_id"],
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
    for _, row in raw_test_df.iterrows():
        c_id = int(row["ClassId"])
        fname_stem = Path(row["Filename"]).stem
        split = "test"
        c_info = concept_map.get(c_id, {})

        image_id = f"test_c{c_id:02d}_{fname_stem}"
        canonical_rel = f"images/test/{image_id}.png"

        master_records.append({
            "image_id": image_id,
            "source_split": "test",
            "split": split,
            "class_id": c_id,
            "class_name": c_info.get("class_name", ""),
            "group_id": row["group_id"],
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
        print("\nRendering canonical 64x64 letterboxed images directly from zip archives...")
        train_zip_path = chinese_root / "tsrd-train.zip"
        test_zip_path = chinese_root / "TSRD-Test.zip"

        with zipfile.ZipFile(train_zip_path) as train_z, zipfile.ZipFile(test_zip_path) as test_z:
            pbar = tqdm(master_df.iterrows(), total=len(master_df), desc="Rendering 64x64 PNGs", unit="img")
            for _, r in pbar:
                split = r["split"]
                src_rel = r["source_relative_path"]
                fname = src_rel.split("/")[-1]
                z = test_z if split == "test" else train_z
                img_bytes = z.read(fname)
                roi = (r["roi_x1"], r["roi_y1"], r["roi_x2"], r["roi_y2"])
                canvas = preprocess_image_bytes(img_bytes, roi, target_size=64, margin_ratio=0.10)
                dst_p = out_dir / r["canonical_image_path"]
                dst_p.parent.mkdir(parents=True, exist_ok=True)
                canvas.save(dst_p, format="PNG", optimize=True)

        print("Canonical image generation complete!")

    # 8. Compute Checksums & Write configs
    train_hash = compute_sha256(train_manifest)
    val_hash = compute_sha256(val_manifest)
    test_hash = compute_sha256(test_manifest)
    master_hash = compute_sha256(master_manifest)
    concept_hash = compute_sha256(concept_csv)

    now_utc = datetime.now(timezone.utc).isoformat()
    version_md_content = f"""# Chinese TSRD Master Dataset Specification and Version Record (data_v3)

- **Creation Timestamp (UTC):** {now_utc}
- **Raw Data Source:** `{chinese_root}`
- **Master Random Seed:** `{args.seed}`
- **Split Protocol:** Stratified Group Split by `(ClassId + InstanceId)`
- **Preprocessing Version:** `v1_roi_margin10_letterbox64`
  - ROI Crop: 10% contextual margin on all sides (clamped to image dimensions).
  - Resizing: Proportional bilinear scaling preserving aspect ratio.
  - Padding: Symmetric zero-padding to 64x64 black canvas.

## Dataset Partition Summary

| Split | Number of Images | Description |
| :--- | :--- | :--- |
| **Train** | `{len(master_df[master_df['split'] == 'train']):,}` | 80% of sign instance groups per class for training |
| **Validation** | `{len(master_df[master_df['split'] == 'val']):,}` | 20% of sign instance groups per class (zero instance/twin leakage) |
| **Test** | `{len(master_df[master_df['split'] == 'test']):,}` | Official TSRD held-out test benchmark (strictly untouched) |
| **Total** | `{len(master_df):,}` | Total annotated samples across 58 classes |

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

    preprocessing_yaml_content = f"""# Canonical Preprocessing Configuration (Version 3.0 - Chinese TSRD)
# Frozen scientific contract matching German GTSRB master_v1 and BelgiumTSC data_v2.

dataset_name: "TSRD (Chinese Traffic Sign Recognition Database)"
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
  source_dataset: "TSRD (CASIA NLPR / Tsinghua University)"
  train_split_policy: "stratified_group_by_instance"
  train_ratio: 0.80
  val_ratio: 0.20
  split_seed: {args.seed}
  grouping_key: "class_id + '_' + instance_id"
  test_split_policy: "held_out_official_tsrd_test"
"""
    with open(cfg_dir / "preprocessing.yaml", "w", encoding="utf-8") as f:
        f.write(preprocessing_yaml_content)

    ontology_md = """# Chinese Traffic Sign Physical Concept Ontology (data_v3)

Grounded in **National Standard of P.R. China GB 5768-2009 (Road Traffic Signs and Markings)** and the **1968 UNECE Vienna Convention on Road Signs and Signals**:

## 1. Shape (4 dimensions)
- `circle`: Regulatory prohibitory (禁令标志) and mandatory/directional (指示标志) signs.
- `triangle_up`: Warning / hazard alert signs (警告标志).
- `triangle_down`: Slow down / Give way (减速慢行 - Class 36).
- `rectangle`: Auxiliary and directional indication signs.

## 2. Colour Profile (5 dimensions)
- `white_red`: White background with red circular outer border and red diagonal slash (Prohibition).
- `blue_white`: Blue background with white arrows / pictograms (Mandatory directions).
- `yellow_white`: Yellow background with black border / black pictograms (Hazard warnings).
- `white_black`: White background with black cancellation slashes (End of prohibition / speed limit).
- `red_white`: Red background with white elements.

## 3. Border Type (3 dimensions)
- `red_border`: Standard prohibition circular rim.
- `white_border`: Inner rim for mandatory blue discs.
- `black_border`: Triangular warning outer border and cancellation slashes.

## 4. Symbol Category (5 dimensions)
- `speed_numeral`: Speed restrictions (Classes 0-7: 5 to 80 km/h).
- `prohibition_restriction`: Turning, overtaking, entry, honking, and parking prohibitions (Classes 8-14, 16-18, 55-57).
- `mandatory_direction`: Compulsory maneuvers, lane usages, and roundabout (Classes 21-34).
- `warning_hazard`: Road geometries, pedestrian/school crossings, railway intersections, and hazards (Classes 35, 37-54).
- `priority_geometry`: Right-of-way regulations (Class 36 Slow down / Yield).
- `deregulation_cancel`: End of restrictions (Classes 15, 19, 20).
"""
    with open(cpt_dir / "ontology.md", "w", encoding="utf-8") as f:
        f.write(ontology_md)

    print("\nChinese TSRD Dataset generation completed successfully!")
    print(f"Data saved to: {out_dir}")


if __name__ == "__main__":
    main()

