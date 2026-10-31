"""
concept_encoder.py ? Concept Bottleneck Model vocabulary and encoding utilities.

Converts the 43-class GTSRB concept ontology (class_concepts.csv) into
integer indices and one-hot vectors for the interpretable bottleneck layer.

Bottleneck design (22-dim total):
  axis          values  offset
  ??????????????????????????????
  shape         6       0
  colour_profile 5      6
  border_type   4       11
  symbol_category 7     15
  ??????????????????????????????
  TOTAL         22

symbol_content (43-way) is NOT part of the bottleneck; it would degenerate
to class identity.  It is used only as an auxiliary supervision signal.

This module has zero torch dependency and can be imported on any Python
3.8+ environment with numpy installed.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ??????????????????????????????????????????????????????????????????????????????
# 1.  Vocabulary definitions (order = integer index)
# ??????????????????????????????????????????????????????????????????????????????

SHAPE_VOCAB: Dict[str, int] = {
    "circle":        0,
    "triangle_up":   1,
    "triangle_down": 2,
    "octagon":       3,
    "diamond":       4,
    "rectangle":     5,
}

COLOUR_VOCAB: Dict[str, int] = {
    "white_red":    0,
    "blue_white":   1,
    "yellow_white": 2,
    "white_black":  3,
    "red_white":    4,
}

BORDER_VOCAB: Dict[str, int] = {
    "red_border":   0,
    "white_border": 1,
    "black_border": 2,
    "none":         3,
}

SYMCAT_VOCAB: Dict[str, int] = {
    "speed_numeral":          0,
    "warning_hazard":         1,
    "mandatory_direction":    2,
    "prohibition_restriction": 3,
    "priority_geometry":      4,
    "deregulation_cancel":    5,
    "none":                   6,
}

# Reverse mappings (index -> string)
SHAPE_VOCAB_INV:  Dict[int, str] = {v: k for k, v in SHAPE_VOCAB.items()}
COLOUR_VOCAB_INV: Dict[int, str] = {v: k for k, v in COLOUR_VOCAB.items()}
BORDER_VOCAB_INV: Dict[int, str] = {v: k for k, v in BORDER_VOCAB.items()}
SYMCAT_VOCAB_INV: Dict[int, str] = {v: k for k, v in SYMCAT_VOCAB.items()}

# Bottleneck dimension constants
N_SHAPE   = len(SHAPE_VOCAB)    # 6
N_COLOUR  = len(COLOUR_VOCAB)   # 5
N_BORDER  = len(BORDER_VOCAB)   # 4
N_SYMCAT  = len(SYMCAT_VOCAB)   # 7
BOTTLENECK_DIM = N_SHAPE + N_COLOUR + N_BORDER + N_SYMCAT  # 22

# One-hot offset positions
SHAPE_OFFSET   = 0
COLOUR_OFFSET  = SHAPE_OFFSET  + N_SHAPE    # 6
BORDER_OFFSET  = COLOUR_OFFSET + N_COLOUR   # 11
SYMCAT_OFFSET  = BORDER_OFFSET + N_BORDER   # 15

# ??????????????????????????????????????????????????????????????????????????????
# 2.  CSV ingestion
# ??????????????????????????????????????????????????????????????????????????????

# Frozen 43-class GTSRB concept table (grounded in Vienna Convention / German StVO)
# Provides fallback if class_concepts.csv is missing on runtime.
FROZEN_CONCEPTS_TABLE: List[Dict[str, Any]] = [
    {"class_id": 0, "class_name": "Speed limit (20km/h)", "shape": "circle", "colour_profile": "white_red", "border_type": "red_border", "symbol_category": "speed_numeral", "symbol_content": "20"},
    {"class_id": 1, "class_name": "Speed limit (30km/h)", "shape": "circle", "colour_profile": "white_red", "border_type": "red_border", "symbol_category": "speed_numeral", "symbol_content": "30"},
    {"class_id": 2, "class_name": "Speed limit (50km/h)", "shape": "circle", "colour_profile": "white_red", "border_type": "red_border", "symbol_category": "speed_numeral", "symbol_content": "50"},
    {"class_id": 3, "class_name": "Speed limit (60km/h)", "shape": "circle", "colour_profile": "white_red", "border_type": "red_border", "symbol_category": "speed_numeral", "symbol_content": "60"},
    {"class_id": 4, "class_name": "Speed limit (70km/h)", "shape": "circle", "colour_profile": "white_red", "border_type": "red_border", "symbol_category": "speed_numeral", "symbol_content": "70"},
    {"class_id": 5, "class_name": "Speed limit (80km/h)", "shape": "circle", "colour_profile": "white_red", "border_type": "red_border", "symbol_category": "speed_numeral", "symbol_content": "80"},
    {"class_id": 6, "class_name": "End of speed limit (80km/h)", "shape": "circle", "colour_profile": "white_black", "border_type": "black_border", "symbol_category": "deregulation_cancel", "symbol_content": "end_speed_80"},
    {"class_id": 7, "class_name": "Speed limit (100km/h)", "shape": "circle", "colour_profile": "white_red", "border_type": "red_border", "symbol_category": "speed_numeral", "symbol_content": "100"},
    {"class_id": 8, "class_name": "Speed limit (120km/h)", "shape": "circle", "colour_profile": "white_red", "border_type": "red_border", "symbol_category": "speed_numeral", "symbol_content": "120"},
    {"class_id": 9, "class_name": "No passing", "shape": "circle", "colour_profile": "white_red", "border_type": "red_border", "symbol_category": "prohibition_restriction", "symbol_content": "no_passing_cars"},
    {"class_id": 10, "class_name": "No passing for vehicles over 3.5t", "shape": "circle", "colour_profile": "white_red", "border_type": "red_border", "symbol_category": "prohibition_restriction", "symbol_content": "no_passing_trucks"},
    {"class_id": 11, "class_name": "Right-of-way at intersection", "shape": "triangle_up", "colour_profile": "white_red", "border_type": "red_border", "symbol_category": "priority_geometry", "symbol_content": "crossroad_priority"},
    {"class_id": 12, "class_name": "Priority road", "shape": "diamond", "colour_profile": "yellow_white", "border_type": "white_border", "symbol_category": "priority_geometry", "symbol_content": "priority_diamond"},
    {"class_id": 13, "class_name": "Yield", "shape": "triangle_down", "colour_profile": "white_red", "border_type": "red_border", "symbol_category": "priority_geometry", "symbol_content": "yield_triangle"},
    {"class_id": 14, "class_name": "Stop", "shape": "octagon", "colour_profile": "red_white", "border_type": "white_border", "symbol_category": "priority_geometry", "symbol_content": "stop_text"},
    {"class_id": 15, "class_name": "No vehicles", "shape": "circle", "colour_profile": "white_red", "border_type": "red_border", "symbol_category": "none", "symbol_content": "empty_field"},
    {"class_id": 16, "class_name": "Vehicles over 3.5t prohibited", "shape": "circle", "colour_profile": "white_red", "border_type": "red_border", "symbol_category": "prohibition_restriction", "symbol_content": "truck_restriction"},
    {"class_id": 17, "class_name": "No entry", "shape": "circle", "colour_profile": "red_white", "border_type": "white_border", "symbol_category": "prohibition_restriction", "symbol_content": "no_entry_bar"},
    {"class_id": 18, "class_name": "General caution", "shape": "triangle_up", "colour_profile": "white_red", "border_type": "red_border", "symbol_category": "warning_hazard", "symbol_content": "general_caution"},
    {"class_id": 19, "class_name": "Dangerous curve to the left", "shape": "triangle_up", "colour_profile": "white_red", "border_type": "red_border", "symbol_category": "warning_hazard", "symbol_content": "curve_left"},
    {"class_id": 20, "class_name": "Dangerous curve to the right", "shape": "triangle_up", "colour_profile": "white_red", "border_type": "red_border", "symbol_category": "warning_hazard", "symbol_content": "curve_right"},
    {"class_id": 21, "class_name": "Double curve", "shape": "triangle_up", "colour_profile": "white_red", "border_type": "red_border", "symbol_category": "warning_hazard", "symbol_content": "double_curve"},
    {"class_id": 22, "class_name": "Bumpy road", "shape": "triangle_up", "colour_profile": "white_red", "border_type": "red_border", "symbol_category": "warning_hazard", "symbol_content": "bumpy_road"},
    {"class_id": 23, "class_name": "Slippery road", "shape": "triangle_up", "colour_profile": "white_red", "border_type": "red_border", "symbol_category": "warning_hazard", "symbol_content": "slippery_road"},
    {"class_id": 24, "class_name": "Road narrows on the right", "shape": "triangle_up", "colour_profile": "white_red", "border_type": "red_border", "symbol_category": "warning_hazard", "symbol_content": "road_narrows_right"},
    {"class_id": 25, "class_name": "Road work", "shape": "triangle_up", "colour_profile": "white_red", "border_type": "red_border", "symbol_category": "warning_hazard", "symbol_content": "road_work"},
    {"class_id": 26, "class_name": "Traffic signals", "shape": "triangle_up", "colour_profile": "white_red", "border_type": "red_border", "symbol_category": "warning_hazard", "symbol_content": "traffic_light"},
    {"class_id": 27, "class_name": "Pedestrians", "shape": "triangle_up", "colour_profile": "white_red", "border_type": "red_border", "symbol_category": "warning_hazard", "symbol_content": "pedestrians"},
    {"class_id": 28, "class_name": "Children crossing", "shape": "triangle_up", "colour_profile": "white_red", "border_type": "red_border", "symbol_category": "warning_hazard", "symbol_content": "children"},
    {"class_id": 29, "class_name": "Bicycles crossing", "shape": "triangle_up", "colour_profile": "white_red", "border_type": "red_border", "symbol_category": "warning_hazard", "symbol_content": "bicycles"},
    {"class_id": 30, "class_name": "Beware of ice/snow", "shape": "triangle_up", "colour_profile": "white_red", "border_type": "red_border", "symbol_category": "warning_hazard", "symbol_content": "ice_snow"},
    {"class_id": 31, "class_name": "Wild animals crossing", "shape": "triangle_up", "colour_profile": "white_red", "border_type": "red_border", "symbol_category": "warning_hazard", "symbol_content": "wild_animals"},
    {"class_id": 32, "class_name": "End speed and passing limits", "shape": "circle", "colour_profile": "white_black", "border_type": "black_border", "symbol_category": "deregulation_cancel", "symbol_content": "end_all_restrictions"},
    {"class_id": 33, "class_name": "Turn right ahead", "shape": "circle", "colour_profile": "blue_white", "border_type": "white_border", "symbol_category": "mandatory_direction", "symbol_content": "turn_right"},
    {"class_id": 34, "class_name": "Turn left ahead", "shape": "circle", "colour_profile": "blue_white", "border_type": "white_border", "symbol_category": "mandatory_direction", "symbol_content": "turn_left"},
    {"class_id": 35, "class_name": "Ahead only", "shape": "circle", "colour_profile": "blue_white", "border_type": "white_border", "symbol_category": "mandatory_direction", "symbol_content": "ahead_only"},
    {"class_id": 36, "class_name": "Go straight or right", "shape": "circle", "colour_profile": "blue_white", "border_type": "white_border", "symbol_category": "mandatory_direction", "symbol_content": "straight_or_right"},
    {"class_id": 37, "class_name": "Go straight or left", "shape": "circle", "colour_profile": "blue_white", "border_type": "white_border", "symbol_category": "mandatory_direction", "symbol_content": "straight_or_left"},
    {"class_id": 38, "class_name": "Keep right", "shape": "circle", "colour_profile": "blue_white", "border_type": "white_border", "symbol_category": "mandatory_direction", "symbol_content": "keep_right"},
    {"class_id": 39, "class_name": "Keep left", "shape": "circle", "colour_profile": "blue_white", "border_type": "white_border", "symbol_category": "mandatory_direction", "symbol_content": "keep_left"},
    {"class_id": 40, "class_name": "Roundabout mandatory", "shape": "circle", "colour_profile": "blue_white", "border_type": "white_border", "symbol_category": "mandatory_direction", "symbol_content": "roundabout"},
    {"class_id": 41, "class_name": "End of no passing", "shape": "circle", "colour_profile": "white_black", "border_type": "black_border", "symbol_category": "deregulation_cancel", "symbol_content": "end_no_passing_cars"},
    {"class_id": 42, "class_name": "End of no passing by vehicles over 3.5t", "shape": "circle", "colour_profile": "white_black", "border_type": "black_border", "symbol_category": "deregulation_cancel", "symbol_content": "end_no_passing_trucks"},
]


def _find_csv(csv_path: Optional[str] = None) -> Optional[Path]:
    """Locate class_concepts.csv relative to the project root or current working directory."""
    if csv_path is not None:
        p = Path(csv_path)
        if p.exists():
            return p
        return None

    # Search candidates
    here = Path(__file__).resolve()
    candidates = [
        Path("data/master_v1/concepts/class_concepts.csv"),
        Path.cwd() / "data" / "master_v1" / "concepts" / "class_concepts.csv",
        Path("/content/ML_Project/data/master_v1/concepts/class_concepts.csv"),
        here.parent.parent.parent / "data" / "master_v1" / "concepts" / "class_concepts.csv",
        here.parent.parent / "data" / "master_v1" / "concepts" / "class_concepts.csv",
    ]
    for c in candidates:
        if c.exists():
            return c

    return None


def load_class_concepts(csv_path: Optional[str] = None) -> List[Dict]:
    """
    Load class_concepts.csv and return a list of row dicts (one per class).
    Falls back gracefully to FROZEN_CONCEPTS_TABLE if the file is not on disk.
    """
    path = _find_csv(csv_path)
    if path is None:
        # Fall back to embedded table
        return [dict(row) for row in FROZEN_CONCEPTS_TABLE]

    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if not row.get("class_id", "").strip():
                continue  # skip empty trailing lines
            rows.append({
                "class_id":        int(row["class_id"]),
                "class_name":      row["class_name"].strip(),
                "shape":           row["shape"].strip(),
                "colour_profile":  row["colour_profile"].strip(),
                "border_type":     row["border_type"].strip(),
                "symbol_category": row["symbol_category"].strip(),
                "symbol_content":  row["symbol_content"].strip(),
            })

    if csv_path is not None and len(rows) > 0:
        return rows

    if len(rows) == 43:
        return rows

    # Fall back if file was truncated or incomplete
    return [dict(row) for row in FROZEN_CONCEPTS_TABLE]



# ??????????????????????????????????????????????????????????????????????????????
# 3.  Encoding functions
# ??????????????????????????????????????????????????????????????????????????????

def encode_all_classes(
    csv_path: Optional[str] = None,
) -> Dict[int, Dict[str, int]]:
    """
    Return a dict mapping class_id -> {shape_idx, colour_idx, border_idx, symcat_idx}.

    Example
    -------
    >>> enc = encode_all_classes()
    >>> enc[0]
    {'shape_idx': 0, 'colour_idx': 0, 'border_idx': 0, 'symcat_idx': 0}
    """
    rows = load_class_concepts(csv_path)
    result: Dict[int, Dict[str, int]] = {}
    for row in rows:
        cid = row["class_id"]
        result[cid] = {
            "shape_idx":   SHAPE_VOCAB[row["shape"]],
            "colour_idx":  COLOUR_VOCAB[row["colour_profile"]],
            "border_idx":  BORDER_VOCAB[row["border_type"]],
            "symcat_idx":  SYMCAT_VOCAB[row["symbol_category"]],
        }
    return result


def get_concept_tensor_for_class(
    class_id: int,
    encoding: Optional[Dict[int, Dict[str, int]]] = None,
    csv_path: Optional[str] = None,
) -> np.ndarray:
    """
    Return the 22-dim one-hot bottleneck vector for a single class_id.

    Layout:
      [0:6]  shape (6-way one-hot)
      [6:11] colour_profile (5-way one-hot)
      [11:15] border_type (4-way one-hot)
      [15:22] symbol_category (7-way one-hot)

    Parameters
    ----------
    class_id  : int  Class index 0?42.
    encoding  : pre-computed dict from encode_all_classes() (optional, avoids
                re-reading the CSV on every call).
    csv_path  : override CSV location (optional).

    Returns
    -------
    np.ndarray, shape (22,), dtype float32
    """
    if encoding is None:
        encoding = encode_all_classes(csv_path)

    enc = encoding[class_id]
    vec = np.zeros(BOTTLENECK_DIM, dtype=np.float32)

    vec[SHAPE_OFFSET  + enc["shape_idx"]]  = 1.0
    vec[COLOUR_OFFSET + enc["colour_idx"]] = 1.0
    vec[BORDER_OFFSET + enc["border_idx"]] = 1.0
    vec[SYMCAT_OFFSET + enc["symcat_idx"]] = 1.0

    return vec


def get_oracle_concept_matrix(
    csv_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the Oracle concept matrix for all 43 classes.

    Returns
    -------
    X : np.ndarray, shape (43, 22), float32 ? one row per class
    y : np.ndarray, shape (43,),  int64   ? class indices 0..42
    """
    encoding = encode_all_classes(csv_path)
    n_classes = len(encoding)
    X = np.zeros((n_classes, BOTTLENECK_DIM), dtype=np.float32)
    y = np.zeros(n_classes, dtype=np.int64)

    for class_id, enc in encoding.items():
        X[class_id] = get_concept_tensor_for_class(class_id, encoding=encoding)
        y[class_id] = class_id

    return X, y


def decode_concept_indices(
    shape_idx: int,
    colour_idx: int,
    border_idx: int,
    symcat_idx: int,
) -> Dict[str, str]:
    """
    Convert integer indices back to human-readable concept strings.

    Returns
    -------
    dict with keys: shape, colour_profile, border_type, symbol_category
    """
    return {
        "shape":           SHAPE_VOCAB_INV[shape_idx],
        "colour_profile":  COLOUR_VOCAB_INV[colour_idx],
        "border_type":     BORDER_VOCAB_INV[border_idx],
        "symbol_category": SYMCAT_VOCAB_INV[symcat_idx],
    }


# ??????????????????????????????????????????????????????????????????????????????
# 4.  Quick self-check when run as a script
# ??????????????????????????????????????????????????????????????????????????????

if __name__ == "__main__":
    print("=" * 60)
    print("concept_encoder.py ? self-check")
    print("=" * 60)

    enc = encode_all_classes()
    print(f"\nLoaded encoding for {len(enc)} classes.")
    print(f"Bottleneck dimension: {BOTTLENECK_DIM}  "
          f"(shape={N_SHAPE}, colour={N_COLOUR}, border={N_BORDER}, symcat={N_SYMCAT})")

    print("\nSample encodings:")
    for cid in [0, 11, 12, 13, 14, 35]:
        e = enc[cid]
        vec = get_concept_tensor_for_class(cid, encoding=enc)
        decoded = decode_concept_indices(
            e["shape_idx"], e["colour_idx"], e["border_idx"], e["symcat_idx"]
        )
        print(f"  class {cid:2d}: idx={e}  sum={int(vec.sum())}  decoded={decoded}")

    # Uniqueness check
    X, y = get_oracle_concept_matrix()
    unique_rows = np.unique(X, axis=0)
    print(f"\nOracle matrix shape: {X.shape}")
    print(f"Unique concept vectors: {len(unique_rows)} / {len(y)}")
    if len(unique_rows) < len(y):
        print("  WARN  Some classes share identical concept vectors!")
    else:
        print("  OK All 43 classes have distinct concept vectors.")

    print("\nDone.")
