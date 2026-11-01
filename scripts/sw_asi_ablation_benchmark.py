"""
Novel Approach: Stability-Weighted Autonomous Self-Intervention (SW-ASI)
========================================================================
Combines Concept Bottleneck Models (CBM), Test-Time Normalization Adaptation (BN-Adapt),
and the 4-Axis UNECE/StVO Traffic Sign Ontology.

Empirical Concept Degradation Rate (CDR) Priors:
  - Shape  : 11.2% CDR (spectrally stable, low-frequency macro-geometry)
  - Colour : 14.5% CDR (high stability, chromatic profile)
  - Border : 15.8% CDR (moderate-high stability, outer rim)
  - Symbol : 20.2% CDR (spectrally fragile, high-frequency inner glyphs)

Under corruption (blur, fog, noise), fragile symbol heads fail first.
SW-ASI uses the high-confidence, low-CDR heads (shape, colour, border) to project
onto the legal ontology and autonomously rectify fragile concept logits before final
classification without requiring human intervention or retraining!
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from src.corruptions import CORRUPTIONS_REGISTRY
from src.dataset import IMAGENET_MEAN, IMAGENET_STD, letterbox_roi_crop, resolve_image_path
from src.models.cbm import ConceptBottleneckModel, get_concept_lookup_tensors
from src.models.concept_encoder import (
    BORDER_OFFSET,
    BOTTLENECK_DIM,
    COLOUR_OFFSET,
    N_BORDER,
    N_COLOUR,
    N_SHAPE,
    N_SYMCAT,
    SHAPE_OFFSET,
    SYMCAT_OFFSET,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class StabilityWeightedSelfIntervention(nn.Module):
    """
    Stability-Weighted Autonomous Self-Intervention (SW-ASI) Wrapper.
    """
    def __init__(
        self,
        cbm_model: ConceptBottleneckModel,
        concepts_csv_path: str | Path = "data/master_v1/concepts/class_concepts.csv",
        cdr_priors: Optional[Dict[str, float]] = None,
        confidence_threshold: float = 0.75,
        intervention_strength: float = 0.85,
        logit_temp: float = 4.0,
    ):
        super().__init__()
        self.model = cbm_model
        self.tau = confidence_threshold
        self.alpha = intervention_strength
        self.logit_temp = logit_temp

        if cdr_priors is None:
            cdr_priors = {
                "shape": 0.112,    # 11.2% CDR
                "colour": 0.145,   # 14.5% CDR
                "border": 0.158,   # 15.8% CDR
                "symcat": 0.202,   # 20.2% CDR (Most fragile)
            }
        # Stability weight: w_f = (1.0 - CDR_f)^2
        self.weights = {k: (1.0 - v) ** 2 for k, v in cdr_priors.items()}

        csv_p = Path(concepts_csv_path)
        if not csv_p.exists() and (PROJECT_ROOT / csv_p).exists():
            csv_p = PROJECT_ROOT / csv_p

        # Load ontology lookup tensors [43] for each family
        s_lut, c_lut, b_lut, sym_lut = get_concept_lookup_tensors(torch.device("cpu"), str(csv_p))
        self.ontology_indices = {
            "shape": s_lut,
            "colour": c_lut,
            "border": b_lut,
            "symcat": sym_lut,
        }

    def _extract_family_logits(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        raw_out = self.model(x)
        c_logits = {
            "shape": raw_out["logits_shape"],
            "colour": raw_out["logits_colour"],
            "border": raw_out["logits_border"],
            "symcat": raw_out["logits_symcat"],
        }
        return raw_out["logits"], c_logits, raw_out

    @torch.no_grad()
    def forward(self, x: torch.Tensor, enable_sw_asi: bool = True) -> Dict[str, torch.Tensor]:
        orig_logits, c_logits, raw_out = self._extract_family_logits(x)

        if not enable_sw_asi:
            return raw_out

        device = x.device
        B = x.size(0)

        # 1. Compute Softmax Probabilities and Max Confidence per family
        c_probs = {}
        c_conf = {}
        for fam, l_f in c_logits.items():
            p_f = F.softmax(l_f.float(), dim=-1)
            c_probs[fam] = p_f
            c_conf[fam] = p_f.max(dim=-1).values  # [B]

        # 2. Stability-Weighted Ontology Prototype Compatibility Scoring: S(x, m) across all 43 classes
        proto_scores = torch.zeros((B, 43), device=device, dtype=torch.float32)

        for fam, p_f in c_probs.items():
            idx_m = self.ontology_indices[fam].to(device)  # [43]
            prob_for_class_m = p_f[:, idx_m].clamp(min=1e-7)  # [B, 43]
            log_prob_m = torch.log(prob_for_class_m)

            w_f = self.weights.get(fam, 1.0)
            dynamic_weight = w_f * c_conf[fam].unsqueeze(1)  # [B, 1]
            proto_scores += dynamic_weight * log_prob_m

        # 3. Identify Best-Matching Prototype m*
        best_proto_class = torch.argmax(proto_scores, dim=1)  # [B]

        # 4. Autonomous Self-Intervention on Low-Confidence Heads
        rectified_probs_list = []
        family_order = ["shape", "colour", "border", "symcat"]

        for fam in family_order:
            p_f = c_probs[fam]
            num_c = p_f.size(-1)

            implied_c_idx = self.ontology_indices[fam].to(device)[best_proto_class]  # [B]
            implied_one_hot = F.one_hot(implied_c_idx, num_classes=num_c).float()    # [B, num_c]

            # Intervene when concept confidence drops below threshold tau
            needs_intervention = (c_conf[fam] < self.tau).float().unsqueeze(1)       # [B, 1]
            gate = self.alpha * needs_intervention

            p_rect = (1.0 - gate) * p_f + gate * implied_one_hot
            rectified_probs_list.append(p_rect)

        # 5. Class Predictor Head Update
        rectified_c_hat = torch.cat(rectified_probs_list, dim=-1)  # [B, 22]
        orig_c_hat = raw_out["c_hat"]                             # [B, 22]

        if hasattr(self.model, "class_head"):
            delta_logits = self.model.class_head(rectified_c_hat) - self.model.class_head(orig_c_hat)
            final_logits = orig_logits + delta_logits
        else:
            uncertain_sample = (c_conf["symcat"] < self.tau).float().unsqueeze(1)
            normalized_proto = F.layer_norm(proto_scores, (43,))
            final_logits = orig_logits + (self.alpha * uncertain_sample) * normalized_proto

        out = dict(raw_out)
        out["logits"] = final_logits
        out["sw_asi_triggered_rate"] = (c_conf["symcat"] < self.tau).float().mean()
        return out


def configure_bn_adaptation(model: nn.Module, enable: bool = True, momentum: float = 0.1):
    """Configures Test-Time Batch Normalization Adaptation without modifying backbone weights."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if enable:
                m.train()
                m.momentum = momentum
            else:
                m.eval()


def find_checkpoint(explicit_path: Optional[str] = None) -> Path:
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return p
        if (PROJECT_ROOT / p).exists():
            return PROJECT_ROOT / p

    candidates = [
        Path("/content/drive/MyDrive/Colab Notebooks/best_cbm_resnet18.pt"),
        Path("/content/drive/MyDrive/ML_Project_Checkpoints/cbm_exp4/best_cbm_resnet18.pt"),
        PROJECT_ROOT / "reports/cbm_resnet18/best_cbm_resnet18.pt",
        PROJECT_ROOT / "checkpoints/cbm_resnet18/best_cbm_resnet18.pt",
        Path("checkpoints/cbm_resnet18/best_cbm_resnet18.pt"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Could not locate CBM checkpoint. Checked: {candidates}")


def run_sw_asi_ablation_benchmark(
    checkpoint_path: Optional[str] = None,
    data_root: str = "german",
    meta_path: str = "data/master_v1/metadata/test.csv",
    concepts_csv: str = "data/master_v1/concepts/class_concepts.csv",
    batch_size: int = 128,
    sample_limit: Optional[int] = 2000,
    output_csv: str = "reports/sw_asi_ablation_sev5.csv",
) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # Resolve paths
    ckpt_path = find_checkpoint(checkpoint_path)
    print(f"Loading checkpoint: {ckpt_path}")

    data_root_p = Path(data_root)
    if not data_root_p.exists() and (PROJECT_ROOT / data_root_p).exists():
        data_root_p = PROJECT_ROOT / data_root_p

    meta_p = Path(meta_path)
    if not meta_p.exists() and (PROJECT_ROOT / meta_p).exists():
        meta_p = PROJECT_ROOT / meta_p

    concepts_p = Path(concepts_csv)
    if not concepts_p.exists() and (PROJECT_ROOT / concepts_p).exists():
        concepts_p = PROJECT_ROOT / concepts_p

    test_df = pd.read_csv(meta_p)
    if sample_limit is not None and len(test_df) > sample_limit:
        test_df = test_df.sample(n=sample_limit, random_state=42).reset_index(drop=True)
    print(f"Benchmark test sample count: {len(test_df):,} images")

    # 1. Load trained CBM model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    b_mode = cfg.get("bottleneck_mode", "residual")

    base_cbm = ConceptBottleneckModel(num_classes=43, pretrained=False, bottleneck_mode=b_mode)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    base_cbm.load_state_dict(state_dict)
    base_cbm.to(device)
    base_cbm.eval()

    clean_bn_state = copy.deepcopy(base_cbm.state_dict())

    # Wrap with SW-ASI
    sw_asi_model = StabilityWeightedSelfIntervention(
        cbm_model=base_cbm,
        concepts_csv_path=concepts_p,
        confidence_threshold=0.75,
        intervention_strength=0.85,
    ).to(device)

    # 2. Pre-cache letterboxed test images into RAM
    print(f"Pre-caching {len(test_df):,} test canvas images...")
    clean_canvases = []
    targets = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Caching"):
        rel_p = str(row["source_relative_path"]).replace("\\", "/")
        img_p = resolve_image_path(data_root_p, rel_p)
        with Image.open(img_p) as raw_img:
            img = raw_img.convert("RGB")
        roi = (int(row["roi_x1"]), int(row["roi_y1"]), int(row["roi_x2"]), int(row["roi_y2"]))
        canvas = letterbox_roi_crop(img, roi, target_size=64)
        clean_canvases.append(canvas)
        targets.append(int(row["class_id"]))
    targets_arr = np.array(targets)

    def _eval_condition(canvases_list, use_bn_adapt: bool, use_sw_asi: bool) -> Tuple[float, float]:
        base_cbm.load_state_dict(clean_bn_state)
        configure_bn_adaptation(base_cbm, enable=use_bn_adapt, momentum=0.1)

        preds = []
        trig_rates = []
        for start in range(0, len(canvases_list), batch_size):
            batch_imgs = canvases_list[start : start + batch_size]
            tensors = [
                TF.normalize(TF.to_tensor(im), mean=IMAGENET_MEAN, std=IMAGENET_STD)
                for im in batch_imgs
            ]
            batch_t = torch.stack(tensors).to(device)
            out = sw_asi_model(batch_t, enable_sw_asi=use_sw_asi)
            batch_preds = torch.argmax(out["logits"], dim=1).cpu().numpy()
            preds.extend(batch_preds)
            if "sw_asi_triggered_rate" in out:
                trig_rates.append(float(out["sw_asi_triggered_rate"].cpu()))

        acc = accuracy_score(targets_arr, np.array(preds))
        avg_trig = float(np.mean(trig_rates)) if trig_rates else 0.0
        return float(acc), avg_trig

    # 3. Clean Evaluation
    print("\nEvaluating Clean Test Set...")
    clean_static, _ = _eval_condition(clean_canvases, use_bn_adapt=False, use_sw_asi=False)
    clean_sw_asi, clean_trig = _eval_condition(clean_canvases, use_bn_adapt=False, use_sw_asi=True)
    print("=" * 80)
    print(f"CLEAN TEST ACCURACY | Static CBM: {clean_static*100:.2f}% | + SW-ASI: {clean_sw_asi*100:.2f}% (Triggered: {clean_trig*100:.1f}%)")
    print("=" * 80)

    # 4. Severity-5 Evaluation across all 15 Corruptions
    print("\nEvaluating Severity-5 Corruptions Suite...")
    results = []
    for c_name, c_func in CORRUPTIONS_REGISTRY.items():
        corrupted_canvases = [c_func(canvas, 5) for canvas in clean_canvases]

        acc_static, _ = _eval_condition(corrupted_canvases, use_bn_adapt=False, use_sw_asi=False)
        acc_sw_asi, trig_s = _eval_condition(corrupted_canvases, use_bn_adapt=False, use_sw_asi=True)
        acc_bn_only, _ = _eval_condition(corrupted_canvases, use_bn_adapt=True, use_sw_asi=False)
        acc_bn_sw_asi, trig_b = _eval_condition(corrupted_canvases, use_bn_adapt=True, use_sw_asi=True)

        err_before = 1.0 - acc_bn_only
        ir_auto = ((acc_bn_sw_asi - acc_bn_only) / err_before) if err_before > 1e-5 else 0.0

        results.append({
            "corruption": c_name,
            "static_cbm_sev5": acc_static,
            "static_plus_sw_asi_sev5": acc_sw_asi,
            "bn_adapt_only_sev5": acc_bn_only,
            "bn_adapt_plus_sw_asi_sev5": acc_bn_sw_asi,
            "sw_asi_trigger_rate": trig_b,
            "autonomous_ir_rate": ir_auto,
        })

        print(
            f"[{c_name:18s} S5] "
            f"Static: {acc_static*100:5.2f}% -> +SW-ASI: {acc_sw_asi*100:5.2f}% | "
            f"BN-Adapt: {acc_bn_only*100:5.2f}% -> +SW-ASI: {acc_bn_sw_asi*100:5.2f}% "
            f"(Trig: {trig_b*100:4.1f}%)"
        )

    res_df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("MEAN SEVERITY-5 ROBUSTNESS SUMMARY (15 CORRUPTIONS)")
    print("=" * 80)
    print(f"1. Static CBM                  : {res_df['static_cbm_sev5'].mean()*100:.2f}%")
    print(f"2. Static CBM + SW-ASI (Ours)  : {res_df['static_plus_sw_asi_sev5'].mean()*100:.2f}%")
    print(f"3. BN-Adapt CBM                : {res_df['bn_adapt_only_sev5'].mean()*100:.2f}%")
    print(f"4. BN-Adapt + SW-ASI (Ours)    : {res_df['bn_adapt_plus_sw_asi_sev5'].mean()*100:.2f}%")
    print("=" * 80)

    out_p = Path(output_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_p, index=False)
    print(f"Saved results to: {out_p}")
    return res_df


def main():
    parser = argparse.ArgumentParser(description="SW-ASI Ablation Benchmark")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data-root", type=str, default="german")
    parser.add_argument("--meta-path", type=str, default="data/master_v1/metadata/test.csv")
    parser.add_argument("--concepts-csv", type=str, default="data/master_v1/concepts/class_concepts.csv")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sample-limit", type=int, default=2000, help="Number of test samples (None for all 12,630)")
    parser.add_argument("--output-csv", type=str, default="reports/sw_asi_ablation_sev5.csv")
    args = parser.parse_args()

    run_sw_asi_ablation_benchmark(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        meta_path=args.meta_path,
        concepts_csv=args.concepts_csv,
        batch_size=args.batch_size,
        sample_limit=args.sample_limit,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
