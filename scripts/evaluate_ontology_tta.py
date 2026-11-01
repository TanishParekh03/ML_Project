"""
Ontology-Constrained Test-Time Adaptation (OCTTA / Ontology Energy Minimization)
===============================================================================
Novel Test-Time Adaptation Paradigm for Concept Bottleneck Models (CBM).

The Core Problem with Prior TTA:
  - Standard BN-Adapt blindly aligns moments without checking semantics.
  - Standard TENT minimizes class entropy H(y_hat), which suffers from confirmation
    bias on severe corruptions (forcing 60% wrong guesses to become 99% confident wrong guesses).

The OCTTA Solution:
  - Traffic signs are bound by the legal Vienna Convention / StVO ontology O in {0, 1}^{43 x K}.
  - Out of all mathematical combinations of (shape, colour, border, symbol), only 43 valid
    prototypes are physically legal in the real world.
  - OCTTA projects the predicted concept distributions onto the nearest legally valid prototype:
      L_ontology(x) = sum_f w_f * CrossEntropy(l_f, idx_{m*, f})
  - Takes 1 gradient step at test time updating ONLY the BatchNorm affine parameters (gamma, beta).
  - All convolutional backbone weights and classifier heads remain strictly frozen.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def find_cbm_checkpoint(explicit_path: Optional[str] = None) -> Path:
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return p
        if (PROJECT_ROOT / p).exists():
            return PROJECT_ROOT / p

    candidates = [
        Path("/content/drive/MyDrive/ML_Project_Results/cbm_resnet18/best_cbm_resnet18.pt"),
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


class OntologyEnergyTTA:
    """
    Manages Ontology-Constrained Test-Time Adaptation.
    """
    def __init__(
        self,
        model: ConceptBottleneckModel,
        concepts_csv: str | Path,
        device: torch.device,
        lr: float = 1e-3,
        lambda_shape_ent: float = 0.1,
    ):
        self.model = model
        self.device = device
        self.lr = lr
        self.lambda_ent = lambda_shape_ent

        # Load 43-class legal ontology lookup tensors
        csv_p = Path(concepts_csv)
        if not csv_p.exists() and (PROJECT_ROOT / csv_p).exists():
            csv_p = PROJECT_ROOT / csv_p

        s_lut, c_lut, b_lut, sym_lut = get_concept_lookup_tensors(self.device, str(csv_p))
        self.luts = {
            "shape": s_lut,
            "colour": c_lut,
            "border": b_lut,
            "symcat": sym_lut,
        }

        # Concept stability weights based on empirical CDR
        self.weights = {
            "shape": (1.0 - 0.112) ** 2,   # 0.789
            "colour": (1.0 - 0.145) ** 2,  # 0.731
            "border": (1.0 - 0.158) ** 2,  # 0.709
            "symcat": (1.0 - 0.202) ** 2,  # 0.637
        }

        # Freeze everything except BatchNorm affine parameters (gamma, beta)
        self.adapt_params: List[nn.Parameter] = []
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            if "bn" in name or "downsample.1" in name:
                param.requires_grad = True
                self.adapt_params.append(param)

    def compute_ontology_energy_loss(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Computes the minimum distance from predictions to the nearest legal Vienna Convention prototype.
        """
        c_logits = {
            "shape": outputs["logits_shape"],
            "colour": outputs["logits_colour"],
            "border": outputs["logits_border"],
            "symcat": outputs["logits_symcat"],
        }
        B = outputs["logits"].size(0)

        # 1. Project onto all 43 class prototypes to find nearest valid prototype m*
        with torch.no_grad():
            proto_scores = torch.zeros((B, 43), device=self.device, dtype=torch.float32)
            for fam, l_f in c_logits.items():
                p_f = F.softmax(l_f, dim=-1)
                idx_m = self.luts[fam]  # [43]
                prob_for_m = p_f[:, idx_m].clamp(min=1e-7)
                proto_scores += self.weights[fam] * torch.log(prob_for_m)

            best_proto_m = torch.argmax(proto_scores, dim=1)  # [B]

        # 2. Compute Ontology Alignment Cross-Entropy Loss toward the nearest prototype m*
        total_ont_loss = torch.tensor(0.0, device=self.device)
        for fam, l_f in c_logits.items():
            target_concept_idx = self.luts[fam][best_proto_m]  # [B]
            ce_f = F.cross_entropy(l_f, target_concept_idx)
            total_ont_loss = total_ont_loss + self.weights[fam] * ce_f

        # 3. Shape Entropy Regularization (Sharpening the macro-geometric anchor)
        p_shape = F.softmax(c_logits["shape"], dim=-1)
        shape_entropy = -(p_shape * torch.log(p_shape.clamp(min=1e-7))).sum(dim=-1).mean()

        return total_ont_loss + self.lambda_ent * shape_entropy

    def adapt_batch_and_predict(self, batch_t: torch.Tensor, steps: int = 1) -> np.ndarray:
        """
        Takes 1 gradient step at test time on BatchNorm affine parameters and returns predictions.
        """
        # Ensure BatchNorm runs in test-time adaptation mode
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.train()
                m.momentum = 0.1

        optimizer = torch.optim.Adam(self.adapt_params, lr=self.lr, betas=(0.9, 0.999))

        for _ in range(steps):
            optimizer.zero_grad()
            outputs = self.model(batch_t)
            loss = self.compute_ontology_energy_loss(outputs)
            loss.backward()
            optimizer.step()

        # Forward pass on adapted parameters
        with torch.no_grad():
            final_out = self.model(batch_t)
            preds = torch.argmax(final_out["logits"], dim=-1).cpu().numpy()

        return preds


def run_octta_benchmark(
    checkpoint_path: Optional[str] = None,
    data_root: str = "german",
    meta_path: str = "data/master_v1/metadata/test.csv",
    concepts_csv: str = "data/master_v1/concepts/class_concepts.csv",
    batch_size: int = 128,
    sample_limit: Optional[int] = 2000,
    lr: float = 1e-3,
    output_csv: str = "reports/octta_ablation_sev5.csv",
) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    ckpt_p = find_cbm_checkpoint(checkpoint_path)
    print(f"Loading CBM Checkpoint: {ckpt_p}")

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
    print(f"Evaluating {len(test_df):,} test images across 15 Severity-5 corruptions...")

    # Load CBM model
    ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
    b_mode = ckpt.get("config", {}).get("bottleneck_mode", "residual") if isinstance(ckpt, dict) else "residual"
    model = ConceptBottleneckModel(num_classes=43, pretrained=False, bottleneck_mode=b_mode)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)

    # Save frozen master weights for exact resets between conditions
    clean_state = copy.deepcopy(model.state_dict())

    # Pre-cache test images into RAM
    print("Pre-caching letterboxed test images...")
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

    # Helper: Evaluate Static & Standard BN-Adapt
    @torch.no_grad()
    def _eval_standard(canvases_list, bn_adapt: bool) -> float:
        model.load_state_dict(clean_state)
        model.eval()
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if bn_adapt:
                    m.train()
                    m.momentum = 0.1
                else:
                    m.eval()

        preds = []
        for start in range(0, len(canvases_list), batch_size):
            b_imgs = canvases_list[start : start + batch_size]
            b_tensors = torch.stack([
                TF.normalize(TF.to_tensor(im), mean=IMAGENET_MEAN, std=IMAGENET_STD)
                for im in b_imgs
            ]).to(device)
            out = model(b_tensors)
            preds.extend(torch.argmax(out["logits"], dim=-1).cpu().numpy())
        return float(accuracy_score(targets_arr, np.array(preds)))

    # Helper: Evaluate OCTTA (Ontology-Constrained TTA)
    def _eval_octta(canvases_list) -> float:
        model.load_state_dict(clean_state)
        tta_engine = OntologyEnergyTTA(model=model, concepts_csv=concepts_p, device=device, lr=lr)

        preds = []
        for start in range(0, len(canvases_list), batch_size):
            b_imgs = canvases_list[start : start + batch_size]
            b_tensors = torch.stack([
                TF.normalize(TF.to_tensor(im), mean=IMAGENET_MEAN, std=IMAGENET_STD)
                for im in b_imgs
            ]).to(device)
            # Adapt affine parameters on batch & predict
            b_preds = tta_engine.adapt_batch_and_predict(b_tensors, steps=1)
            preds.extend(b_preds)
        return float(accuracy_score(targets_arr, np.array(preds)))

    # Clean Evaluation
    acc_clean_static = _eval_standard(clean_canvases, bn_adapt=False)
    acc_clean_octta = _eval_octta(clean_canvases)
    print("\n" + "=" * 80)
    print(f"CLEAN TEST ACCURACY | Static CBM: {acc_clean_static*100:.2f}% | OCTTA: {acc_clean_octta*100:.2f}%")
    print("=" * 80)

    # 15 Severity-5 Corruptions Evaluation
    print("\nEvaluating 15 Corruptions at Severity 5...")
    records = []
    for c_name, c_fn in CORRUPTIONS_REGISTRY.items():
        corrupted_canvases = [c_fn(canvas, 5) for canvas in clean_canvases]

        acc_static = _eval_standard(corrupted_canvases, bn_adapt=False)
        acc_bn = _eval_standard(corrupted_canvases, bn_adapt=True)
        acc_octta = _eval_octta(corrupted_canvases)

        records.append({
            "corruption": c_name,
            "static_cbm_sev5": acc_static,
            "bn_adapt_cbm_sev5": acc_bn,
            "octta_cbm_sev5": acc_octta,
            "octta_gain_over_static": acc_octta - acc_static,
            "octta_gain_over_bn": acc_octta - acc_bn,
        })

        print(
            f"[{c_name:18s} S5] "
            f"Static: {acc_static*100:5.2f}% | "
            f"BN-Adapt: {acc_bn*100:5.2f}% | "
            f"OCTTA (Ours): {acc_octta*100:5.2f}% "
            f"(Gain vs BN: {(acc_octta - acc_bn)*100:+4.2f}%)"
        )

    df = pd.DataFrame(records)
    print("\n" + "=" * 80)
    print("MEAN SEVERITY-5 ROBUSTNESS SUMMARY (15 CORRUPTIONS)")
    print("=" * 80)
    print(f"1. Static CBM Baseline            : {df['static_cbm_sev5'].mean()*100:.2f}%")
    print(f"2. Standard BN-Adapt CBM          : {df['bn_adapt_cbm_sev5'].mean()*100:.2f}%")
    print(f"3. Ontology-Constrained TTA (Ours): {df['octta_cbm_sev5'].mean()*100:.2f}%")
    print(f"   Net Gain of OCTTA over BN-Adapt: {(df['octta_cbm_sev5'].mean() - df['bn_adapt_cbm_sev5'].mean())*100:+.2f}%")
    print("=" * 80)

    out_p = Path(output_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_p, index=False)
    print(f"Detailed condition records saved to: {out_p}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Ontology-Constrained TTA (OCTTA) Benchmark")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data-root", type=str, default="german")
    parser.add_argument("--meta-path", type=str, default="data/master_v1/metadata/test.csv")
    parser.add_argument("--concepts-csv", type=str, default="data/master_v1/concepts/class_concepts.csv")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sample-limit", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-csv", type=str, default="reports/octta_ablation_sev5.csv")
    args = parser.parse_args()

    run_octta_benchmark(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        meta_path=args.meta_path,
        concepts_csv=args.concepts_csv,
        batch_size=args.batch_size,
        sample_limit=args.sample_limit,
        lr=args.lr,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
