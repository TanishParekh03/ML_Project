"""AugMix Implementation for GTSRB (ICLR 2020 Grounded)

Methodological Grounding:
Hendrycks, D., Mu, N., Cubuk, E. D., Zoph, B., Gilmer, J., & Dietterich, T. (2020).
AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty.
International Conference on Learning Representations (ICLR 2020).

GTSRB Domain Specific Constraints (Mentor-Defense Engineered):
1. Test-Leakage Elimination: Strictly excludes noise (Gaussian/shot), blur (motion/defocus),
   weather (fog/snow/frost), and digital compression (JPEG) from the training transforms.
2. Orientation Bounding: Traffic signs are orientation-sensitive (e.g. Turn Right vs Turn Left,
   Give Way vs Warning Triangle). Rotation is strictly bounded to max +/- 10 degrees.
3. Shear Bounding: Bounded to max +/- 8 degrees to prevent distorting circular signs into ellipses.
4. Translation Bounding: Bounded to max +/- 4 pixels (+/- 6.25% of 64x64 canvas) to keep the sign
   firmly inside the canonical canvas.
5. Numerical Stability: Jensen-Shannon Divergence computed in float32 with log-space clamping
   to prevent underflow/NaN under Automatic Mixed Precision (AMP).
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF

from src.dataset import GTSRBDataset, IMAGENET_MEAN, IMAGENET_STD, letterbox_roi_crop, resolve_image_path


# ==============================================================================
# 1. Domain-Constrained Augmentation Primitives
# ==============================================================================

def _autocontrast(img: Image.Image, _severity: float) -> Image.Image:
    return ImageOps.autocontrast(img)


def _equalize(img: Image.Image, _severity: float) -> Image.Image:
    return ImageOps.equalize(img)


def _posterize(img: Image.Image, severity: float) -> Image.Image:
    # Severity in [1, 10] maps to bits in [4, 7] (4=strongest quantization, 7=gentle)
    bits = max(4, min(7, int(round(7.0 - (severity - 1.0) * (3.0 / 9.0)))))
    return ImageOps.posterize(img, bits)


def _solarize(img: Image.Image, severity: float) -> Image.Image:
    # Threshold in [140, 240] (inverts only the brightest highlights, preserving dark symbols)
    threshold = int(round(240.0 - (severity - 1.0) * (100.0 / 9.0)))
    return ImageOps.solarize(img, threshold)


def _rotate(img: Image.Image, severity: float) -> Image.Image:
    # Bounded strictly to [-10, +10] degrees to preserve directional semantics
    max_angle = min(10.0, 2.0 + (severity - 1.0) * (8.0 / 9.0))
    angle = random.uniform(-max_angle, max_angle)
    return img.rotate(angle, resample=Image.Resampling.BILINEAR)


def _shear_x(img: Image.Image, severity: float) -> Image.Image:
    # Bounded to [-8, +8] degrees to preserve circular/triangular geometry
    max_shear = min(8.0, 1.5 + (severity - 1.0) * (6.5 / 9.0))
    shear_deg = random.uniform(-max_shear, max_shear)
    shear_rad = math.radians(shear_deg)
    return img.transform(
        img.size,
        Image.Transform.AFFINE,
        (1, math.tan(shear_rad), 0, 0, 1, 0),
        resample=Image.Resampling.BILINEAR,
    )


def _shear_y(img: Image.Image, severity: float) -> Image.Image:
    max_shear = min(8.0, 1.5 + (severity - 1.0) * (6.5 / 9.0))
    shear_deg = random.uniform(-max_shear, max_shear)
    shear_rad = math.radians(shear_deg)
    return img.transform(
        img.size,
        Image.Transform.AFFINE,
        (1, 0, 0, math.tan(shear_rad), 1, 0),
        resample=Image.Resampling.BILINEAR,
    )


def _translate_x(img: Image.Image, severity: float) -> Image.Image:
    # Bounded to [-4, +4] pixels on a 64x64 canvas (+/- 6.25%)
    max_dx = min(4, int(round(1 + (severity - 1.0) * (3.0 / 9.0))))
    dx = random.randint(-max_dx, max_dx)
    return img.transform(
        img.size,
        Image.Transform.AFFINE,
        (1, 0, -dx, 0, 1, 0),
        resample=Image.Resampling.BILINEAR,
    )


def _translate_y(img: Image.Image, severity: float) -> Image.Image:
    max_dy = min(4, int(round(1 + (severity - 1.0) * (3.0 / 9.0))))
    dy = random.randint(-max_dy, max_dy)
    return img.transform(
        img.size,
        Image.Transform.AFFINE,
        (1, 0, 0, 0, 1, -dy),
        resample=Image.Resampling.BILINEAR,
    )


# 9 canonical operations strictly disjoint from GTSRB-C test corruptions
AUGMIX_OPERATIONS: list[Callable[[Image.Image, float], Image.Image]] = [
    _autocontrast,
    _equalize,
    _posterize,
    _solarize,
    _rotate,
    _shear_x,
    _shear_y,
    _translate_x,
    _translate_y,
]

# 5 geometric-only operations (Ablation 5A: strictly chromatic-free)
AUGMIX_GEOMETRIC_OPERATIONS: list[Callable[[Image.Image, float], Image.Image]] = [
    _rotate,
    _shear_x,
    _shear_y,
    _translate_x,
    _translate_y,
]


# ==============================================================================
# 2. Augment-and-Mix Algorithm (Hendrycks et al., 2020)
# ==============================================================================

def augment_and_mix(
    img: Image.Image,
    severity: int = 3,
    width: int = 3,
    depth: int = -1,
    alpha: float = 1.0,
    exclude_color_ops: bool = False,
) -> Image.Image:
    """Applies AugMix transformation to an input PIL Image.

    Args:
        img: Input PIL Image (centered, letterboxed canvas).
        severity: Severity of augmentation operations (default: 3, range: [1, 10]).
        width: Number of parallel augmentation chains / branches (default: 3).
        depth: Number of chained operations per branch (-1 for random in [1, 3]).
        alpha: Hyperparameter for Dirichlet and Beta distributions (default: 1.0).

    Returns:
        Augmented PIL Image blended via convex combination.
    """
    img_np = np.asarray(img).astype(np.float32) / 255.0
    h, w, c = img_np.shape

    # Sample mixture weights across parallel branches: w ~ Dirichlet(alpha, ..., alpha)
    ws = np.random.dirichlet([alpha] * width).astype(np.float32)
    # Sample clean image interpolation coefficient: m ~ Beta(alpha, alpha)
    m = float(np.random.beta(alpha, alpha))

    # Initialize convex accumulator for the branches
    mixed = np.zeros_like(img_np)

    for i in range(width):
        branch_img = img.copy()
        # Stochastic depth: d ~ Uniform({1, 2, 3})
        d = random.randint(1, 3) if depth < 0 else depth

        ops_pool = AUGMIX_GEOMETRIC_OPERATIONS if exclude_color_ops else AUGMIX_OPERATIONS
        for _ in range(d):
            op = random.choice(ops_pool)
            branch_img = op(branch_img, float(severity))

        branch_np = np.asarray(branch_img).astype(np.float32) / 255.0
        mixed += ws[i] * branch_np

    # Blend mixed branch representation with clean original image
    final_np = m * img_np + (1.0 - m) * mixed
    final_np = np.clip(final_np * 255.0, 0.0, 255.0).astype(np.uint8)

    return Image.fromarray(final_np)


# ==============================================================================
# 3. AugMix GTSRB Dataset (Produces Clean + 2 Augmented Views)
# ==============================================================================

class AugMixDataset(GTSRBDataset):
    """GTSRB Dataset yielding triplets (clean, aug1, aug2) for AugMix JSD training."""

    def __init__(
        self,
        manifest_path: str | Path,
        data_root: str | Path,
        target_size: int = 64,
        margin_ratio: float = 0.10,
        severity: int = 3,
        width: int = 3,
        depth: int = -1,
        alpha: float = 1.0,
        exclude_color_ops: bool = False,
        is_training: bool = True,
    ):
        super().__init__(
            manifest_path=manifest_path,
            data_root=data_root,
            target_size=target_size,
            margin_ratio=margin_ratio,
            is_training=is_training,
        )
        self.severity = severity
        self.width = width
        self.depth = depth
        self.alpha = alpha
        self.exclude_color_ops = exclude_color_ops

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        img_path = resolve_image_path(self.data_root, row["source_relative_path"])

        with Image.open(img_path) as raw_img:
            img = raw_img.convert("RGB")

        roi = (int(row["roi_x1"]), int(row["roi_y1"]), int(row["roi_x2"]), int(row["roi_y2"]))
        canvas = letterbox_roi_crop(img, roi, target_size=self.target_size, margin_ratio=self.margin_ratio)

        def _to_normalized_tensor(pil_im: Image.Image) -> torch.Tensor:
            t = TF.to_tensor(pil_im)
            return TF.normalize(t, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        t_clean = _to_normalized_tensor(canvas)

        if self.is_training:
            # Generate two distinct augmented views via independent AugMix sampling
            aug1_pil = augment_and_mix(
                canvas, severity=self.severity, width=self.width, depth=self.depth, alpha=self.alpha,
                exclude_color_ops=self.exclude_color_ops,
            )
            aug2_pil = augment_and_mix(
                canvas, severity=self.severity, width=self.width, depth=self.depth, alpha=self.alpha,
                exclude_color_ops=self.exclude_color_ops,
            )

            t_aug1 = _to_normalized_tensor(aug1_pil)
            t_aug2 = _to_normalized_tensor(aug2_pil)
        else:
            t_aug1 = t_clean
            t_aug2 = t_clean

        class_id = int(row["class_id"])

        return {
            "image": t_clean,           # Standard clean tensor for evaluation
            "image_clean": t_clean,     # Primary view
            "image_aug1": t_aug1,       # Augmented view 1
            "image_aug2": t_aug2,       # Augmented view 2
            "label": torch.tensor(class_id, dtype=torch.long),
            "image_id": row["image_id"],
            "class_name": row.get("class_name", ""),
        }


# Alias for clean naming
AugMixGTSRB = AugMixDataset


# ==============================================================================
# 4. Jensen-Shannon Consistency Loss Function
# ==============================================================================

def jensen_shannon_loss(
    logits_clean: torch.Tensor,
    logits_aug1: torch.Tensor,
    logits_aug2: torch.Tensor,
    targets: torch.Tensor,
    lambda_jsd: float = 12.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes Cross-Entropy + Jensen-Shannon Consistency Loss (Hendrycks et al., 2020).

    Args:
        logits_clean: Logits for clean images [B, num_classes].
        logits_aug1: Logits for augmented view 1 [B, num_classes].
        logits_aug2: Logits for augmented view 2 [B, num_classes].
        targets: Ground-truth class labels [B].
        lambda_jsd: Weight for the JSD consistency regularization (default: 12.0).

    Returns:
        total_loss: L_CE(p_clean, y) + lambda_jsd * JSD(p_clean, p_aug1, p_aug2)
        loss_ce: Cross-Entropy loss on clean images.
        loss_jsd: Symmetric Jensen-Shannon divergence across the three distributions.
    """
    # 1. Supervised Cross-Entropy on clean view
    loss_ce = F.cross_entropy(logits_clean, targets)

    # 2. Probability distributions (casted to float32 for absolute numerical stability)
    p_clean = F.softmax(logits_clean.float(), dim=1)
    p_aug1 = F.softmax(logits_aug1.float(), dim=1)
    p_aug2 = F.softmax(logits_aug2.float(), dim=1)

    # 3. Average distribution: M = (p_clean + p_aug1 + p_aug2) / 3
    p_mean = (p_clean + p_aug1 + p_aug2) / 3.0
    log_p_mean = torch.log(p_mean.clamp(min=1e-7))

    # 4. Symmetrical KL Divergences: D_KL(P || M)
    # PyTorch kl_div: input is log-probabilities (log_p_mean), target is probabilities (p)
    kl_clean = F.kl_div(log_p_mean, p_clean, reduction="batchmean")
    kl_aug1 = F.kl_div(log_p_mean, p_aug1, reduction="batchmean")
    kl_aug2 = F.kl_div(log_p_mean, p_aug2, reduction="batchmean")

    loss_jsd = (kl_clean + kl_aug1 + kl_aug2) / 3.0
    total_loss = loss_ce + lambda_jsd * loss_jsd

    return total_loss, loss_ce, loss_jsd

