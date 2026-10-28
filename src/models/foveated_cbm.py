"""
Foveated Spatial-Transformer Concept Bottleneck Model (Exp 8: Foveated-CBM)
===========================================================================
Addresses two structural vulnerabilities of GTSRB traffic sign perception:
1. Within-Group Speed-Limit Numerals (66.3% of within-group errors):
   - Adds a Foveated Interior Feature Branch extracting the central 50% feature grid
     where numeral/pictogram strokes reside, decoupled from outer circular borders.
   - Adds Within-Group Contrastive Margin Loss (L_wg) forcing inter-digit separation.
2. Catastrophic Elastic Warp Collapse:
   - Integrates a lightweight Spatial Transformer Network (STN) localization head
     ahead of the auxiliary glyph branch to rectify non-rigid elastic warps.

Fully independent module: keeps src/models/cbm.py 100% frozen and backward-compatible.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights

from src.models.concept_encoder import (
    N_BORDER,
    N_COLOUR,
    N_SHAPE,
    N_SYMCAT,
    BOTTLENECK_DIM,
)

NUM_CLASSES = 43
NUM_CONCEPTS_SHAPE = N_SHAPE
NUM_CONCEPTS_COLOUR = N_COLOUR
NUM_CONCEPTS_BORDER = N_BORDER
NUM_CONCEPTS_SYMCAT = N_SYMCAT
TOTAL_CONCEPT_DIM = BOTTLENECK_DIM


class FoveatedConceptBottleneckModel(nn.Module):
    """
    Foveated Spatial-Transformer Concept Bottleneck Model.

    Combines:
    - Global feature extraction for 4 macroscopic physical concept heads
      (Shape: 6, Colour: 5, Border: 4, Symbol Category: 7 -> 22 dims).
    - Spatial Transformer Network (STN) grid rectifier for elastic distortions.
    - Foveated Interior ROI pooling branch focusing on inner numeral typography.
    - Auxiliary glyph head fed by concatenated global + foveated interior features.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        zero_init_residual: bool = True,
        bottleneck_mode: str = "residual",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.bottleneck_mode = bottleneck_mode

        # 1. Base ResNet-18 Feature Extractor
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        base = models.resnet18(weights=weights)

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3  # Output: [B, 256, 8, 8]
        self.layer4 = base.layer4  # Output: [B, 512, 4, 4]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 2. Lightweight Spatial Transformer Localization Network
        # Operates on layer3 features (8x8x256) to predict 2x3 affine matrix theta
        self.stn_loc = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 4x4
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),  # 2x2x32 = 128
        )
        self.stn_fc = nn.Linear(32 * 4, 6)
        # Initialize STN to identity transformation: [ [1, 0, 0], [0, 1, 0] ]
        self.stn_fc.weight.data.zero_()
        self.stn_fc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        # 3. Intermediate Concept Heads (from global 512-dim pooled features)
        self.head_shape = nn.Linear(512, NUM_CONCEPTS_SHAPE)
        self.head_colour = nn.Linear(512, NUM_CONCEPTS_COLOUR)
        self.head_border = nn.Linear(512, NUM_CONCEPTS_BORDER)
        self.head_symcat = nn.Linear(512, NUM_CONCEPTS_SYMCAT)

        # 4. Concept-to-Class Head: maps 22 concepts -> 43 classes
        self.class_head = nn.Linear(TOTAL_CONCEPT_DIM, num_classes)

        # 5. Foveated Auxiliary Head: fuses global (512) + foveated interior (512) -> 43 classes
        self.aux_head = nn.Linear(512 + 512, num_classes)

        # 6. Apply initializations
        self._initialize_custom_heads(zero_init_residual)

    def _initialize_custom_heads(self, zero_init_residual: bool) -> None:
        """Applies Kaiming normal init to linear heads and zero-gamma BN init."""
        heads = [
            self.head_shape, self.head_colour, self.head_border,
            self.head_symcat, self.class_head, self.aux_head,
        ]
        for head in heads:
            nn.init.kaiming_normal_(head.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(head.bias)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, models.resnet.BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extracts global features, foveated interior features, and STN transformation:
        Returns:
            h_global: [B, 512]
            h_interior: [B, 512]
            feat4_rectified: [B, 512, 4, 4]
            theta: [B, 2, 3]
        """
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual Stages
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)  # [B, 256, 8, 8]
        x4 = self.layer4(x3)  # [B, 512, 4, 4]

        # Global features
        h_global = self.avgpool(x4).flatten(1)  # [B, 512]

        # STN Localization
        stn_feat = self.stn_loc(x3).flatten(1)  # [B, 128]
        theta = self.stn_fc(stn_feat).view(-1, 2, 3)  # [B, 2, 3]

        # Rectify x4 spatial grid
        grid = F.affine_grid(theta, x4.size(), align_corners=False)
        feat4_rectified = F.grid_sample(x4, grid, align_corners=False)  # [B, 512, 4, 4]

        # Foveated Interior ROI Crop:
        # Central 50% spatial region on 4x4 grid corresponds to indices [1:3, 1:3] (2x2)
        interior_patch = feat4_rectified[:, :, 1:3, 1:3]
        h_interior = F.adaptive_avg_pool2d(interior_patch, (1, 1)).flatten(1)  # [B, 512]

        return h_global, h_interior, feat4_rectified, theta

    def predict_concepts(self, h_global: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predicts concept logits and concatenated probability vector c_hat."""
        logits_shape = self.head_shape(h_global)
        logits_colour = self.head_colour(h_global)
        logits_border = self.head_border(h_global)
        logits_symcat = self.head_symcat(h_global)

        p_shape = F.softmax(logits_shape, dim=-1)
        p_colour = F.softmax(logits_colour, dim=-1)
        p_border = F.softmax(logits_border, dim=-1)
        p_symcat = F.softmax(logits_symcat, dim=-1)

        c_hat = torch.cat([p_shape, p_colour, p_border, p_symcat], dim=-1)
        return logits_shape, logits_colour, logits_border, logits_symcat, c_hat

    def forward(
        self,
        x: torch.Tensor,
        c_oracle: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with foveated glyph branch and counterfactual intervention support."""
        h_global, h_interior, _, theta = self.extract_features(x)
        logits_shape, logits_colour, logits_border, logits_symcat, c_hat = self.predict_concepts(h_global)

        c_effective = c_oracle if c_oracle is not None else c_hat

        # Fuse global + foveated interior features for auxiliary glyph branch
        h_aux_fused = torch.cat([h_global, h_interior], dim=-1)  # [B, 1024]
        logits_aux = self.aux_head(h_aux_fused)  # [B, 43]

        if self.bottleneck_mode == "pure":
            logits_class = self.class_head(c_effective)
        elif self.bottleneck_mode == "concat":
            logits_class = self.class_head(c_effective) + logits_aux
        else:  # residual (default)
            logits_class = self.class_head(c_effective) + logits_aux

        return {
            "logits": logits_class,
            "c_hat": c_hat,
            "logits_shape": logits_shape,
            "logits_colour": logits_colour,
            "logits_border": logits_border,
            "logits_symcat": logits_symcat,
            "logits_aux": logits_aux,
            "h_global": h_global,
            "h_interior": h_interior,
            "theta": theta,
        }

    def intervene(self, x: torch.Tensor, intervened_concepts: torch.Tensor) -> torch.Tensor:
        """Applies concept intervention counterfactual override."""
        out = self.forward(x, c_oracle=intervened_concepts)
        return out["logits"]


class FoveatedCBMLoss(nn.Module):
    """
    Multi-Task Loss for Foveated Spatial-Transformer CBM with Within-Group Contrastive Margin.

    L_total = L_task(y, y_hat)
            + lambda_c * sum(L_concept_k)
            + lambda_aux * L_aux(y, y_aux)
            + lambda_wg * L_within_group(h_interior)
    """

    def __init__(
        self,
        lambda_c: float = 1.0,
        lambda_aux: float = 0.5,
        lambda_wg: float = 0.2,
        wg_margin: float = 0.20,
    ) -> None:
        super().__init__()
        self.lambda_c = lambda_c
        self.lambda_aux = lambda_aux
        self.lambda_wg = lambda_wg
        self.wg_margin = wg_margin
        self.ce = nn.CrossEntropyLoss()

    def compute_within_group_loss(
        self,
        h_interior: torch.Tensor,
        target_class: torch.Tensor,
        target_shape: torch.Tensor,
        target_colour: torch.Tensor,
        target_border: torch.Tensor,
        target_symcat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes contrastive margin penalty between sample pairs that share identical
        concept signatures C(y_i) == C(y_j) but have different class labels y_i != y_j.
        Forces the interior features to maximize separation between speed-limit numerals.
        """
        B = h_interior.size(0)
        if B < 2:
            return torch.tensor(0.0, device=h_interior.device)

        # Normalize interior feature representations
        z = F.normalize(h_interior, p=2, dim=-1)  # [B, 512]
        sim_matrix = torch.matmul(z, z.t())  # [B, B]

        # Pairwise condition masks
        same_shape = (target_shape.unsqueeze(1) == target_shape.unsqueeze(0))
        same_colour = (target_colour.unsqueeze(1) == target_colour.unsqueeze(0))
        same_border = (target_border.unsqueeze(1) == target_border.unsqueeze(0))
        same_symcat = (target_symcat.unsqueeze(1) == target_symcat.unsqueeze(0))

        same_concept = same_shape & same_colour & same_border & same_symcat
        diff_class = (target_class.unsqueeze(1) != target_class.unsqueeze(0))

        # Upper triangular mask (exclude self-comparison and duplicates)
        triu_mask = torch.triu(torch.ones(B, B, dtype=torch.bool, device=h_interior.device), diagonal=1)
        wg_pairs_mask = same_concept & diff_class & triu_mask

        num_pairs = wg_pairs_mask.sum()
        if num_pairs == 0:
            return torch.tensor(0.0, device=h_interior.device)

        # Penalize cosine similarities exceeding margin: max(0, sim - margin)^2
        wg_sims = sim_matrix[wg_pairs_mask]
        penalties = F.relu(wg_sims - self.wg_margin).pow(2)
        return penalties.mean()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        target_class: torch.Tensor,
        target_shape: torch.Tensor,
        target_colour: torch.Tensor,
        target_border: torch.Tensor,
        target_symcat: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_task = self.ce(outputs["logits"], target_class)
        loss_shape = self.ce(outputs["logits_shape"], target_shape)
        loss_colour = self.ce(outputs["logits_colour"], target_colour)
        loss_border = self.ce(outputs["logits_border"], target_border)
        loss_symcat = self.ce(outputs["logits_symcat"], target_symcat)
        loss_concepts = loss_shape + loss_colour + loss_border + loss_symcat

        loss_aux = self.ce(outputs["logits_aux"], target_class)

        loss_wg = self.compute_within_group_loss(
            outputs["h_interior"],
            target_class,
            target_shape,
            target_colour,
            target_border,
            target_symcat,
        )

        total_loss = (
            loss_task
            + self.lambda_c * loss_concepts
            + self.lambda_aux * loss_aux
            + self.lambda_wg * loss_wg
        )

        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_task": float(loss_task.item()),
            "loss_concepts": float(loss_concepts.item()),
            "loss_shape": float(loss_shape.item()),
            "loss_colour": float(loss_colour.item()),
            "loss_border": float(loss_border.item()),
            "loss_symcat": float(loss_symcat.item()),
            "loss_aux": float(loss_aux.item()),
            "loss_wg": float(loss_wg.item()),
        }
        return total_loss, metrics


def get_parameter_groups_foveated(
    model: nn.Module,
    weight_decay: float = 1e-4,
) -> List[Dict[str, Any]]:
    """
    Partitions parameters for No-Bias-Decay optimization:
    - 2D/4D weights (Conv2d, Linear): weight_decay = 1e-4
    - 1D biases and BN scale/shift: weight_decay = 0.0
    """
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
