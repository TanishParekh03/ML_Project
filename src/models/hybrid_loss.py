"""
Hybrid CBM Loss: Combines Supervised Task Cross-Entropy, Multi-Axis Concept Cross-Entropy,
and Dual-Level Jensen-Shannon Divergence Consistency (Task Level + Concept Level).

Methodological Grounding:
1. Hendrycks et al. (ICLR 2020) AugMix: Jensen-Shannon consistency across triplet views.
2. Koh et al. (ICML 2020) Concept Bottleneck Models: Multi-task semantic bottleneck supervision.

Mathematical Formulation:
L_total = L_clean + lambda_jsd_task * JSD(y_clean, y_aug1, y_aug2)
                  + lambda_jsd_cpt * (1/4) * sum_{k in {sh, col, bor, sym}} JSD(c_clean^(k), c_aug1^(k), c_aug2^(k))

where:
L_clean = L_CE(y_clean, y) + lambda_c * sum_k L_CE(c_clean^(k), c_k) + lambda_aux * L_CE(y_aux, y)
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_jsd(
    logits1: torch.Tensor,
    logits2: torch.Tensor,
    logits3: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Computes symmetric Jensen-Shannon Divergence across 3 distributions.
    JSD(P1, P2, P3) = (1/3) * sum_{i=1}^3 KL(Pi || M) where M = (P1 + P2 + P3) / 3.
    Casts to float32 and clamps probabilities for absolute numerical stability under AMP.
    """
    p1 = F.softmax(logits1.float(), dim=-1)
    p2 = F.softmax(logits2.float(), dim=-1)
    p3 = F.softmax(logits3.float(), dim=-1)

    m = (p1 + p2 + p3) / 3.0
    log_m = torch.log(m.clamp(min=eps))

    kl1 = F.kl_div(log_m, p1, reduction="batchmean")
    kl2 = F.kl_div(log_m, p2, reduction="batchmean")
    kl3 = F.kl_div(log_m, p3, reduction="batchmean")

    return (kl1 + kl2 + kl3) / 3.0


class HybridCBMLoss(nn.Module):
    """
    Dual-Level Jensen-Shannon Consistency + Concept Multi-Task Loss for CBM.

    Parameters
    ----------
    lambda_c : float
        Weight for concept supervision cross-entropy losses (default: 1.0).
    lambda_aux : float
        Weight for auxiliary visual/symbol classification loss (default: 0.5).
    lambda_jsd_task : float
        Weight for task-level JSD consistency across triplet views (default: 12.0).
    lambda_jsd_cpt : float
        Weight for concept-level JSD consistency across triplet views (default: 4.0).
    """

    def __init__(
        self,
        lambda_c: float = 1.0,
        lambda_aux: float = 0.5,
        lambda_jsd_task: float = 12.0,
        lambda_jsd_cpt: float = 4.0,
        decouple_concept_jsd: bool = False,
        zero_colour_jsd: bool = False,
    ) -> None:
        super().__init__()
        self.lambda_c = lambda_c
        self.lambda_aux = lambda_aux
        self.lambda_jsd_task = lambda_jsd_task
        self.lambda_jsd_cpt = lambda_jsd_cpt
        self.decouple_concept_jsd = decouple_concept_jsd
        self.zero_colour_jsd = zero_colour_jsd
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        outputs_clean: Dict[str, torch.Tensor],
        outputs_aug1: Dict[str, torch.Tensor],
        outputs_aug2: Dict[str, torch.Tensor],
        target_class: torch.Tensor,
        target_shape: torch.Tensor,
        target_colour: torch.Tensor,
        target_border: torch.Tensor,
        target_symcat: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # 1. Supervised clean task loss
        loss_task = self.ce(outputs_clean["logits"], target_class)

        # 2. Supervised clean concept losses across 4 axes
        loss_shape = self.ce(outputs_clean["logits_shape"], target_shape)
        loss_colour = self.ce(outputs_clean["logits_colour"], target_colour)
        loss_border = self.ce(outputs_clean["logits_border"], target_border)
        loss_symcat = self.ce(outputs_clean["logits_symcat"], target_symcat)
        loss_concepts = loss_shape + loss_colour + loss_border + loss_symcat

        # 3. Supervised clean auxiliary visual/symbol loss
        loss_aux = self.ce(outputs_clean["logits_aux"], target_class)

        # 4. Clean composite loss
        loss_clean = loss_task + self.lambda_c * loss_concepts + self.lambda_aux * loss_aux

        # 5. Task-level JSD consistency across clean, aug1, aug2
        loss_jsd_task = compute_jsd(
            outputs_clean["logits"],
            outputs_aug1["logits"],
            outputs_aug2["logits"],
        )

        # 6. Concept-level JSD consistency across each concept head
        jsd_shape = compute_jsd(
            outputs_clean["logits_shape"],
            outputs_aug1["logits_shape"],
            outputs_aug2["logits_shape"],
        )
        jsd_colour = compute_jsd(
            outputs_clean["logits_colour"],
            outputs_aug1["logits_colour"],
            outputs_aug2["logits_colour"],
        )
        jsd_border = compute_jsd(
            outputs_clean["logits_border"],
            outputs_aug1["logits_border"],
            outputs_aug2["logits_border"],
        )
        jsd_symcat = compute_jsd(
            outputs_clean["logits_symcat"],
            outputs_aug1["logits_symcat"],
            outputs_aug2["logits_symcat"],
        )
        if self.decouple_concept_jsd:
            loss_jsd_cpt = torch.tensor(0.0, device=loss_task.device, dtype=loss_task.dtype)
        elif self.zero_colour_jsd:
            loss_jsd_cpt = (jsd_shape + jsd_border + jsd_symcat) / 3.0
        else:
            loss_jsd_cpt = (jsd_shape + jsd_colour + jsd_border + jsd_symcat) / 4.0

        # 7. Total multi-scale objective
        total_loss = (
            loss_clean
            + self.lambda_jsd_task * loss_jsd_task
            + self.lambda_jsd_cpt * loss_jsd_cpt
        )

        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_clean": float(loss_clean.item()),
            "loss_task": float(loss_task.item()),
            "loss_concepts": float(loss_concepts.item()),
            "loss_shape": float(loss_shape.item()),
            "loss_colour": float(loss_colour.item()),
            "loss_border": float(loss_border.item()),
            "loss_symcat": float(loss_symcat.item()),
            "loss_aux": float(loss_aux.item()),
            "loss_jsd_task": float(loss_jsd_task.item()),
            "loss_jsd_cpt": float(loss_jsd_cpt.item()),
        }

        return total_loss, metrics

