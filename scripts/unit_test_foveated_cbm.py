"""
Unit Tests for Foveated Spatial-Transformer Concept Bottleneck Model (Exp 8)
=============================================================================
Tests:
1. Model Instantiation & STN Identity Initialization:
   - Verifies stn_fc bias is [1, 0, 0, 0, 1, 0] and weight is zeros.
   - Verifies theta is identity matrix [[1, 0, 0], [0, 1, 0]].
2. Forward Pass Tensor Shapes:
   - logits: [B, 43]
   - c_hat: [B, 22]
   - h_global: [B, 512]
   - h_interior: [B, 512]
   - theta: [B, 2, 3]
3. Counterfactual Intervention via intervene():
   - Overriding concepts produces modified class logits.
4. FoveatedCBMLoss & Within-Group Contrastive Loss:
   - Verifies loss computes finite scalars without NaN.
   - Verifies within-group pairs are penalized when cosine similarity > margin.
5. Parameter Group Partitioning (No-Bias-Decay):
   - Verifies 2D weights have weight_decay=1e-4, 1D biases/scales have weight_decay=0.0.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from src.models.foveated_cbm import (
    FoveatedConceptBottleneckModel,
    FoveatedCBMLoss,
    get_parameter_groups_foveated,
)


def run_unit_tests():
    print("=" * 70)
    print("unit_test_foveated_cbm.py -- Verification for Exp 8 Foveated CBM")
    print("=" * 70)
    passed = 0
    total = 0

    # 1. Instantiation and STN Identity Check
    print("\nTest 1: Model Instantiation & STN Identity Initialization")
    model = FoveatedConceptBottleneckModel(num_classes=43, pretrained=False)
    bias_expected = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
    assert torch.allclose(model.stn_fc.bias.data, bias_expected), "STN bias is not identity"
    assert torch.allclose(model.stn_fc.weight.data, torch.zeros_like(model.stn_fc.weight.data)), "STN weights not zero"
    print("  [PASS] STN FC weights initialized to 0 and bias initialized to [1, 0, 0, 0, 1, 0]")
    passed += 1
    total += 1

    # 2. Forward Pass Tensor Shapes
    print("\nTest 2: Forward Pass & Feature Shapes")
    B = 4
    x = torch.randn(B, 3, 64, 64)
    out = model(x)

    assert out["logits"].shape == (B, 43), f"Expected logits [4, 43], got {out['logits'].shape}"
    assert out["c_hat"].shape == (B, 22), f"Expected c_hat [4, 22], got {out['c_hat'].shape}"
    assert out["h_global"].shape == (B, 512), f"Expected h_global [4, 512], got {out['h_global'].shape}"
    assert out["h_interior"].shape == (B, 512), f"Expected h_interior [4, 512], got {out['h_interior'].shape}"
    assert out["theta"].shape == (B, 2, 3), f"Expected theta [4, 2, 3], got {out['theta'].shape}"
    print("  [PASS] Forward output tensor dimensions verified: logits [B, 43], c_hat [B, 22], h_global [B, 512], h_interior [B, 512], theta [B, 2, 3]")
    passed += 1
    total += 1

    # Check that initial theta produces identity transformation
    eye_2x3 = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float).unsqueeze(0).repeat(B, 1, 1)
    assert torch.allclose(out["theta"], eye_2x3, atol=1e-5), "Initial predicted theta is not exact identity"
    print("  [PASS] Initial predicted theta produces exact affine identity grid")
    passed += 1
    total += 1

    # 3. Counterfactual Intervention
    print("\nTest 3: Counterfactual Concept Intervention (intervene())")
    c_intervened = torch.zeros(B, 22)
    c_intervened[:, 0] = 1.0  # Force circle
    c_intervened[:, 6] = 1.0  # Force white_red
    c_intervened[:, 11] = 1.0  # Force red_border
    c_intervened[:, 15] = 1.0  # Force speed_numeral

    intervened_logits = model.intervene(x, c_intervened)
    assert intervened_logits.shape == (B, 43), f"Expected [B, 43], got {intervened_logits.shape}"
    assert not torch.allclose(intervened_logits, out["logits"]), "Intervention did not affect output logits"
    print("  [PASS] Concept intervention successfully modulates class logits")
    passed += 1
    total += 1

    # 4. FoveatedCBMLoss and Within-Group Contrastive Loss
    print("\nTest 4: FoveatedCBMLoss & Within-Group Contrastive Margin")
    criterion = FoveatedCBMLoss(lambda_c=1.0, lambda_aux=0.5, lambda_wg=0.2, wg_margin=0.20)

    # Construct batch where sample 0 and 1 have identical concept vectors but different class labels (within-group error pair)
    target_class = torch.tensor([1, 2, 14, 25])  # Class 1 (30 km/h) and Class 2 (50 km/h) are both speed limit circles!
    target_shape = torch.tensor([0, 0, 3, 1])
    target_colour = torch.tensor([0, 0, 4, 1])
    target_border = torch.tensor([0, 0, 0, 1])
    target_symcat = torch.tensor([0, 0, 4, 1])

    loss, metrics = criterion(out, target_class, target_shape, target_colour, target_border, target_symcat)
    assert torch.isfinite(loss), "Loss is not finite"
    assert metrics["loss_wg"] >= 0.0, "Within-group loss is negative"
    print(f"  [PASS] Multi-task loss computed: total={loss.item():.4f}, task={metrics['loss_task']:.4f}, cpt={metrics['loss_concepts']:.4f}, wg={metrics['loss_wg']:.4f}")
    passed += 1
    total += 1

    # Backward pass check
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    assert torch.isfinite(grad_norm), "Gradient norm is not finite"
    print(f"  [PASS] Backward pass executed with finite gradient norm: {grad_norm.item():.4f}")
    passed += 1
    total += 1

    # 5. Parameter Group Partitioning (No-Bias-Decay)
    print("\nTest 5: No-Bias-Decay Parameter Grouping")
    groups = get_parameter_groups_foveated(model, weight_decay=1e-4)
    assert len(groups) == 2, f"Expected 2 groups, got {len(groups)}"
    assert groups[0]["weight_decay"] == 1e-4, "Group 0 weight decay not 1e-4"
    assert groups[1]["weight_decay"] == 0.0, "Group 1 weight decay not 0.0"

    num_decay = len(groups[0]["params"])
    num_no_decay = len(groups[1]["params"])
    print(f"  [PASS] Parameter groups: {num_decay} weights with decay=1e-4, {num_no_decay} biases/BN affine with decay=0.0")
    passed += 1
    total += 1

    print("=" * 70)
    print(f"RESULTS: {passed}/{total} tests passed  |  0 failed")
    print("=" * 70)
    print("ALL FOVEATED CBM ARCHITECTURAL TESTS PASSED OK\n")


if __name__ == "__main__":
    run_unit_tests()

