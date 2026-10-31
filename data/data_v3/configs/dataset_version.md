# Chinese TSRD Master Dataset Specification and Version Record (data_v3)

- **Creation Timestamp (UTC):** 2026-10-31T09:57:36.664634+00:00
- **Raw Data Source:** `D:\ML_Project\chinese`
- **Master Random Seed:** `42`
- **Split Protocol:** Stratified Group Split by `(ClassId + InstanceId)`
- **Preprocessing Version:** `v1_roi_margin10_letterbox64`
  - ROI Crop: 10% contextual margin on all sides (clamped to image dimensions).
  - Resizing: Proportional bilinear scaling preserving aspect ratio.
  - Padding: Symmetric zero-padding to 64x64 black canvas.

## Dataset Partition Summary

| Split | Number of Images | Description |
| :--- | :--- | :--- |
| **Train** | `3,328` | 80% of sign instance groups per class for training |
| **Validation** | `842` | 20% of sign instance groups per class (zero instance/twin leakage) |
| **Test** | `1,994` | Official TSRD held-out test benchmark (strictly untouched) |
| **Total** | `6,164` | Total annotated samples across 58 classes |

## Manifest SHA-256 Checksums

```text
train.csv:           41b0c8948a69ff5cc6dbb553e0c11216c2f414daf36b7c376941c9ac0485dcff
val.csv:             86daec06f01fbdd42c3ac0ef51687154be78c736b872b84daa0d4c7bc5b44298
test.csv:            84fc459e3db774961566c6e8c0a40e4a818b12dc444584b1dd168771bde1d89d
master_metadata.csv: 44321d4781dea74b1b4ea9e0c243caa9a3a0a7d6342b1332c9d28075dd332482
class_concepts.csv:  a168ae31b6572e8208cc6c6b8de5ee6791cb034e4b54b99864b95056f7cea74b
```
