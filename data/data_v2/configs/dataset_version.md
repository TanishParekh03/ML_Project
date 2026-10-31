# BelgiumTSC Master Dataset Specification and Version Record (data_v2)

- **Creation Timestamp (UTC):** 2026-10-31T07:54:49.125328+00:00
- **Raw Data Source:** `D:\ML_Project\belgium`
- **Master Random Seed:** `42`
- **Split Protocol:** Stratified Group Split by `(ClassId + PoleId)`
- **Preprocessing Version:** `v1_roi_margin10_letterbox64`
  - ROI Crop: 10% contextual margin on all sides (clamped to image dimensions).
  - Resizing: Proportional bilinear scaling preserving aspect ratio.
  - Padding: Symmetric zero-padding to 64x64 black canvas.

## Dataset Partition Summary

| Split | Number of Images | Description |
| :--- | :--- | :--- |
| **Train** | `3,644` | 80% of physical poles per class for training |
| **Validation** | `931` | 20% of physical poles per class (zero physical pole leakage) |
| **Test** | `2,534` | Official BelgiumTSC held-out test benchmark (strictly untouched) |
| **Total** | `7,109` | Total annotated samples across 62 classes |

## Manifest SHA-256 Checksums

```text
train.csv:           6ad20ddba9fb8215e7284e42cc14a6eb688d7a4ac5045d5fe52639edfbb6615a
val.csv:             9733e4278d96fb69accf4de0da4ca7dafc769789f2b8d8700d3e43e386bf5be5
test.csv:            5ac50cb3f0e8d14a644d538ed46498d76a887a0203d4e2d7f05008850e0a5af2
master_metadata.csv: 33adf54ba32c71a89ab7b0af3926758175d8316343ba4f4c31a6cea987df52fc
class_concepts.csv:  2ff54f666e207885d3c934cf84f0e338c28b6e9b2dc5d68d7b93346705105f3e
```
