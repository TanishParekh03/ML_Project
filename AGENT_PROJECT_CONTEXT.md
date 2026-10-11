# AI Agent Context: Robust and Correct Traffic Sign Recognition

## Purpose of This File

This document is the authoritative working context for AI agents assisting with this research project. Read it before proposing experiments, changing data pipelines, implementing models, or interpreting results.

The goal is to keep every agent aligned with the same scientific question, fixed experimental controls, and current project decisions. If a later user instruction conflicts with this document, follow the user’s latest instruction and update this file to reflect the new decision.

---

## 1. Project at a Glance

### Working title

**Data-Centric, Representation-Centric, and Hybrid Robustness for Correct Traffic Sign Recognition**

### Research setting

- **Task:** multi-class traffic sign recognition.
- **Primary dataset:** German Traffic Sign Recognition Benchmark (**GTSRB**) only.
- **Target output:** a conference-style research paper.
- **Primary goals:** improve and rigorously evaluate **robustness** and **correctness**.
- **Compute:** free Google Colab with a T4 GPU.
- **Team:** three members. One member owns dataset and experimental standardisation for the whole project.

### Central scientific question

> Is traffic-sign robustness improved more effectively by changing the visual variation a CNN sees during training, by forcing it to learn a structured semantic representation, or by combining both approaches?

The project is not simply “apply a CBM to GTSRB” and not simply “improve accuracy with augmentation.” It is a controlled comparison of two robustness philosophies:

```text
Data-centric approach:            Representation-centric approach:
change what the model sees        change what the model must represent

                \                 /
                 \               /
                  Hybrid approach
```

---

## 2. Research Rationale

Standard CNNs can perform well on clean images while relying on unstable features such as texture, background correlations, or high-frequency artefacts. Their accuracy can decline under blur, noise, fog, lighting shifts, weather effects, and related distribution changes.

This project studies two complementary mechanisms:

1. **Data-centric robustness** exposes a CNN to diverse, label-preserving training conditions.
2. **Representation-centric robustness** requires a CNN to predict meaningful traffic-sign concepts before making the final class decision.

The representation-centric mechanism is a **Concept Bottleneck Model (CBM)**. Rather than directly mapping an image to a class, it predicts a vector of sign characteristics, then maps that vector to the class. This supports analysis of whether an error arose from visual perception, the concept vocabulary, or the final mapping.

The hybrid approach tests whether data diversity and structured semantics are complementary.

---

## 3. Scope: Included and Excluded

### In scope

- GTSRB classification.
- Standard CNN baseline and data-centric training interventions.
- Standards-grounded class-level concept labels.
- Sequential and joint CBM variants.
- Oracle-CBM analysis.
- A hybrid of selected data-centric intervention(s) and the CBM.
- Held-out clean and corruption evaluation.
- Class correctness, concept correctness, consistency, calibration, intervention, and compute reporting.
- Reproducibility through fixed splits, preprocessing, data versions, seeds, and experiment logs.

### Out of scope for now

- External/OOD datasets beyond GTSRB.
- Vision Transformer (ViT) comparison. It may be reconsidered only if the team later decides it is needed.
- CLIP/VLM baselines.
- Manual image-level annotation of every GTSRB sample.
- Training directly on the final corruption test suite.
- Claims of autonomous-driving deployment readiness or legal certification.

Do not expand the scope without explicit user approval.

---

## 4. Settled Decisions

| Topic | Current decision |
| --- | --- |
| Primary dataset | GTSRB only |
| Dataset labels | Original class labels plus class-level concept labels |
| Concept label source | Vienna Convention review, then mapping to signs present in GTSRB and their German implementation |
| Annotation granularity | One concept vector per class, verified on a sample of images per class |
| Initial concept families | Shape, colour profile, border, symbol category; exact values are not yet frozen |
| Baseline for initial runs | ResNet-18, due to Colab T4 constraints |
| Larger final model | ResNet-50 may be used only for selected final experiments if justified |
| ViT | Not included for now |
| Corruption evaluation | Custom, reproducible suite based on `corruption_code.py` or a repository to be provided |
| Corruption data during training | Not used as stored test-corruption data; training transformations are online only |
| Paper style | Conference-style paper |
| Main outcome | Robustness and correctness, not clean accuracy alone |

---

## 5. Key Definitions

### Robustness

Performance stability when a clean traffic-sign image is altered by a visual corruption that should not change its class, such as blur, noise, fog, or illumination changes. Robustness must be measured on held-out corruptions and severity levels.

### Correctness

Correctness has several layers in this project:

1. **Class correctness:** the final traffic-sign class is correct.
2. **Concept correctness:** predicted concepts match the standards-grounded concept vector.
3. **Semantic consistency:** predicted concepts and final class do not violate documented traffic-sign constraints.
4. **Intervention correctness:** correcting an erroneous concept can repair the final decision when the concept representation is sufficient.

### Data-centric robustness

Methods that alter training inputs, while preserving the class label. Expected components include AugMix, frequency-aware transformations, and texture/shape-bias interventions.

### Representation-centric robustness

Methods that constrain the model’s internal decision path through explicit concepts. The core implementation is a Concept Bottleneck Model (CBM).

### Oracle CBM

The class head is given ground-truth concepts rather than predicted concepts. This reveals whether CBM failures are due to:

- poor visual concept prediction, or
- an insufficient concept vocabulary / concept-to-class mapping.

---

## 6. Intended Model Families

### A. Standard black-box CNN baseline

```text
Image -> ResNet-18 backbone -> class logits -> traffic-sign class
```

Purpose: establish the clean, corrupted, calibration, efficiency, and error-analysis reference.

### B. Data-centric CNN variants

```text
Image -> training-time intervention -> ResNet-18 -> class
```

Candidate interventions:

- controlled label-preserving augmentation;
- AugMix;
- low-/high-frequency processing or perturbation;
- texture/shape-bias mitigation;
- carefully justified combinations of the above.

Each component must first be evaluated independently. Only promising methods should be combined.

### C. Sequential CBM

```text
Image -> concept predictor -> predicted concept vector -> class predictor -> class
```

The concept predictor may use the same ResNet-family backbone. The class predictor should be simple and transparent where possible (for example, a linear classifier or clearly documented rule/constraint-aware mapping).

### D. Joint CBM

```text
Image -> shared CNN backbone -> concepts and class predictions
```

Use a weighted task and concept loss such as:

$$
\mathcal{L}=\mathcal{L}_{task}(y,\hat{y}) + \lambda\sum_i \mathcal{L}_{concept}(c_i,\hat{c}_i)
$$

The concept-loss formulation must match the final ontology: each concept family may be a separate multi-class classification task rather than one unrestricted multi-label field.

### E. Constraint-aware CBM

Use documented standards-derived rules to detect or penalise implausible concept/class combinations. Do not invent rules casually. Each rule must be traceable to the frozen ontology and its source/justification.

### F. Hybrid model

```text
Training-time data-centric intervention
                -> CNN backbone
                -> concepts
                -> optional consistency layer
                -> class
```

The hybrid should combine the CBM with only the strongest data-centric intervention(s), as shown by prior ablations.

---

## 7. Dataset Governance: Non-Negotiable Rules

One member is the **Dataset and Experimental Standardisation Lead**. This role maintains the master dataset and prevents experimental confounding.

### The master dataset must contain

```text
images/                 # raw or canonical clean images; do not overwrite originals
metadata/
  train.csv             # frozen IDs and labels
  validation.csv
  test.csv
concepts/
  class_concepts.csv    # one concept record per GTSRB class
  annotation_guidelines.md
  ontology.md
configs/
  preprocessing.yaml
  dataset_version.md
```

Recommended metadata fields:

```text
image_id, image_path, class_id, class_name, split,
shape, colour_profile, border_type, symbol_category,
annotation_source, annotator, annotation_confidence
```

### Fixed rules

1. All methods use the same source images, class labels, train/validation/test split, base preprocessing, and evaluation protocol.
2. No member independently regenerates data splits.
3. The master clean dataset is never overwritten by method-specific transformed images.
4. AugMix, frequency processing, texture interventions, and similar techniques run only in the relevant training pipeline.
5. Validation data receives deterministic base preprocessing only.
6. Clean test data receives deterministic base preprocessing only.
7. Corruption test data is generated/evaluated from held-out test images only.
8. Do not apply a transformation that changes a sign’s semantic class unless an explicit class-aware policy allows it.
9. Freeze and version every change to data, concepts, preprocessing, or splits.
10. All validated data and code modules are shared project resources, not private member assets.

### Transformation safety

Do not blindly use common image augmentations. Horizontal flip, large rotation, crop, and perspective changes can alter the meaning or visibility of some traffic signs. Each transformation needs a documented rationale and parameter range. A class-aware allowlist is preferred.

---

## 8. Concept Ontology and Annotation Plan

### Status: not yet frozen

The exact ontology must be created after reviewing the Vienna Convention and identifying which sign classes occur in GTSRB. The ontology must be grounded in the standard and mapped to German implementation where necessary.

### Planned workflow

1. List all GTSRB classes and class names.
2. Review relevant sign design characteristics from the Vienna Convention.
3. Identify where German implementation differs or is more specific.
4. Define a small, controlled vocabulary that can distinguish relevant classes without unnecessary complexity.
5. Create `class_concepts.csv`: one record per GTSRB class.
6. Write annotation guidelines with definitions, edge cases, source references, and examples.
7. Verify the mapping by inspecting a representative sample of clean images from each class.
8. Record provenance, annotator, and confidence for each class mapping.

### Important framing

The project should say that concepts are **derived from** or **grounded in** traffic-sign standards. Avoid saying every concept value is legally mandated. The Vienna Convention provides conventions; national implementation and the project’s machine-readable ontology must be distinguished.

### Why class-level labels are appropriate

For clean GTSRB images, an image’s semantic sign properties should normally follow its class. Class-level labels are much more feasible and reproducible than manually labelling every image. A sampled visual audit detects unusual cases. Corrupted images retain the original class-level ground-truth concepts; corruption tests whether the model still recovers those concepts.

---

## 9. Corruption Evaluation Plan

### Status: pending input

The team has `corruption_code.py`, or may provide the repository from which it came. An agent must inspect this source before finalising the corruption protocol.

### Required protocol decisions after inspection

- Exact corruption types.
- Severity levels.
- Parameter settings and random seeds.
- Whether corruptions are generated once and cached or generated deterministically at evaluation time.
- Image-ID mapping between clean and corrupted samples.
- Output directory and metadata format.
- Whether all methods are evaluated on exactly the same generated corruptions.

### Non-negotiable fairness rule

The corruption suite is held out for evaluation. Do not tune repeatedly to individual corruption results without recording the tuning process. Do not use the exact evaluation corruptions as training examples.

### Expected reporting

Report clean results and corruption results by:

- model;
- corruption type;
- severity;
- class accuracy/macro-F1;
- concept F1 for CBM models;
- mean and variation across fixed training seeds.

---

## 10. Experimental Protocol

### Required controls

Before serious runs, freeze a shared experimental contract containing:

- dataset version and split files;
- input resolution and aspect-ratio policy;
- RGB conversion and normalisation;
- backbone version and pretrained/non-pretrained policy;
- batch size, optimiser, learning-rate schedule, epochs, and early-stopping policy;
- fixed training seeds and number of repetitions;
- hardware/runtime logging;
- corruption generation configuration;
- evaluation scripts and reporting format.

### Recommended execution order

1. Build and validate the master GTSRB dataset.
2. Freeze the ontology and construct class-level concept labels.
3. Implement the clean ResNet-18 baseline.
4. Establish clean-test reproducibility and baseline metrics.
5. Inspect/freeze the corruption suite and run baseline corruption evaluation.
6. Run data-centric methods as individual ablations.
7. Build sequential, joint, and oracle CBMs.
8. Run concept, consistency, and intervention analyses.
9. Select the strongest data-centric component(s) and construct the hybrid.
10. Perform final multi-seed comparison, error analysis, and paper figures/tables.
11. Consider ResNet-50 final confirmation only if the T4 budget supports it and the result is warranted.

### Do not do the following

- Start hybrid experiments before baseline and individual ablations exist.
- Compare models trained with different splits or different base preprocessing.
- Report a single best run without seed variation.
- Claim robustness from clean accuracy alone.
- Claim interpretability from Grad-CAM alone.
- Change the ontology after seeing model results without versioning and rerunning affected experiments.

---

## 11. Evaluation Metrics

| Dimension | Required measurements |
| --- | --- |
| Clean class performance | Accuracy, macro-F1, per-class results where useful |
| Corruption robustness | Accuracy/macro-F1 by corruption and severity; mean corruption score; relative performance drop |
| Concept quality | Per-family concept accuracy and macro-F1 on clean and corrupted data |
| Calibration | Expected Calibration Error (ECE); optionally reliability diagrams |
| Intervention | Accuracy before/after correcting concepts; single-concept versus all-incorrect-concepts correction |
| Consistency | Rate of invalid concepts or incompatible concept-to-class outputs |
| Oracle CBM | Class performance using ground-truth concepts |
| Efficiency | Parameter count, FLOPs where feasible, inference latency, training time |
| Statistical reliability | Results across fixed seeds: mean, standard deviation, and appropriate paired comparisons |

### Intervention recovery rate

Where useful, report:

$$
IR=\frac{Acc_{after\ intervention}-Acc_{before\ intervention}}
{1-Acc_{before\ intervention}}
$$

This is the proportion of initially incorrect predictions recovered after concept correction.

---

## 12. Team Responsibilities

The three research tracks are conceptually distinct, but data and validated code remain shared.

| Track | Owner | Primary question | Deliverables |
| --- | --- | --- | --- |
| Dataset & standardisation | Designated lead | Are comparisons fair and reproducible? | Master data, frozen splits, metadata, ontology, annotations, data documentation |
| Data-centric robustness | Member 1 | Does changing training input variation improve robustness? | AugMix/frequency/texture modules, ablations, results |
| Representation-centric robustness | Member 2 | Does forcing semantic concepts improve correctness and robustness? | CBMs, oracle analysis, constraints, intervention study |
| Hybrid & overall comparison | Member 3 | Are the approaches complementary? | Hybrid model, final comparison, compute/statistical analysis |

If there are only three people, one person may hold both the dataset-standardisation responsibility and one modelling track. This must not change the shared protocol.

---

## 13. Expected Paper Narrative

The intended paper story is:

1. CNNs can be accurate yet fragile and opaque in safety-relevant TSR.
2. Data-centric and representation-centric methods both aim to reduce reliance on unstable features, but act differently.
3. A controlled GTSRB study compares a standard CNN, data-centric variants, CBM variants, and a hybrid under identical data and evaluation conditions.
4. Class correctness alone is not enough: concept quality, semantic consistency, calibration, and intervention recoverability are also assessed.
5. The results determine whether training diversity, structured semantic reasoning, or their combination offers the strongest robustness/correctness trade-off.

### Contribution statement

> This work provides a controlled comparison of training-data diversity and standards-grounded semantic bottlenecks for traffic-sign recognition. It evaluates an intervention-capable CBM and hybrid CNN pipeline using clean accuracy, held-out corruption robustness, concept correctness, semantic consistency, calibration, and computational cost under one reproducible protocol.

---

## 14. Confirmed Decisions & Resolved Parameters

1. **Class-to-Concept Ontology:** Frozen in `data/master_v1/concepts/class_concepts.csv` (43 classes, 5 dimensions: Shape, Colour Profile, Border Type, Symbol Category, Symbol Content).
2. **Splits & Leakage Policy:** Stratified Group 80/20 train/val split by `ClassId + TrackId` (0 track leakage); frozen in `data/master_v1/metadata/`. Test split (12,630) preserved untouched.
3. **Preprocessing Contract:** `v1_roi_margin10_letterbox64`: 10% context margin ROI crop $\to$ aspect-preserving bilinear scale $\to$ centered zero-pad to $64 \times 64 \times 3 \to$ ImageNet normalization.
4. **Model Backbone:** ResNet-18 with ImageNet pretrained initialization (`ResNet18_Weights.DEFAULT`), Kaiming Normal head initialization, and Zero-$\gamma$ residual BN initialization.
5. **Optimization Protocol:** SGD with Nesterov momentum $\mu=0.9$, No-Bias-Decay rule ($\lambda=10^{-4}$ on 2D weights, $\lambda=0$ on biases/BN), 3-epoch linear warmup + Cosine Annealing decay down to $10^{-5}$, mixed precision (AMP).
6. **Held-out Corruption Benchmark:** 15 Hendrycks corruptions across 4 categories (Noise, Blur, Weather, Digital) $\times$ 5 severities = 75 evaluation points. Evaluates clean Acc, clean Macro-F1, clean ECE, $mCA$, and $RPD$.
7. **Empirical Ladder Status:**
   - Exp 1 (Baseline ResNet-18): Clean Acc $97.91\%$, F1 $97.07\%$, ECE $0.0082$, mCA $83.88\%$, RPD $14.33\%$.
   - Exp 2 (AugMix ResNet-18): Clean Acc $98.47\%$, F1 $97.78\%$, ECE $0.0071$, mCA $83.10\%$, RPD $15.61\%$.
   - Exp 3 (Frequency ResNet-18): Clean Acc $97.47\%$, F1 $95.68\%$, ECE $0.0127$, mCA $83.20\%$, RPD $14.64\%$.
   - Exp 4 (Concept Bottleneck Model): Clean Acc **97.59%**, F1 **96.75%**, ECE **0.0114**, mCA **84.29%** (project record), RPD **13.62%** (lowest degradation).
   - Exp 5 (Hybrid AugMix + CBM): Clean Acc **97.61%**, F1 **96.53%**, ECE **0.0127**, mCA **81.63%**, RPD **16.37%**. Falsified naive additive complementarity ($H_3$): revealed Representation-Augmentation Interference where AugMix color-destructive transformations (`solarize`, `posterize`) conflict with physical concept consistency (`colour_profile`).
7. **Empirical Ladder Status:**
   - Exp 1 (Baseline ResNet-18): Clean Acc $97.91\%$, F1 $97.07\%$, ECE $0.0082$, mCA $83.88\%$, RPD $14.33\%$.
   - Exp 2 (AugMix ResNet-18): Clean Acc $98.47\%$, F1 $97.78\%$, ECE $0.0071$, mCA $83.10\%$, RPD $15.61\%$.
   - Exp 3 (Frequency ResNet-18): Clean Acc $97.47\%$, F1 $95.68\%$, ECE $0.0127$, mCA $83.20\%$, RPD $14.64\%$.
   - Exp 4 (Concept Bottleneck Model): Clean Acc **97.59%**, F1 **96.75%**, ECE **0.0114**, mCA **84.29%**, RPD **13.62%**.
   - Exp 5 (Hybrid AugMix + CBM): Clean Acc **97.61%**, F1 **96.53%**, ECE **0.0127**, mCA **81.63%**, RPD **16.37%**. Falsified naive additive complementarity ($H_3$): revealed Representation-Augmentation Interference where AugMix color-destructive transformations (`solarize`, `posterize`) conflict with physical concept consistency (`colour_profile`).
   - Exp 6 (Frequency-Aware CBM): Clean Acc **97.45%**, F1 **96.60%**, mCA **84.21%**, RPD **13.59%**. Confirmed that color-preserving spectral regularization avoids the destructive interference of AugMix, but proved that training-time tricks saturate at $\sim 84.2\%$ mCA.
8. **Concept Degradation Rate (CDR) Analysis:**
   - Shape is the most robust primitive (11.2% CDR at Sev 5), while Symbol Category is the most fragile (20.2% CDR). Confirms hierarchical semantic stability.
9. **Error Mode & Ontology Diagnostic:**
   - Audited 46,770 misclassifications: 53.3% within-group vs. 46.7% cross-group.
   - 66.3% of all within-group errors concentrate in speed-limit numerals (16,527 errors), confirming the 25.58% oracle Bayes ceiling of a 4-axis ontology.
   - Falsified Ontology-Consistent Decoding (OCS): +0.06% gain, AUROC 0.852 vs 0.879 for MSP due to synchronous failure of shared backbone.
10. **Test-Time Normalization Adaptation (BN-Adapt) Breakthrough:**
    - Full 75-condition sweep across 3 seeds (42, 43, 44):
      - Clean Accuracy: 97.59% (Static) $\to$ **97.59% (Shift-Gated, zero harm)**.
      - 75-point mCA: **84.25% $\to$ 88.50% (+4.25 percentage points raw boost)**.
      - Severe noise leaps: Fog Sev 5 (+35.91%), Impulse Noise Sev 5 (+26.65%), Shot Noise Sev 5 (+26.63%), Gaussian Noise Sev 5 (+21.03%).
      - Deployment stream sweep (Gaussian Sev 5): $N=1$ (49.87%), $N=8$ (**66.73%**, +16.86 points with source momentum), $N=128$ (**71.03%**).

---

## 15. Agent Operating Instructions

When assisting this project, an AI agent should:

1. Preserve the current scope unless the user explicitly expands it.
2. Treat the master dataset and fixed protocol as shared scientific controls.
3. Separate confirmed decisions from assumptions and unresolved choices.
4. Recommend the smallest experiment that answers the question before proposing expensive runs.
5. Prefer ablations and multi-seed comparisons over untraceable “best model” claims.
6. Flag transformations that may change traffic-sign semantics.
7. Keep data preparation separate from training-time method-specific transformations.
8. Do not present the Vienna Convention as a complete ML ontology without the documented mapping work.
9. Do not claim legal compliance, autonomous-driving safety certification, or real-world deployment readiness from benchmark experiments.
10. Record configurations, versions, random seeds, and evaluation outputs so results can be reproduced.
11. Before modifying experiment-critical code or metadata, inspect existing files and preserve unrelated user work.
12. Update this document when a material design decision is finalised.

## 16. Immediate Next Steps

1. Run the Full 75-Condition Benchmark on Exp 1 (Baseline ResNet-18) across seeds 42, 43, 44 to test whether CBM retains an advantage post-BN-adaptation ($H_3$ complementarity).
2. Compare Baseline vs. CBM side-by-side to finalize the conference paper manuscript.


---

## 17. Companion Documents

- `Structured_Research_Proposal.md`: structured research proposal.
- `Research_context.md`: original conversation/context; it is historical reference, not the governing protocol.
- `AGENT_PROJECT_CONTEXT.md` (this file): current operating context for agents.

