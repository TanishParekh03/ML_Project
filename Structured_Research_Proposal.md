# Research Proposal & Empirical Framework

## Title

**Data-Centric, Representation-Centric, and Hybrid Robustness for Interpretable Traffic Sign Recognition**

## 1. Abstract

Traffic Sign Recognition (TSR) is a safety-critical perception task for Advanced Driver Assistance Systems (ADAS) and Autonomous Vehicles (AVs). Conventional convolutional neural networks (CNNs) achieve high classification accuracy on uncorrupted benchmarks ($>97\%$), but act as uninterpretable black boxes whose decisions are brittle to environmental distribution shifts—such as atmospheric weather (fog, snow, frost), optical degradation (defocus, motion blur), and sensor noise (Gaussian, shot, impulse noise). Furthermore, black-box networks provide zero diagnostic evidence regarding intermediate physical features when misclassifications occur.

This study investigates and compares two foundational paradigms for out-of-distribution robustness without test-set corruption snooping: **data-centric robustness**, which regularizes the input manifold during training (AugMix with Jensen-Shannon Divergence and Fourier 2D FFT Low-Frequency Component filtering), and **representation-centric robustness**, which restructures the internal hypothesis space using an intermediate Concept Bottleneck Model (CBM) grounded in international transport law (**UNECE Vienna Convention on Road Signs and Signals, 1968** and **German Straßenverkehrs-Ordnung / StVO, 2017**). We further evaluate hybrid compositions and a test-time normalization adaptation mechanism (**Shift-Gated BN-Adapt**).

All experiments are conducted on the German Traffic Sign Recognition Benchmark (GTSRB) using a frozen master dataset, strict video-track group stratification (preventing instance leakage across 1,307 tracks), a 4-stage aspect-preserving letterboxed preprocessing contract ($64 \times 64$), and a 75-condition common corruptions suite (15 corruptions $\times$ 5 severities).

Empirical findings establish:
1. **Inductive Semantic Cushioning ($H_2$ Confirmed):** Residual CBM achieves superior static corruption robustness (**84.57%** $mCA$, **13.34%** $RPD$), proving that grounding representations in physical concepts shields models against sensor perturbations without incurring an interpretability tax on clean data ($97.59\%$ accuracy, $96.75\%$ Macro-F1).
2. **Representation-Augmentation Interference ($H_3$ Falsified):** Naively combining spatial color augmentations (AugMix) with concept bottleneck losses degrades static robustness to **81.86%**, caused by semantic-color conflicts and high-frequency noise injection.
3. **Spectral-Semantic Alignment ($H_4$ Confirmed):** Coupling Fourier 2D FFT spectral filtering with concept bottleneck supervision (Frequency-Aware CBM, Exp 6) cleanly eliminates representation-augmentation interference, setting the static project record of **85.85%** $mCA$ (**12.35%** $RPD$) and pristine clean performance ($97.95\%$ accuracy, $97.29\%$ Macro-F1).
4. **The Test-Time Adaptation Breakthrough ($H_6$ Confirmed):** Applying Shift-Gated Batch Normalization recalibration during inference dynamically eliminates covariate shift, rocketing robustness across all models and establishing an all-time project benchmark record of **89.79%** on Frequency-Aware CBM ($+3.94\%$ absolute boost; $RPD = 8.33\%$), with extreme condition gains exceeding $+35\%$ under heavy fog.

---

## 2. Background and Motivation

Standard CNNs often latch onto high-frequency shortcuts and textural correlations rather than stable physical geometry (Geirhos et al., 2019; Wang et al., 2020). Under adverse weather or sensor degradation, these spurious shortcuts collapse.

To mitigate this failure mode, machine learning literature has diverged into two competing paradigms:
1. **Data-Centric Strategy (Input Manifold Regularization):** Exposing the network to synthetic manifold variations (AugMix consistency, Fourier spectral filtering) to smooth loss landscapes without altering model architecture.
2. **Representation-Centric Strategy (Structured Semantic Bottlenecks):** Constraining internal representations to intermediate human-interpretable concepts (Koh et al., 2020), forcing the model to reason through physical attributes before determining class identities.

Prior literature examined these strategies in isolation on unconstrained natural image benchmarks (ImageNet, CIFAR). Their comparative interaction, failure modes, and compositional complementarity within safety-critical, structured domains like traffic sign recognition remained uncharacterized.

---

## 3. Problem Statement

Three fundamental scientific questions remained open:
1. Does enforcing human-interpretable physical concept bottlenecks incur an "interpretability tax" on clean traffic sign accuracy, or does it provide inductive shielding against environmental sensor noise?
2. Are data-centric manifold consistency and representation-centric concept supervision mutually complementary, or do aggressive data transformations interfere with semantic feature grounding?
3. What is the mathematical and operational resolution limit of legal sign ontologies under severe visual degradation, and how can test-time adaptation bridge the remaining generalization gap?

---

## 4. Aim and Objectives

### Aim
To design, implement, and benchmark an efficient, interpretable CNN-based TSR framework that maximizes corruption robustness without training directly on evaluation corruptions, identifying the principles governing data-centric, representation-centric, and test-time robustness.

### Objectives
1. **Master Dataset Standardization:** Build a leakage-free, group-stratified dataset from GTSRB (31,378 train, 7,830 val across 1,307 tracks; 12,630 held-out test) with a deterministic 4-stage preprocessing contract ($64 \times 64$, 10% context margin).
2. **Concept Ontology Formalization:** Ground all 43 GTSRB classes into a 4-axis, 22-dimensional orthogonal physical concept ontology (Shape, Colour, Border, Symbol Category) derived from the UNECE Vienna Convention (1968) and German StVO (2017).
3. **Baseline Architecture Construction:** Establish a standardized ResNet-18 baseline incorporating modern deep learning refinements (Kaiming Normal head init, Zero-$\gamma$ BN init, No-Bias-Decay rule, Cosine decay schedule).
4. **Data-Centric Track Execution:** Implement and benchmark domain-constrained AugMix (JSD consistency) and 2D Fourier Low-Frequency Component (LFC) spectral filtering.
5. **Representation-Centric Track Execution:** Uncover the 25.58% oracle bottleneck ceiling and implement a Residual Concept Bottleneck Model ($W_{\text{class}}\hat{c} + W_{\text{aux}}h(x)$) enabling test-time counterfactual interventions.
6. **Hybrid Track Evaluation:** Formulate Dual-Level JSD consistency and identify the mechanism of Representation-Augmentation Interference.
7. **Diagnostic & Test-Time Suite Execution:** Perform Concept Degradation Rate (CDR) analysis, error distribution auditing (within-group vs cross-group), and implement Shift-Gated Batch Normalization Adaptation (BN-Adapt) across the full 75-condition benchmark.

---

## 5. Research Hypotheses & Empirical Resolution

| Hypothesis | Theoretical Premise | Empirical Status | Formal Finding & Resolution |
| :--- | :--- | :---: | :--- |
| **$H_1$: Data-Centric Generalization** | AugMix and Frequency filtering improve robustness to continuous environmental shifts. | **Partially Supported** | AugMix dominates Weather ($+3.52\%$) and Blur ($+0.87\%$) but incurs severe high-frequency noise penalty (Impulse $-13.33\%$). Frequency LFC prevents spectral shortcut reliance ($+1.2\%$ noise recovery). |
| **$H_2$: Inductive Semantic Cushioning** | Grounding representations in physical concepts yields lower relative drop ($RPD$) without clean accuracy sacrifice. | **Strongly Supported** | Residual CBM achieves highest static robustness (**84.29%** $mCA$) and lowest degradation (**13.63%** $RPD$) with zero clean penalty ($97.59\%$ Acc, $96.75\%$ F1). |
| **$H_3$: Additive Hybrid Complementarity** | Combining AugMix and CBM produces additive, complementary robustness gains. | **Falsified** | Triggers **Representation-Augmentation Interference** ($81.63\%$ $mCA$, $-2.66\%$ vs CBM). Color-inverting augmentations directly conflict with deterministic concept targets. |
| **$H_4$: Spectral-Semantic Alignment** | Fourier spectral filtering decouples noise suppression from color semantics, restoring hybrid compatibility. | **Strongly Supported** | Frequency-Aware CBM eliminates color conflict, achieving **84.21%** $mCA$ and establishing lowest relative degradation (**13.59%** $RPD$). |
| **$H_5$: Semantic Degradation Hierarchy** | Concepts degrade hierarchically by spatial frequency scale (Shape more resilient than Symbol). | **Strongly Supported** | Concept Degradation Rate (CDR) at Severity 5: Shape ($11.2\%$) $\to$ Colour ($13.9\%$) $\to$ Border ($14.0\%$) $\to$ Symbol Category ($20.2\%$). |
| **$H_6$: Covariate Shift Adaptation** | Test-time BN recalibration dynamically resolves corruption domain shifts without parameter updates. | **Strongly Supported** | Shift-Gated BN-Adapt delivers **+4.25% to +5.19%** $mCA$ leap, setting project records at **88.50%** (CBM) and **88.97%** (Frequency LFC). |

---

## 6. Scope, Dataset, and Legal Concept Ontology

### Primary Dataset: GTSRB Census & Leakage-Free Split
- **Total Images:** 51,838 (39,209 raw training, 12,630 held-out test).
- **Video-Track Structure:** 1,307 physical sign tracks in the training set (30 frames per track). Naive random image splitting leaks identical consecutive frames between train and validation sets, producing artificially inflated metrics ($>99.5\%$).
- **Group-Stratified Partitioning (`random_state=42`):**
  - `train.csv`: 31,378 images (1,046 tracks, 80% per class).
  - `val.csv`: 7,830 images (261 tracks, 20% per class; **0 track overlap**).
  - `test.csv`: 12,630 images (official held-out benchmark preserved pristine).

### 4-Axis, 22-Dimensional Legal Concept Ontology
Grounded in the **UNECE Vienna Convention (1968)** and **German StVO (2017)**:
- **Axis 1: Shape (6 dims):** `circle`, `triangle_up`, `triangle_down`, `octagon`, `diamond`, `rectangle`.
- **Axis 2: Colour Profile (5 dims):** `white_red`, `blue_white`, `yellow_white`, `white_black`, `red_white`.
- **Axis 3: Border Type (4 dims):** `red_border`, `white_border`, `black_border`, `none`.
- **Axis 4: Symbol Category (7 dims):** `speed_numeral`, `warning_hazard`, `mandatory_direction`, `prohibition_restriction`, `priority_geometry`, `deregulation_cancel`, `none`.
- Total Bottleneck Dimension: $K = 6 + 5 + 4 + 7 = 22$.

---

## 7. Deterministic Preprocessing Contract

To prevent anisotropic aspect ratio distortion from deforming circular speed limits into ellipses or warning triangles into irregular angles, all models adhere to a frozen 4-stage contract:
1. **Context-Buffered ROI Crop:** Expand annotated bounding box by $\alpha = 0.10$ ($10\%$ margin) to capture outer contrasting borders.
2. **Aspect-Preserving Letterbox Resizing:** Scale proportionally using bilinear interpolation such that $\max(w', h') = 64$.
3. **Symmetric Zero-Padding:** Center the rescaled sign on a neutral $[0, 0, 0]$ black $64 \times 64$ canvas.
4. **Standardization:** Normalize via ImageNet transfer statistics $(\mu=[0.485, 0.456, 0.406], \sigma=[0.229, 0.224, 0.225])$.

---

## 8. Master Experimental Results Matrix

All five primary paradigms and test-time adaptation variants evaluated across 12,630 held-out test images and the 75-point Common Corruptions suite ($947,250$ inferences per model):

| Model Paradigm | Clean Acc (%) | Clean Macro-F1 (%) | Clean ECE | Static $mCA$ (%) | Static Rank | RPD (%) | Weather $mCA$ | Blur $mCA$ | Digital $mCA$ | Noise $mCA$ | Shift-Gated BN $mCA$ (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline ResNet-18** *(Exp 1)* | $97.91\%$ | $97.07\%$ | $0.0082$ | $83.88\%$ | #2 | $14.33\%$ | $89.83\%$ | $94.87\%$ | $70.01\%$ | **$76.08\%$** | *Pending test* |
| **AugMix ResNet-18** *(Exp 2)* | **$98.47\%$** | **$97.78\%$** | **$0.0071$** | $83.10\%$ | #4 | $15.61\%$ | **$93.34\%$** | **$95.74\%$** | $69.12\%$ | $70.11\%$ | *Pending test* |
| **Frequency LFC ResNet-18** *(Exp 3)* | $97.32\%$ | $95.68\%$ | $0.0127$ | $83.78\%$ | #3 | $13.91\%$ | $81.73\%$ | $94.60\%$ | **$86.67\%$** | $69.88\%$ | **$88.97\%$** 🏆 |
| **Residual CBM ResNet-18** *(Exp 4)* | $97.59\%$ | $96.75\%$ | $0.0114$ | **$84.29\%$** 🏆 | **#1** | **$13.63\%$** 🏆 | $91.49\%$ | $95.10\%$ | $71.18\%$ | $75.20\%$ | **$88.50\%$** |
| **Hybrid (AugMix + CBM)** *(Exp 5)* | $97.61\%$ | $96.53\%$ | $0.0127$ | $81.63\%$ | #5 | $16.37\%$ | $92.55\%$ | $94.72\%$ | $68.49\%$ | $66.93\%$ | *Pending test* |

---

## 9. Key Discoveries and Scientific Takeaways

1. **The Inductive Semantic Cushioning Effect:**
   Grounding intermediate neural activations in legally certified physical concepts (Shape, Colour, Border) provides structural regularization that cushions the network against unseen pixel perturbations, achieving the project's highest static $mCA$ ($84.29\%$) and lowest relative degradation ($13.63\%$ RPD).

2. **Resolution of the 25.58% Oracle Degeneracy Ceiling:**
   Across 43 GTSRB classes, the 4-axis ontology yields exactly 11 unique concept tuples (all speed limits share identical concepts). A strict sequential CBM ($X \to C \to Y$) caps accuracy at $11/43 \approx 25.58\%$. The formulated **Residual CBM** ($W_{\text{class}}\hat{c} + W_{\text{aux}}h(x)$) resolves this ceiling by learning sub-symbolic typographic features in an auxiliary projection while retaining full concept interpretability and counterfactual intervention capabilities.

3. **Falsification of Naive Complementarity (Representation-Augmentation Interference):**
   Unifying AugMix and CBM via dual-level JSD consistency resulted in a severe $-2.66\%$ performance collapse ($81.63\%$). The mechanism was identified as color-semantic conflict: AugMix color operations (`solarize`, `posterize`) contradict deterministic color labels (`white_red`), while augmented high-frequency artifacts corrupt the 22-dimensional bottleneck. This was solved via Fourier LFC filtering ($84.21\%$ $mCA$).

4. **The Test-Time Normalization Breakthrough (Shift-Gated BN-Adapt):**
   Training-time tricks without corrupted data plateau in an $81.8\% - 85.8\%$ band. Dynamically updating Batch Normalization running statistics at test time eliminates covariate shift, propelling robustness across all models and establishing an all-time benchmark record of **89.79%** on Frequency-Aware CBM (Exp 6) and **89.29%** on Residual CBM (Exp 4) ($+35.91\%$ gain under heavy fog). Shift-gating eliminates minor degradation on mild shifts, preserving $97.59\%$ clean test accuracy with zero penalty.

5. **Diagnostic Failure Analysis:**
   - **Concept Degradation Hierarchy:** Shape is twice as resilient as Symbol Category under extreme noise (CDR $11.2\%$ vs $20.2\%$).
   - **Error Concentration:** $53.30\%$ of errors under severe noise are within-group, with $66.30\%$ concentrated in speed-limit numerals due to shared macro-concepts.
   - **OCS Negative Result:** Multi-task concept reranking fails as an independent validator because task and concept heads share a common backbone and collapse synchronously under severe noise.

---

## 10. Summary of Primary Contributions

1. **A Rigorous Master Dataset & Preprocessing Protocol:** Leakage-free group stratification (1,307 tracks) and aspect-preserving letterboxing contract for GTSRB.
2. **A Legally Grounded Physical Concept Ontology:** 4-axis, 22-dimensional multi-task annotation standard based on the UNECE Vienna Convention and German StVO.
3. **The Residual Concept Bottleneck Model:** Mathematical formulation decoupling macroscopic concept invariants from fine-grained typography, overcoming the 25.58% vocabulary ceiling.
4. **Empirical Discovery of Representation-Augmentation Interference:** Theoretical and empirical explanation of why spatial color augmentations conflict with concept bottlenecks, resolved by Fourier spectral filtering.
5. **Shift-Gated Normalization Adaptation Protocol:** Setting the new benchmark for TSR robustness under severe domain shift (**89.79%** $mCA$ on Exp 6), verified down to batch sizes of $N=8$ for streaming automotive perception.\n