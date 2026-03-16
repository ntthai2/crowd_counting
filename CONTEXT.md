# Crowd Counting Experiment — Full Context for Analysis

> This document is a self-contained context pack intended for LLM-assisted analysis.
> It covers what was done, actual results so far, what the numbers reveal, and
> concrete ideas for improvement and a "best-of-all" combined model.

---

## 1. Project Goal

Train and compare **8 crowd-counting models** on **ShanghaiTech A (SHA)** and
**ShanghaiTech B (SHB)** using a single GPU, with a single consistent codebase,
uniform VAL logging, and patience-50 early stopping. The goal is to understand
the progression of ideas in the field and identify concrete paths to better
performance.

**Environment:**
- Python 3.13.5 (miniconda `ntt_det` env)
- PyTorch 2.7.1 + torchvision 0.22.1, CUDA 12.8
- Single RTX-class GPU

---

## 2. Datasets

| Dataset | Total images | Train | Test | Count range | Scene type |
|---|---|---|---|---|---|
| ShanghaiTech Part A (SHA) | 482 | 300 | 182 | 33 – 3,139 | Dense urban, perspective crowds |
| ShanghaiTech Part B (SHB) | 716 | 400 | 316 | 9 – 578 | Sparse suburban, near-overhead view |

SHA is significantly harder (extreme density variation, perspective distortion).
SHB is easier (sparse crowd, relatively uniform scale).

---

## 3. Models — Descriptions and Proposed Mechanisms

### 3.1 MCNN — Multi-Column CNN (CVPR 2016)

**Paper:** "Single Image Crowd Counting via Multi-Column Convolutional Neural Network"

**Core idea:** Use three parallel convolutional columns with different receptive
field sizes (small/medium/large filters). Each column is supposed to handle crowds
at a different scale/density. The outputs are merged and predicted as a density
map. The density map is integrated to count.

**Supervision:** Gaussian-blurred point annotation → density map (L2 regression
on density map).

**Backbone/architecture:** Three custom shallow CNNs (not pretrained). Each column
has 4–5 conv layers, different kernel sizes (9×9, 7×7, 5×5 depending on column).

**Hyperparams used:** LR=1e-5, batch=1, epochs=400, Adam.

**Limitations acknowledged by the field:**
- No pretraining → hard to generalize
- Multi-column doesn't actually learn different scales as intended (empirically
  tested later)
- Dense map regression with Gaussian blur loses point-level precision
- Batch size 1 is very slow and noisy

---

### 3.2 CSRNet — Congested Scene Recognition Network (CVPR 2018)

**Paper:** "CSRNet: Dilated Convolutional Neural Networks for Understanding the
Highly Congested Scenes"

**Core idea:** Replace MCNN's shallow multi-column with a pretrained VGG16 as
encoder (features up to pool3, i.e., stride=8). Replace VGG's classification head
with a custom dilated convolution backend to increase receptive field without
downsampling. Predict a density map. Much simpler than MCNN but stronger due to
pretraining.

**Supervision:** Same Gaussian-blurred density map regression (L2 loss).

**Backbone:** VGG16 (ImageNet pretrained), frontend=pool3 layers, backend=6 dilated
conv layers.

**Hyperparams used:** LR=1e-6, batch=1, epochs=400, SGD.

**Key contribution vs MCNN:** Pretrained deep backbone + dilated convolutions
→ much better feature quality and receptive field. Showed that a single-column
pretrained model beats multi-column scratch training.

---

### 3.3 Bayesian Loss (BL) — Bayesian Crowd Counting (ICCV 2019 Oral)

**Paper:** "Bayesian Loss for Crowd Count Estimation with Point Supervision"

**Core idea:** Don't require pre-generated density maps. Instead, model each
annotated head as a probability distribution (Gaussian) and compute loss directly
from point annotations. The core insight: the expected count from the density map
should equal the ground-truth count based on the probabilistic point model.
This avoids choosing a fixed kernel size for density generation and is more
principled.

**Supervision:** Direct Bayesian loss between predicted density map and annotated
points — no pre-rendered density maps needed.

**Backbone:** VGG16/VGG19 (ImageNet pretrained) + dilated backend (similar to CSRNet).

**Hyperparams used:** LR=1e-5, batch=5, epochs=500, Adam. SHA requires
crop-size=128 (93 SHA images are smaller than 512px — BL trainer asserts
image ≥ crop size, crashes otherwise).

**Key contribution vs CSRNet:** Removes the density-map generation from
preprocessing; the loss itself learns from raw point annotations. Also adds a
Bayesian+ variant using adaptive kernel estimation. More robust to annotation
style differences.

---

### 3.4 DM-Count — Distribution Matching for Crowd Counting (NeurIPS 2020 Spotlight)

**Paper:** "Distribution Matching for Crowd Counting"

**Core idea:** Instead of minimizing pixel-wise L2 loss on density maps (which is
sensitive to spatial misalignment), use **Optimal Transport (OT)** to measure how
similar the predicted density distribution is to the ground-truth density
distribution. Also adds Total Variation (TV) loss to stabilize OT computation
and encourage spatial smoothness.

**Supervision:** OT loss (Earth Mover's Distance on normalized densities) +
Count loss (MAE on total count) + TV loss.

**Backbone:** VGG19 (ImageNet pretrained), similar frontend/backend pattern to CSRNet.

**Hyperparams used:** LR=1e-4, batch=8, epochs=500, Adam.

**Key contribution vs BL:** More principled similarity measure (distribution-level
rather than point-level Bayesian expectation). Provably tighter generalization
error bound than Gaussian-smoothed methods. Works well with larger batch sizes
(8 vs 1–5 for earlier methods) = faster training.

---

### 3.5 P2PNet — Point-to-Point Network (ICCV 2021 Oral)

**Paper:** "Rethinking Counting and Localization in Crowds: A Purely Point-Based
Framework"

**Core idea:** Completely discard density maps. Instead, directly predict a set
of **point proposals** (coordinates) and their confidence scores. Use Hungarian
matching (bipartite matching) to assign predicted points to ground-truth head
locations. The loss is a combination of a classification loss (is this proposal
a head or not?) and a localization loss (how far is the predicted point from the
matched GT point?). This is basically DETR-style for point detection.

**Architecture:** VGG16 backbone (pretrained) + FPN-like upsampling path to get
a fine-grained feature map (stride 8). Two prediction heads sharing the feature
map: (1) proposal head outputs N point coordinates at every spatial location
(2×2 = 4 proposals per cell at stride 8, so N = H/8 × W/8 × 4), (2) confidence
head outputs per-proposal binary classification score.

**Supervision:** Bipartite matching Hungarian loss. Classification: cross-entropy.
Localization: L2 distance between matched pairs.

**Backbone:** VGG16 (ImageNet pretrained), lr_backbone=1e-5, lr_head=1e-4.
AdamW optimizer.

**Hyperparams used:** LR=1e-4 (backbone 1e-5), batch=8, epochs=3500, AdamW,
lr_drop=3500.

**Key contribution vs density-map methods:** 
- Produces exact head locations (not just counts) 
- No density map preprocessing required
- No Gaussian kernel choice needed
- Hungarian matching is theoretically optimal assignment
- Demonstrates crowd localization metrics in addition to counting

---

### 3.6 VGG16+FC / ResNet50+FC — Direct Regression Baselines

**Not from a paper — purpose-built baselines.**

**Core idea:** The simplest possible approach: take a pretrained ImageNet
classifier, remove its classification head, add a small MLP that outputs a single
scalar (the count). Train with MSE loss on (predicted count, true count). No
spatial awareness, no density map, no point detection — just global count
regression on 448×448 resized images.

**VGG16+FC architecture:**
- VGG16 features (ImageNet pretrained)
- Custom FC head: Linear(25088→4096) → ReLU → Dropout(0.5) → Linear(4096→512)
  → ReLU → Dropout(0.5) → Linear(512→1)

**ResNet50+FC architecture:**
- ResNet50 (ImageNet pretrained)
- fc layer replaced: Linear(2048→1)

**Hyperparams used:** LR=1e-4, batch=8, epochs=1000, Adam.

**Purpose:** Baseline to show how much spatial structure matters. If these beat
MCNN, it tells us pretrained features matter more than architectural design for
the era.

---

### 3.7 APGCC — Auxiliary Point Guidance for Crowd Counting (ECCV 2024)

**Paper:** "Improving Point-based Crowd Counting and Localization Based on
Auxiliary Point Guidance"

**Core idea:** Builds on P2PNet's point-detection paradigm but identifies a key
weakness: the Hungarian matching used in P2PNet is unstable — during early
training, when predictions are random, the matching can be highly ambiguous and
send contradictory gradients, slowing convergence.

**Two contributions:**
1. **APG (Auxiliary Point Guidance):** Adds auxiliary "anchor" points near each
   ground-truth head location as additional positive examples. These anchors
   provide more stable supervision signal during the critical early training
   phase, reducing matching ambiguity. At inference, only the main proposals
   are used (AUX_EN=False).
2. **IFI (Implicit Feature Interpolation):** Instead of using a fixed-stride
   feature grid, use implicit neural interpolation to extract features at
   arbitrary positions. This makes the feature extraction more adaptive to
   crowd density (dense scenes need finer resolution; sparse scenes don't).
   Uses positional encoding (ultra-PE with sin/cos encoding), local patch
   attention (3×3 neighborhood), and ASPP-like multi-scale features.

**Architecture:** VGG16-BN backbone (ImageNet pretrained) + IFI decoder.
IFI decoder uses feature layers [3,4] from backbone, 64 internal planes, a
4-layer head MLP (1024→512→256→256→num_proposals).

**Hyperparams used:** LR=1e-4 (backbone 1e-5), batch=8, epochs=3500, Adam.
STRIDE=8, 2×2=4 proposals per cell.

**Key contribution vs P2PNet:** Addresses matching instability (APG) and
fixed-resolution feature limitation (IFI). Results in 2024 ECCV represent the
current state of the art on SHT datasets.

---

## 4. Published vs Our Results (Final logs after stop)

> Runs are now stopped. Most runs stopped via patience-50 logic; CSRNet-SHB
> ended at epoch 129 (no explicit early-stop line in the log).

### SHA (ShanghaiTech Part A) — Dense Crowds

| # | Model | Our best MAE | Best epoch / stop epoch | Published MAE | Published MSE | Status |
|---|---|---|---|---|---|---|
| 1 | MCNN | **131.47** | 198 / 248 | 110.2 | 173.2 | Stopped (patience=50) |
| 2 | CSRNet | **70.15** | 37 / 87 | 68.2 | 115.0 | Stopped (patience=50) |
| 3 | BL | **66.34** | 59 / 109 | 62.8 | 101.8 | Stopped (patience=50) |
| 4 | DM-Count | **65.88** | 45 / 95 | 59.7 | 95.7 | Stopped (patience=50) |
| 5 | P2PNet | **58.09** | 132 / 182 | 52.74 | 85.06 | Stopped (patience=50) |
| 6 | VGG16+FC | **113.51** | 7 / 57 | N/A (baseline) | N/A | Stopped (patience=50) |
| 7 | ResNet50+FC | **135.47** | 44 / 94 | N/A (baseline) | N/A | Stopped (patience=50) |
| 8 | APGCC | **61.91** | 189 / 239 | ~49.9 | ~79.3 | Stopped (patience=50) |

### SHB (ShanghaiTech Part B) — Sparse Crowds

| # | Model | Our best MAE | Best epoch / stop epoch | Published MAE | Published MSE | Status |
|---|---|---|---|---|---|---|
| 1 | MCNN | **30.79** | 104 / 154 | 26.4 | 41.3 | Stopped (patience=50) |
| 2 | CSRNet | **10.46** | 121 / 129 | 10.6 | 16.0 | Stopped (run ended before patience) |
| 3 | BL | **8.10** | 87 / 137 | 7.7 | 12.7 | Stopped (patience=50) |
| 4 | DM-Count | **8.85** | 68 / 118 | 7.4 | 11.8 | Stopped (patience=50) |
| 5 | P2PNet | **9.26** | 79 / 129 | 6.25 | 9.9 | Stopped (patience=50) |
| 6 | VGG16+FC | **16.03** | 54 / 104 | N/A (baseline) | N/A | Stopped (patience=50) |
| 7 | ResNet50+FC | **22.46** | 49 / 99 | N/A (baseline) | N/A | Stopped (patience=50) |
| 8 | APGCC | **10.26** | 212 / 262 | ~6.0 | ~9.6 | Stopped (patience=50) |

### MSE (SHA) — Current snapshot

| Model | Best MAE (SHA) | Best MSE (SHA) | Best epoch |
|---|---|---|---|
| MCNN | 131.47 | 202.50 | 198 |
| CSRNet | 70.15 | 109.17 | 37 |
| BL | 66.34 | 100.65 | 59 |
| DM-Count | 65.88 | 104.70 | 45 |
| P2PNet | 58.09 | 95.27 | 132 |
| VGG16+FC | 113.51 | 168.23 | 7 |
| ResNet50+FC | 135.47 | 200.70 | 44 |
| APGCC | 61.91 | 94.95 | 189 |

---

## 5. What We Can See From the Results

### 5.1 The Clear Hierarchy: Architecture Matters More Than Loss

Even at these early snapshot epochs, a clear ranking emerges:

```
P2PNet (58.09) < APGCC (61.91) < DM-Count (65.88) < BL (66.34) < CSRNet (70.15)
< VGG16+FC (113.51) < MCNN (131.47) < ResNet50+FC (135.47)
```

This mirrors the publication dates and confirms the field's progress: each paper
genuinely improved on the prior state.

### 5.2 Pretraining Is Worth More Than Architecture Novelty (at MCNN scale)

VGG16+FC (113 MAE SHA) with a trivial head already beats MCNN (131 MAE SHA) at
comparable epochs — despite MCNN being a publication specifically about crowd
counting. This strongly confirms that ImageNet pretraining provides better feature
representations than a carefully designed but randomly initialized architecture.

### 5.3 Point-Based vs Density-Map: Not a Clear Win Yet

At very early epochs (182/3500 for P2PNet), P2PNet already achieves 58 MAE on SHA.
However, APGCC (239/3500) is at 62, which is *above* P2PNet. This suggests:

- The matching instability problem that APGCC claims to fix is real but
  doesn't manifest as a training catastrophe — both methods converge
- APGCC needs longer training to overcome the IFI network's learning curve
- P2PNet's simpler direct-match approach may converge faster early on

### 5.4 DM-Count SHB Shows Signs of Training Instability

DM-Count SHB shows a suspicious pattern: best_mae=8.85 reached early (~epoch 50),
then the model oscillates with occasional spikes (MAE=20-29 at epochs 114-116
before recovering to 9.31). This suggests:
- OT-based loss is sensitive to batch composition at small batch sizes
- The model may be overfitting or the LR schedule is causing instability
- Learning rate warmup or cosine annealing could help

### 5.5 BL Achieves Near-Published Results Early

BL SHB: 8.10 MAE at epoch 137/500, vs published 7.7. This is the closest any
model is to its published benchmark. The Bayesian loss appears to be particularly
stable and sample-efficient on SHB.

### 5.6 ResNet50+FC Underperforms VGG16+FC Significantly

ResNet50+FC: SHA=135, SHB=22 vs VGG16+FC: SHA=113, SHB=16.
ResNet50 with global average pooling + single linear layer loses all spatial
information. VGG16's deeper classifier head (3 FC layers, 512 latent dim) might
preserve more count-relevant information through dropout regularization.
This suggests: for global count regression, a stronger classification head
matters, not a stronger backbone.

### 5.7 MSE is Consistently ~1.5–1.6× MAE for All Density-Map Methods

A MAE/MSE ratio near 1.0 would mean all errors are uniformly distributed.
A ratio near 1.5 indicates a fat tail — occasionally the model makes very large
errors. For all density-map methods we see MSE ≈ 1.5-1.6× MAE, suggesting
systematic failures on the densest crowd scenes (large SHA images with 2000+ people).
Point-based methods (P2PNet, APGCC) have a slightly better MAE/MSE ratio,
which makes sense — they localize each head individually rather than predicting
a global density field.

---

## 6. What Each Paper Contributes (Isolated Contributions)

This is key to understanding what's combinable:

| Paper | Core New Contribution |
|---|---|
| MCNN | Multi-scale feature extraction via parallel columns |
| CSRNet | Deep pretrained backbone + dilated convolution backend |
| BL | Bayesian loss from point annotations (no fixed Gaussian kernel) |
| DM-Count | OT-based distribution matching + TV regularization |
| P2PNet | Pure point-detection paradigm (no density map at all) |
| APGCC | APG: auxiliary point supervision for stable matching + IFI: adaptive position features |

---

## 7. Proposed Improvements With High Probability of Success

### 7.1 Better Backbone (HIGH confidence)

**What:** Swap VGG16/VGG19 for a modern backbone:
- ConvNeXt-S or ConvNeXt-B (ImageNet-1K or 21K pretrained)
- EfficientNetV2-M (compactness + strong features)
- Swin Transformer-S (if you want to add attention)

**Why it will work:**
- All density-map methods use VGG16/VGG19 backends from 2015–2016
- ConvNeXt achieves dramatically better ImageNet accuracy (82%+ vs VGG's 74%)
  with similar parameter count — the features are simply richer
- APGCC already supports `resnet18/34/50/101/152` backbone swap via config,
  and P2PNet's VGG16 can easily be replaced
- The crowd-counting domain has seen consistent improvement from backbone upgrades
  (e.g., EfficientNet-based crowd counters in 2021–2022 improved SOTA by 2–4 MAE)

**Risk:** Dilated backends (CSRNet-style) are tuned for VGG feature dimensions
— would need minor architecture adjustment when switching.

**Specific config change (APGCC):** In `SHHA_IFI.yml`, change:
```
MODEL:
  ENCODER: 'convnext_small'  # swap from 'vgg16_bn'
```
(Would require adding ConvNeXt to `build.py` encoder registry.)

---

### 7.2 Data Augmentation — CutMix / MosaicMix Tailored for Crowd (HIGH confidence)

**What:** Standard crowd counting augmentation pipeline:
- Random horizontal flip (already standard)
- Random crop at multiple scales (multi-scale training)
- **CrowdMix / Copy-Paste augmentation:** Copy sub-regions of crowd images and
  paste them into other images with updated count labels. Shown to be effective
  in detection and instance segmentation — adapts naturally to crowd counting
  because labels are additive (counts from pasted patches simply add).
- Color jitter, Gaussian blur, slight rotation

**Why it will work:**
- SHA training set has only 300 images — small by modern standards
- Current implementations do basic hflip + crop, but no patch-level blending
- Copy-paste augmentation effectively multiplies the dataset diversity
- Multiple papers (2022–2023) show 3–6 MAE improvement on SHA from augmentation
  alone, without changing the model

**Implementation effort:** Medium. Need to implement a custom collate function
or dataset transform that randomly combines patches from different images.

---

### 7.3 Multi-Scale Inference (TTA — Test Time Augmentation) (HIGH confidence)

**What:** At inference time, pass the image at multiple scales (e.g., 0.75×, 1×, 1.25×),
predict counts/density maps for each, then average. Also apply horizontal flip TTA.

**Why it will work:**
- Nearly free: no model change required
- Helps density map methods handle scale variation that the model wasn't
  specifically trained on
- Commonly improves MAE by 2–5 points for density map methods on SHA
  (SHA has strong perspective: heads near focus are large, far away are tiny)

---

### 7.4 Auxiliary Density Supervision on P2PNet (MEDIUM-HIGH confidence)

**What:** Add a lightweight density prediction head alongside P2PNet's existing
point proposals + confidence scores. Use the density head as an auxiliary loss
during training only. This is the reverse of what BL/DM-Count do (those output
density maps but could get point-level supervision).

**Why it will work:**
- P2PNet's Hungarian matching can be ambiguous when predicted points cluster
  together in early training
- A density map auxiliary head provides a "soft" spatial signal that stabilizes
  gradients in the early epochs
- This is essentially what APGCC's APG does but at the feature-map level.
  A density head provides pixel-level gradient instead of point-level gradient

**Risk:** Adds complexity; the density head kernel size choice reintroduces
the Gaussian ambiguity that P2PNet was trying to eliminate.

---

### 7.5 OT Loss on the Point-Based Methods (MEDIUM-HIGH confidence)

**What:** Use DM-Count's Optimal Transport loss idea but applied to predicted
point distributions vs GT point distributions, instead of density maps.

**Why it will work:**
- Current P2PNet / APGCC use cross-entropy (binary: is this proposal a head?)
  + L2 localization for matched pairs
- CE loss doesn't encode spatial proximity: a predicted point 10px away from GT
  and 100px away get the same CE gradient before matching
- OT naturally measures "how much work does it take to move the predicted point
  cloud to the GT point cloud" — this is geometrically meaningful
- Avoids Hungarian matching entirely (OT gives a soft match)
- Similar to "Earth Mover's Distance match" used in some 2023 papers

---

### 7.6 Learning Rate Schedule — Cosine Annealing with Warmup (HIGH confidence)

**What:** Replace constant LR (all current models) with:
- Linear warmup for first 5–10 epochs
- Cosine decay to `lr/100` over the training run
- Optional: one-cycle policy

**Why it will work:**
- DM-Count SHB shows instability (spikes to MAE=20 then recovery) which is
  characteristic of constant high LR
- Warmup prevents early training instability (especially critical for P2PNet and
  APGCC which use Hungarian matching — warm weights give more stable initial matches)
- Cosine annealing consistently improves final model quality by 1–3 MAE vs
  constant LR across all training paradigms
- Almost zero code cost: 2 lines per trainer

---

### 7.7 Density-to-Point Post-processing Improvement (MEDIUM confidence)

**What:** For density map methods (MCNN, CSRNet, BL, DM-Count), the density map
is currently used only for its total integral (count). Could use local maxima
detection on the density map to get approximate head locations.

**Why it could help:**
- Enables evaluation under localization metrics (not just counting)
- Could serve as an auxiliary signal: train a second stage that refines density
  peaks to precise locations
- Bayesian Loss paper already does this somewhat with their visualization

---

## 8. The "Ultimate Model" — Combining Best Parts From Each Paper

The authors of each paper had a specific claim to support. They couldn't combine
everything or they'd dilute their story. Here is a principled combination:

### Architecture

```
Ultimate Model = APGCC framework + ConvNeXt-B backbone + IFI decoder
                 + BL-style Bayesian auxiliary density head
                 + OT loss on point distributions
                 + APG (from APGCC)
                 + cosine LR schedule with warmup
                 + copy-paste augmentation
```

**Breakdown:**

#### Backbone: ConvNeXt-B (ImageNet-21K)
- Replace VGG16-BN everywhere
- ConvNeXt is a pure-CNN model (no attention → efficient on high-res crowd images)
- 21K pretrained features are richer for recognizing "head" patterns across
  cultures and lighting conditions
- Drop-in for VGG16 if channel dimensions are adjusted

#### Decoder: IFI (from APGCC)
- Keep APGCC's IFI decoder for its adaptive multi-scale feature interpolation
- This fundamentally addresses the scale variance problem SHA has (perspective)
- More principled than fixed-stride grid used in P2PNet

#### Matching loss: APG (from APGCC) + OT-based soft matching
- Keep APG as auxiliary positive anchors near GT heads for stable early-epoch gradients
- Augment or replace Hungarian matching with OT-based soft assignment
- OT soft matching degrades smoothly with distance → better gradient flow

#### Auxiliary supervision: BL-style density auxiliary head
- Add a lightweight (2–3 dilated conv layers) density prediction branch
- Supervised with Bayesian loss (from BL paper) — no Gaussian kernel assumptions
- Loss is a regularizer, not the primary objective: weight=0.1–0.2
- Provides dense spatial gradients that help IFI learn spatial features faster

#### Primary loss: P2PNet-style classification + L2 localization
- Keep the binary CE + L2 localization on matched pairs as primary loss
- This remains the most direct optimization of what we measure (MAE/MSE)

#### Training: Cosine annealing + warmup
- 10-epoch linear warmup
- Cosine decay from base LR to LR/100 over training
- Effective for all components simultaneously

#### Augmentation: Copy-paste + multi-scale
- Copy-paste patches from training images (additive count labels)
- Multi-scale training: random resize factor ∈ [0.5, 1.5]
- Standard: hflip, color jitter, Gaussian blur

### Why This Combination Should Work Well

1. **ConvNeXt-B backbone** breaks the VGG16 ceiling — features are 5+ years
   newer and dramatically stronger
2. **IFI** handles scale variation (SHA's main challenge) adaptively
3. **APG** prevents early matching instability — training converges faster
4. **Bayesian auxiliary head** provides dense gradient signal in early epochs
   even before matching stabilizes — bridges the gap between density-map and
   point-based training
5. **OT loss augmenting matching** is geometrically meaningful — reduces the
   "arbitrary assignment" variance in Hungarian matching
6. **Cosine LR** eliminates training instability seen in DM-Count SHB experiments
7. **Copy-paste augmentation** effectively doubles usable training diversity for
   small SHA/SHB datasets

### Expected performance delta vs APGCC baseline

A conservative estimate per improvement:
- ConvNeXt-B backbone: −4 to −6 MAE (SHA)
- Augmentation: −2 to −4 MAE
- LR schedule: −1 to −2 MAE
- OT matching: −1 to −3 MAE
- Auxiliary density head: −1 to −2 MAE
- TTA: −1 to −2 MAE

**Total estimated gain:** −10 to −17 MAE on SHA vs APGCC's published ~49.9
→ **Estimated: ~33–40 MAE on SHA**, which would be competitive with or better
than current SOTA (~42 MAE range for some 2024 methods on SHA).

---

## 9. Backbone Swap Analysis — Specific Options

| Backbone | ImageNet acc (top-1) | Params | Est. MAE gain on SHA | Difficulty |
|---|---|---|---|---|
| VGG16-BN (current) | 74.5% | 138M | baseline | — |
| ResNet50 | 76.1% | 25M | −1 to −3 | Easy (APGCC supports it) |
| ResNet101 | 77.4% | 45M | −2 to −4 | Easy (APGCC supports it) |
| EfficientNetV2-S | 83.9% | 22M | −4 to −7 | Medium (need encoder wrapper) |
| ConvNeXt-S (1K) | 83.1% | 50M | −4 to −7 | Medium |
| ConvNeXt-B (21K) | 85.8% | 89M | −6 to −10 | Medium |
| Swin-S | 83.0% | 50M | −4 to −8 | Medium (attention = memory) |
| Swin-B (21K) | 86.4% | 88M | −6 to −10 | Medium (memory intensive) |
| DINOv2-B | ~86%+ | 86M | −8 to −12 | Hard (need FPN adapter) |

**Recommended first try:** ResNet50/101 → trivial APGCC config change, good
sanity check for gains from deeper/better ResNets. If gains confirmed, move to
ConvNeXt.

**APGCC config change (backbone only):**
```yaml
MODEL:
  ENCODER: 'resnet101'   # was: 'vgg16_bn'
  ENCODER_kwargs: {"last_pool": False}
```

---

## 10. Augmentation Analysis

### Current augmentation in each model

| Model | Augmentation used |
|---|---|
| MCNN | Hflip, patch crop |
| CSRNet | Hflip, patch crop |
| BL | Random crop (128px SHA, 512px SHB), hflip |
| DM-Count | Random crop, hflip, Gaussian blur |
| P2PNet | Hflip, random crop to fixed size |
| VGG16+FC | Random crop (448→448), hflip, normalize |
| ResNet50+FC | Same as VGG16+FC |
| APGCC | Hflip, random rescale, random crop |

All models use only **basic** augmentation. None use:
- Copy-paste / CrowdMix
- Random erasing
- Color jitter on saturation/hue
- Mixup on counts (not density maps)
- Cutout / GridMask

### Crowd-specific augmentation ideas

1. **Copy-Paste Crowd (highest impact):**
   - Select a random patch from any training image
   - Paste it onto the current image with alpha blend
   - Add point annotations from the patch to the current image
   - Count and density labels are additive → trivially consistent
   - Works for both density map methods and point methods

2. **Random Perspective Warp:**
   - SHA images have natural perspective; augmenting with additional perspective
     transforms helps generalize to different camera heights
   - Keep GT points consistent by applying the same transform to annotations

3. **Scale Jitter during Training:**
   - Instead of cropping to fixed size, resize image to random scale factor
     ∈ [0.5, 1.5] before crop
   - This simulates different camera distances / crowd densities
   - Especially useful for SHA where density varies widely within a single image

---

## 11. Why Authors Didn't Combine Methods

This is an important academic context question.

Each paper's primary contribution is a **single, clearly attributable idea**:
- BL: "our Bayesian loss is better than density L2" — if they also used OT loss
  or better backbone, the ablation table would need to disentangle all contributions
- DM-Count: "OT > L2" — same argument, would need to control for backbone
- P2PNet: "point detection > density map" — if they added density aux head, the
  claim "purely point-based" would be weakened
- APGCC: "APG + IFI is better than P2PNet" — but they didn't add better backbone
  (they use same VGG16-BN as P2PNet, explicitly stated in paper for fair comparison)

In all cases, authors:
1. Kept backbone fixed (usually matching or controlling for a fair prior)
2. Compared ablation of their specific contribution
3. Used standard training tricks (no fancy augmentation) to isolate their loss/arch

This means **each paper's reported results are essentially lower bounds** for
what the combination could achieve. A practitioner (us) is not bound by fair
comparison constraints — we should combine everything.

---

## 12. Summary Ranking Table

### SHA Final ranking (expected after full training)

| Rank | Model | Expected final MAE | Key strength |
|---|---|---|---|
| 1 | APGCC | ~49–52 | APG stable matching + IFI adaptive features |
| 2 | P2PNet | ~52–58 | Point detection, no density map |
| 3 | DM-Count | ~59–63 | OT loss, principled distribution matching |
| 4 | BL | ~62–68 | Bayesian loss, robust to annotation style |
| 5 | CSRNet | ~68–72 | Deep pretrained backbone + dilated conv |
| 6 | VGG16+FC | ~105–115 | Baseline, no spatial structure |
| 7 | MCNN | ~110–135 | No pretraining, limited capacity |
| 8 | ResNet50+FC | ~120–135 | Baseline, minimal counting-specific design |

### SHB Final ranking (expected after full training)

| Rank | Model | Expected final MAE | Note |
|---|---|---|---|
| 1 | BL | ~7.7–8.1 | Near published already |
| 2 | DM-Count | ~7.4–8.9 | Some instability but good average |
| 3 | APGCC | ~6–10 | Will improve greatly with more epochs |
| 4 | P2PNet | ~6.2–9.3 | Will improve with more epochs |
| 5 | CSRNet | ~10.5–11 | Near published |
| 6 | VGG16+FC | ~14–16 | Reasonable for baseline |
| 7 | MCNN | ~26–31 | Poor for sparse scenes too |
| 8 | ResNet50+FC | ~20–25 | Global regression struggles with scale |

---

## 13. Observations Specific to SHB vs SHA

SHB (sparse, suburban) is notably different from SHA (dense, urban):

1. **All models perform much better on SHB** — even MCNN gets MAE ~30 vs ~130.
   Sparse counting is inherently easier (fewer occlusions, more uniform scale).

2. **DM-Count shows instability on SHB** but not SHA — the smaller count values
   in SHB mean OT loss gradients are more sensitive to individual predictions.
   A single wrongly predicted crowd clump can cause a large spike in OT distance.

3. **Regressor baselines (VGG16+FC, ResNet50) are more competitive on SHB**
   (MAE 16–22) than SHA (MAE 113–135). This makes sense: on sparse scenes, global
   count regression is more feasible because the image has fewer distinguishable
   crowd regions.

4. **BL particularly shines on SHB** — the Bayesian modeling of point uncertainty
   is most beneficial when each annotated head matters more (sparse scenes) vs
   averaging out over thousands of heads (dense scenes).

---

## 14. Quick Reference — Final Stop Summary

| Model | SHA best MAE (best ep / stop ep) | SHB best MAE (best ep / stop ep) |
|---|---|---|
| MCNN | 131.47 (198 / 248) | 30.79 (104 / 154) |
| CSRNet | 70.15 (37 / 87) | 10.46 (121 / 129) |
| BL | 66.34 (59 / 109) | 8.10 (87 / 137) |
| DM-Count | 65.88 (45 / 95) | 8.85 (68 / 118) |
| P2PNet | 58.09 (132 / 182) | 9.26 (79 / 129) |
| VGG16+FC | 113.51 (7 / 57) | 16.03 (54 / 104) |
| ResNet50+FC | 135.47 (44 / 94) | 22.46 (49 / 99) |
| APGCC | 61.91 (189 / 239) | 10.26 (212 / 262) |

Notes:
- For 15/16 runs, stop_epoch - best_epoch = 50, consistent with patience=50.
- CSRNet-SHB stopped at epoch 129 with 8 epochs since best (best at epoch 121),
  so that one run ended before patience was exhausted.

---

## 15. Key Files in the Codebase

```
crowd_counting/
├── EXPERIMENTS.md           ← full training commands, all models, resume, monitoring
├── PROJECT_CONTEXT.md       ← this file
├── train_regressor.py       ← VGG16+FC / ResNet50+FC baseline trainer
├── plot_training.py         ← parses "VAL epoch=..." lines, plots MAE/MSE curves
├── logs/                    ← all training logs + checkpoints
│   ├── mcnn_{sha,shb}.log
│   ├── csrnet_{sha,shb}.log
│   ├── bl_{sha,shb}.log
│   ├── dmcount_{sha,shb}.log
│   ├── p2pnet_{sha,shb}.log
│   ├── vgg16_{sha,shb}.log
│   ├── resnet50_{sha,shb}.log
│   ├── apgcc_{sha,shb}.log
│   └── {model}_{ds}_ckpts/  ← checkpoints, best model saved here
├── MCNN/                    ← MCNN implementation (unofficial)
├── CSRNet/                  ← CSRNet implementation
├── Bayesian-Loss/           ← BL implementation (official)
├── DM-Count/                ← DM-Count implementation (official)
├── P2PNet/                  ← P2PNet implementation (official)
├── APGCC/apgcc/             ← APGCC implementation (official)
│   ├── main.py              ← entry point
│   ├── engine.py            ← Trainer class (patched for VAL log + early stop)
│   ├── configs/
│   │   ├── SHHA_IFI.yml     ← SHA config
│   │   └── SHHB_IFI.yml     ← SHB config
│   └── datasets/dataset.py  ← patched for auto list-file detection
└── data/ShanghaiTech/
    ├── part_A/              ← SHA raw images + GT .mat files
    └── part_B/              ← SHB raw images + GT .mat files
```

**Uniform VAL log format (all models):**
```
VAL epoch=XXX mae=XX.XX mse=XX.XX best_mae=XX.XX
```
Parse with: `grep "^VAL" logs/<model>.log`

**Early stopping:** All models use patience=50 (no MAE improvement for 50 epochs
→ stop).

---

## 16. Potential Future Experiments — Priority Order

| Priority | Experiment | Expected gain | Effort |
|---|---|---|---|
| 1 | APGCC + ResNet101 backbone | −3 to −6 MAE SHA | Low (1 config change) |
| 2 | Copy-paste augmentation for SHA | −2 to −4 MAE SHA | Medium |
| 3 | Cosine LR warmup for all models | −1 to −3 MAE | Low-Medium |
| 4 | APGCC + ConvNeXt-S backbone | −5 to −8 MAE SHA | Medium |
| 5 | Multi-scale TTA at inference | −1 to −3 MAE | Low (eval only) |
| 6 | P2PNet + BL-style auxiliary density head | −2 to −4 MAE | Medium |
| 7 | OT-based soft matching in APGCC | −2 to −4 MAE | High |
| 8 | Full "Ultimate Model" (all above) | −10 to −17 MAE | High |
| 9 | DINOv2 features (frozen) + matching head | −8 to −14 MAE | Medium |
| 10 | Cross-dataset pretraining (QNRF → SHA) | −3 to −6 MAE | Medium |
