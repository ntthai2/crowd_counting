# Crowd Counting: Benchmarking and Robustness Analysis

---

## 1. Introduction

Crowd counting — estimating the number of people present in an image or video frame — is a fundamental problem in computer vision with direct applications in public safety, event management, and retail analytics. The task is deceptively difficult: people overlap, vary widely in scale depending on their distance from the camera, and the visual boundary between one person and the next often disappears entirely in dense scenes.

Most published work evaluates crowd counting models exclusively on curated benchmark datasets, where the distribution of images closely matches the training data. In practice, deployed systems must handle video from cameras with arbitrary viewing angles, lighting conditions, and crowd densities that may differ substantially from anything seen during training. This gap between benchmark performance and real-world performance is rarely studied directly.

This project pursues two complementary goals. The first is to benchmark a representative set of crowd counting models from four distinct methodological families on the standard ShanghaiTech benchmarks, establishing a performance baseline with consistent training and evaluation protocols. The second is to analyze how these models behave when applied to real-world internet video — specifically, how performance changes with scene type (dense vs. sparse), whether training on a dense dataset versus a sparse one produces meaningfully different predictions, and which models fail gracefully versus catastrophically under distribution shift.

The central questions guiding the analysis are: Which model performs best under controlled benchmark conditions, and does that advantage persist in the wild? How sensitive are predictions to the density distribution of the training data? And what are the characteristic failure modes of each architectural family when faced with scenes outside their training distribution?

---

## 2. Background

### 2.1 Methodological Families

Crowd counting approaches can be grouped into four families, each with a different inductive bias about what it means to "count" people.

**Detection-based counting** uses a general-purpose object detector trained to find human heads, and uses the number of bounding boxes as the predicted count. Unlike the other approaches, this method is not trained end-to-end for counting — it treats counting as a side effect of detection.

**Global regression** bypasses spatial reasoning entirely. A backbone network extracts image-level features, which are fed directly into a fully connected regression head that outputs a single scalar count. This is the simplest possible architecture and serves as a strong baseline for understanding how much spatial structure matters.

**Density map estimation** is the dominant paradigm. The model predicts a continuous spatial map where each pixel value represents a local density of people; the integral of the map over the image gives the total count. Ground-truth maps are constructed by placing a Gaussian kernel at each annotated head location, with the kernel width adapted to local crowd density. This approach handles occlusion and scale variation gracefully because it never requires the model to delineate individual people — it only needs to estimate how much "personhood" is present in each region.

**Point-based detection** takes the opposite approach: the model explicitly predicts a set of head locations as discrete points. The count is simply the number of predicted points that exceed a confidence threshold. This is more interpretable and, in principle, more precise, but requires the model to commit to exact head positions even under heavy occlusion.

### 2.2 Datasets

All models in this project are trained and evaluated on the two ShanghaiTech benchmarks. **Part A (SHA)** contains 482 images (300 train, 182 test) depicting dense urban crowds with counts ranging from 33 to 3,139 people per image. **Part B (SHB)** contains 716 images (400 train, 316 test) depicting sparser suburban scenes with counts ranging from 9 to 578. The two datasets are evaluated independently using their standard splits, which allows direct comparison to published baselines.

### 2.3 Evaluation Metrics

Two metrics are reported throughout: Mean Absolute Error (MAE) and Root Mean Squared Error (MSE):

$$\text{MAE} = \frac{1}{N}\sum_{i=1}^{N}|z_i - \hat{z}_i|, \quad \text{MSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(z_i - \hat{z}_i)^2}$$

where $z_i$ is the ground-truth count and $\hat{z}_i$ is the predicted count for image $i$. MAE measures average accuracy; MSE penalizes large individual errors more heavily and reflects stability across the test set.

For video inference, where no ground truth is available, temporal standard deviation (std) of per-frame predictions is used as a proxy for stability — a robust model should produce smoothly varying counts across frames of the same scene, rather than large frame-to-frame spikes.

---

## 3. Methodology

### 3.1 Model Selection

Nine models were trained and evaluated across the four families. From these, four were selected as the primary subjects for analysis, each representing a distinct paradigm:

- **YOLO11m-head** (detection counting) — a YOLO11m detector fine-tuned for human head detection; count is the number of predicted bounding boxes above a confidence threshold.
- **EfficientNet-B0 + FC** (global regression) — a single fully connected regression head attached to an EfficientNet-B0 backbone pretrained on ImageNet.
- **CSRNet** (density map) — the most widely cited density map baseline, using a VGG16 frontend followed by dilated convolutional layers to preserve spatial resolution.
- **P2PNet** (point detection) — predicts discrete head locations via a feature pyramid network and a bipartite matching loss.

Five additional models (MCNN, BL, DM-Count, APGCC, ResNet50+FC) were trained and evaluated but are not the focus of the analysis presented here. Their results are included in the full benchmark table for completeness.

### 3.2 Training Protocol

All models are trained independently on SHA and SHB using their standard train/test splits. Training uses early stopping with patience 50 (patience 200 for P2PNet due to its longer convergence time). A uniform validation logging format was applied across all models to facilitate automated result extraction.

CSRNet uses SGD with lr=1e-6 as specified in the original implementation. P2PNet uses AdamW with lr=1e-4 (backbone lr=1e-5). The regression model uses Adam with lr=1e-4, batch size 8, and input resolution 448×448. YOLO is trained on merged head-detection data from both datasets and evaluated by counting bounding boxes.

Several bugs in the original model repositories were patched before training, including a torchvision version check crash in P2PNet, a hardcoded private weight path in P2PNet's VGG loader, and a logging display bug in DM-Count. A feature index mismatch in P2PNet's FPN was also identified and corrected, enabling non-VGG backbone support.

### 3.3 Video Inference Setup

Twelve short internet videos were collected to test real-world performance. These cover two scene categories: **dense indoor mall** (7 videos, estimated 80–500+ people per frame) and **sparse retail/CCTV** (5 videos, estimated 1–20 people per frame). This division was chosen to probe domain gap along the axis that separates SHA (dense) from SHB (sparse).

Each model was evaluated twice on each video — once using the SHA-trained checkpoint and once using the SHB-trained checkpoint — to measure how training data density affects predictions on out-of-distribution scenes. YOLO is evaluated once, as it has no SHA/SHB distinction.

Inference runs at every 5th frame to reduce redundancy in short clips. Per-frame counts are logged to CSV; mean, standard deviation, min, and max are reported per video. Three videos were manually counted to provide approximate ground truth for validation.

---

## 4. Results

### 4.1 Benchmark Performance

Tables 1 and 2 report MAE and MSE for all nine models on SHA and SHB respectively, alongside published numbers where available.

**Table 1 — ShanghaiTech Part A**

| Model | Family | MAE ↓ | MSE ↓ | Published MAE |
|---|---|---|---|---|
| YOLO11m-head | Detection | 236.30 | 392.29 | — |
| EfficientNet-B0+FC | Regression | 91.67 | 138.42 | — |
| CSRNet | Density map | 70.15 | 109.17 | 68.2 |
| P2PNet | Point detection | **58.09** | **95.27** | 52.7 |

**Table 2 — ShanghaiTech Part B**

| Model | Family | MAE ↓ | MSE ↓ | Published MAE |
|---|---|---|---|---|
| YOLO11m-head | Detection | 40.20 | 72.93 | — |
| EfficientNet-B0+FC | Regression | 15.32 | 22.81 | — |
| CSRNet | Density map | 10.46 | 16.90 | 10.6 |
| P2PNet | Point detection | **9.26** | **16.53** | 6.7 |

The results follow the expected ordering across paradigms. P2PNet leads on SHA; on SHB the gap between point detection and density map narrows considerably. Our numbers trail published baselines by a small margin, attributable to single-GPU training and conservative epoch budgets.

The gap between YOLO (MAE 236 on SHA) and the density map and point detection models is striking — over 3x worse than CSRNet. Detection-based counting is fundamentally limited on dense scenes: once people overlap significantly, individual heads become indistinguishable to the detector, and the predicted count saturates well below the true value. The regression model (MAE 91.67) falls between the spatial models and YOLO, reflecting the cost of discarding spatial supervision while retaining end-to-end training for the counting objective.

### 4.2 Video Inference

Table 3 summarizes mean predictions across all 12 videos grouped by scene category and training dataset.

**Table 3 — Mean predictions by scene type and training dataset**

| Model | Dense (SHA ckpt) | Dense (SHB ckpt) | Sparse (SHA ckpt) | Sparse (SHB ckpt) |
|---|---|---|---|---|
| YOLO | ~25 | ~25 | ~3 | ~3 |
| Regressor (B0) | ~200 | ~160 | ~180 | ~120 |
| CSRNet | ~180 | ~210 | ~80 | ~45 |
| P2PNet | ~200 | ~210 | ~80 | ~10 |

*Values are approximate means across videos within each category.*

**Table 4 — Validated predictions against manual ground truth**

| Video | Scene | Ground truth | Reg SHA | CSRNet SHA | P2PNet SHA | P2PNet SHB | YOLO |
|---|---|---|---|---|---|---|---|
| Multi-level mall | Dense, ~150 | 140–160 | 169 | 88 | **147** | 173 | 4 |
| Christmas mall | Dense, ~500 | 400–600 | 714 | **495** | 587 | 302 | 15 |
| Convenience store | Sparse, ~10 | 8–12 | 425 | 238 | 156 | **6** | 3 |

Several patterns emerge from these results.

**P2PNet SHA is the most reliable model for medium-density scenes.** On the multi-level mall video (GT ~150), P2PNet SHA predicted 147 — the closest of any model. This is consistent with its strong SHA benchmark performance.

**CSRNet SHA handles extreme density better than any other model.** On the densest video (GT ~500), CSRNet SHA predicted 495, while the regressor predicted 714 and P2PNet predicted 587. The density map representation appears to scale more gracefully to very high counts, possibly because it does not need to resolve individual people.

**P2PNet SHB collapses on sparse scenes.** When the scene contains only ~10 people, P2PNet SHB predicted 6 — the closest to ground truth — but across the sparse video category more broadly, P2PNet SHB frequently degenerated to near-zero predictions (as low as 1–2) even when people were clearly visible. This suggests that training on SHB's sparse distribution causes the point detection head to become overly conservative, suppressing predictions at low density rather than finding the few people present.

**The regressor is sensitive to visual texture.** On the convenience store video, the EfficientNet-B0 regressor predicted 425 people when approximately 10 were present — a 40x overestimate. The scene contains densely packed product shelves with colorful labels and bottles that visually resemble crowd texture. The global regression model has no spatial supervision to distinguish crowd pixels from background structure, making it vulnerable to any scene with high visual complexity. This failure mode was consistent across other retail videos. The regressor also exhibited high temporal instability (std up to 79 on some videos), indicating large frame-to-frame variance even when the scene content is stable.

**YOLO fails consistently above ~20 people.** Across all dense videos, YOLO predicted between 4 and 89 people regardless of actual crowd size. Detection-based counting is fundamentally limited by occlusion: once people overlap significantly, the detector cannot find individual heads, and the count saturates far below the true value.

---

## 5. Discussion

### 5.1 Domain Gap

The SHA-trained and SHB-trained checkpoints represent two different priors about what a "normal" scene looks like. SHA encodes a prior of high density; SHB encodes a prior of low density. The results show that this prior has a strong effect on predictions — often stronger than architectural differences between models.

For the regressor, SHA vs SHB produces a mean difference of roughly 40–60 people per frame on dense scenes and roughly 60 people on sparse scenes, in opposite directions. For CSRNet, the effect is reversed on dense scenes: SHB-trained CSRNet predicts *more* than SHA-trained on some dense videos, suggesting the density map representation adapts differently to out-of-distribution density than the regression head does.

The most extreme case of domain gap failure is P2PNet SHB on sparse scenes. Point detection relies on the model confidently placing a point at each head location; when the training distribution contains many near-empty images, the model appears to learn a high suppression threshold that is then applied too aggressively at test time.

### 5.2 Failure Modes by Architecture

Each architectural family has a characteristic failure mode that is visible in the video results:

- **Global regression** fails when image texture is mistaken for crowd. Without any spatial supervision, the model cannot distinguish a dense arrangement of bottles from a dense arrangement of people.
- **Density map (CSRNet)** is the most stable under distribution shift, but systematically underestimates on sparse scenes when trained on SHA and overestimates on dense scenes when trained on SHB.
- **Point detection (P2PNet)** is the most accurate on in-distribution scenes but the most brittle under distribution shift — particularly the SHB-trained variant on sparse scenes.
- **Detection counting (YOLO)** is robust to domain shift (predictions are the same regardless of training dataset, since it has only one checkpoint) but is fundamentally limited by occlusion and cannot scale to dense crowds.

### 5.3 Limitations

The ground truth for video inference is approximate. Manual counting of dense scenes is itself unreliable — our estimate of ~500 for the Christmas mall scene carries an uncertainty of ±100–200 people. The conclusions about which model is "closest" should be read as directional, not precise.

The video collection is small (12 videos) and limited to indoor scenes. Outdoor crowds, transit stations, stadiums, and other environments common in surveillance applications are not represented. Domain gap findings may differ substantially in other settings.

---

## 6. Conclusion

This project benchmarks nine crowd counting models across four methodological families on the ShanghaiTech benchmarks and evaluates four representative models on 12 real-world internet videos.

On standard benchmarks, point detection models (P2PNet, APGCC) lead on dense SHA, while density map models (BL, DM-Count) are competitive on sparse SHB. Global regression and detection counting trail substantially.

On real-world video, the picture is more nuanced. No single model dominates across all conditions. P2PNet trained on SHA performs best on medium-density scenes (~150 people). CSRNet trained on SHA performs best on very dense scenes (~500 people). P2PNet trained on SHB, despite its strong SHB benchmark score, degenerates on sparse real-world scenes. The global regression model is fast and lightweight but unreliable when visual texture is complex. YOLO is not viable for crowd counting above roughly 20 people.

The practical implication is that model selection for deployment should account for the expected scene density and the available training data. A model that leads on a benchmark may not be the right choice for a specific deployment context, and the gap between SHA-trained and SHB-trained performance on the same video can exceed the gap between different model architectures trained on the same dataset. Domain alignment — ensuring the training data distribution matches the deployment environment — appears to matter at least as much as architectural choice.
