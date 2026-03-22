# Crowd Counting

This repository benchmarks four representative crowd counting models and evaluates real-world robustness on Internet videos. The goal is to compare benchmark performance with wild deployment behavior, identify distribution shift failure modes, and recommend model selection based on scene density.

## 1. Goals

- Benchmark models on ShanghaiTech Part A (SHA, dense) and Part B (SHB, sparse).
- Analyze domain gap via video inference on 12 Internet clips (7 dense indoor; 5 sparse retail/CCTV).
- Focus on four families in the report order:
  - Detection counting (YOLO11m-head)
  - Global regression (EfficientNet-B0+FC)
  - Density map (CSRNet)
  - Point detection (P2PNet)

## 2. Datasets

- ShanghaiTech A: 482 images (300 train, 182 test), 33–3139 people.
- ShanghaiTech B: 716 images (400 train, 316 test), 9–578 people.
- Other datasets (UCF-QNRF, UCF-CC-50, Unidata, mall) were excluded for this analysis.

## 3. Metrics

- MAE: mean absolute error.
- MSE: root mean squared error.
- Video stability: per-frame std in predicted count.

## 4. Model overview

This project focuses on four representative models (report primary analyses).

| Model | Family | Notes |
|---|---|---|
| YOLO11m-head | Detection counting | --- |
| EfficientNet-B0+FC | Global regression | baseline |
| CSRNet | Density map | CVPR 2018 |
| P2PNet | Point detection | ICCV 2021 |

## 5. ShanghaiTech results (4-model focus)

### Part A (Dense)

| Model | MAE | MSE |
|---|---|---|
| YOLO11m-head | 236.30 | 392.29 |
| EfficientNet-B0+FC | 91.67 | 138.42 |
| CSRNet | 70.15 | 109.17 |
| P2PNet | 58.09 | 95.27 |

### Part B (Sparse)

| Model | MAE | MSE |
|---|---|---|
| YOLO11m-head | 40.20 | 72.93 |
| EfficientNet-B0+FC | 15.32 | 22.81 |
| CSRNet | 10.46 | 16.90 |
| P2PNet | 9.26 | 16.53 |

### Key benchmark conclusions

- Point detection (P2PNet) leads on SHA.
- Density map is competitive and most stable under distribution shift.
- Global regression and YOLO underperform on dense scenes; regression is more volatile and texture-sensitive.

## 6. Backbone swap experiment (global regression)

- Tested on SHA with fixed head and training config.
- EfficientNet-B0 best MAE (91.67) + small params.
- ResNet50 worse than VGG16 despite higher ImageNet accuracy.
- ConvNeXt-Tiny often failed to converge with lr=1e-4.

## 7. Real-world video inference

12 videos (7 dense, 5 sparse), evaluated with SHA/SHB checkpoints where available.

### Domain gap patterns

- SHA-trained models tend to overestimate sparse scenes; SHB-trained models under-predict dense scenes.
- Regressor (EfficientNet-B0+FC): high sensitivity to texture (e.g., shelves → huge overestimates).
- CSRNet: most stable, best on extreme density.
- P2PNet SHA: best on medium density; P2PNet SHB collapses on very sparse scenes.
- YOLO: effective <20 people, fails badly in dense crowds due to occlusion.

### Example manual ground truth (selected)

- Mall multi-level (dense ~150): best P2PNet SHA (147).
- Christmas mall (dense ~500): best CSRNet SHA (495).
- Convenience store (sparse ~10): best P2PNet SHB (6).

### Failure modes

- Global regression misinterprets non-human patterns as crowd texture.
- Point detection can be brittle under domain shift (especially SHB-trained sparse models).
- Detection counting saturates under heavy occlusion.

## 8. Practical takeaway

- Benchmark superiority does not guarantee real-world robustness.
- Select model by expected scene density and deployment distribution.
- Domain alignment (training data similar to target scenario) is as important as model architecture.

## 9. Structure

- `preprocess/`: dataset and density map prep.
- `CSRNet/`, `P2PNet/`, `DM-Count/`, etc.: model code.
- `train_regressor.py`, `train_yolo.py`, `eval_yolo.py`, `video_inference.py`, `check_backbone.py`.
- `logs/`, `results/`, `videos/` store checkpoints and outputs.
