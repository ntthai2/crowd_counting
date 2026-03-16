# Crowd Counting

> Comparing nine models spanning four methodological families: **density map estimation**, **point-based counting**, **global regression**, and **object detection counting**.

---

## Abstract

Crowd counting — estimating the number of people in an image — is a core task in intelligent video surveillance, event management, and public safety. This project benchmarks a representative set of models from four paradigms:

- **Density map estimation** (MCNN, CSRNet, BL, DM-Count): predict a per-pixel density map whose integral equals the crowd count. Ground-truth maps are generated from head annotations via adaptive Gaussian kernels.
- **Point-based counting** (P2PNet, APGCC): directly regress head coordinates/proposals and derive count by aggregation.
- **Global regression** (VGG16+FC, ResNet50+FC): map the whole image directly to a scalar count without spatial supervision.
- **Object detection counting** (YOLO11m-head): run head detection and use the number of boxes as the predicted count.

All models are evaluated on the two ShanghaiTech benchmarks: **Part A** (dense urban crowds, 33–3,139 people) and **Part B** (sparse suburban crowds, 9–578 people). The two datasets are complementary in difficulty and are the standard shared benchmark across virtually all crowd counting publications, enabling direct comparison to reported baselines.

---

## Model Overview

| # | Model | Family | Publication | Backbone |
|---|---|---|---|---|
| 1 | MCNN | Density map | CVPR 2016 | Multi-column CNN |
| 2 | CSRNet | Density map | CVPR 2018 | VGG16 + dilated conv |
| 3 | BL (Bayesian Loss) | Density map | ICCV 2019 | VGG19 |
| 4 | DM-Count | Density map | NeurIPS 2020 | VGG19 |
| 5 | P2PNet | Point detection | ICCV 2021 | VGG16 |
| 6 | APGCC | Point detection | ECCV 2024 | VGG16-BN + IFI |
| 7 | VGG16+FC | Regression | — | VGG16 |
| 8 | ResNet50+FC | Regression | — | ResNet-50 |
| 9 | YOLO11m-head | Detection counting | — | YOLO11m |

---

## Datasets

| Dataset | Images | Count range | Annotation | Status |
|---|---|---|---|---|
| ShanghaiTech A | 482 (300+182) | 33–3,139 | Point (`.mat`) | ✅ Used |
| ShanghaiTech B | 716 (400+316) | 9–578 | Point (`.mat`) | ✅ Used |
| UCF-QNRF | 1,535 (1201+334) | 49–12,865 | Point (`_ann.mat`) | ❌ Excluded |
| UCF-CC-50 | 50 (5-fold CV) | 94–4,543 | Point (`_ann.mat`) | ❌ Excluded |
| Unidata | 20 | varies | JSON keypoint | ❌ Excluded |
| mall | 2,000 | 13–53 | Point-per-frame (`.mat`) | ❌ Excluded |

SHA and SHB are trained and evaluated independently using their standard splits, matching the protocol used in the original publications. This allows direct comparison to reported baselines.

Excluded datasets:
- **UCF-QNRF**: images contain up to 12,865 people and are extremely large; training is slow and the insight ("models struggle at extreme density") is expected and not actionable.
- **UCF-CC-50**: only 50 images with 5-fold cross-validation — methodologically complex, adds overhead, and provides no insight beyond QNRF.
- **Unidata**: 20 images — too small to draw statistically meaningful conclusions.
- **mall**: max 53 people per frame — trivially easy for any modern model; differences would be noise.

---

## Project Structure

```
crowd_counting/
├── data/                    # Raw datasets (not in repo)
│   └── ShanghaiTech/
│       ├── part_A/
│       └── part_B/
├── preprocess/              # Shared preprocessing scripts
│   ├── gen_density_maps.py      # Adaptive Gaussian density maps (.npy)
│   ├── gen_h5_density.py        # CSRNet-format .h5 density maps
│   ├── gen_point_npy.py         # BL / DM-Count point npy files
│   ├── gen_csrnet_json.py       # CSRNet JSON file lists
│   ├── gen_p2pnet_data.py       # P2PNet .list annotation files
│   └── reorganise_bl_dm.py      # train/ + val/ layout for BL/DM-Count
├── MCNN/
├── CSRNet/
├── Bayesian-Loss/
├── DM-Count/
├── P2PNet/
├── APGCC/
├── train_regressor.py           # Standalone VGG16+FC / ResNet50+FC trainer
├── train_yolo.py                # YOLO11m head-detector training
├── eval_yolo.py                 # YOLO count evaluation on SHA/SHB
├── visualize_pred.py            # Best/worst prediction visualization
└── logs/                        # Checkpoints and training logs
```

---

## Preprocessing & Training

See [EXPERIMENTS.md](EXPERIMENTS.md) for all preprocessing commands, implementation details, and progress tracking.

Core crowd-counting models are trained and evaluated twice — once on SHA and once on SHB — using the standard train/test splits from the original dataset releases. YOLO11m-head is trained on merged head-detection data, then evaluated on SHA/SHB by counting detections.

---

## Results

### ShanghaiTech Part A (MAE / MSE)

| Model | Family | MAE ↓ | MSE ↓ | Published MAE |
|---|---|---|---|---|
| MCNN | Density map | 131.47 | 202.50 | 110.2 |
| CSRNet | Density map | 70.15 | 109.17 | 68.2 |
| BL | Density map | 66.34 | 100.65 | 62.8 |
| DM-Count | Density map | 65.88 | 104.70 | 59.7 |
| P2PNet | Point detection | 58.09 | 95.27 | 52.7 |
| APGCC | Point detection | 61.91 | 94.95 | 49.9 |
| VGG16+FC | Regression | 113.51 | 168.23 | — |
| ResNet50+FC | Regression | 135.47 | 200.70 | — |
| YOLO11m-head | Detection | 236.30 | 392.29 | — |

### ShanghaiTech Part B (MAE / MSE)

| Model | Family | MAE ↓ | MSE ↓ | Published MAE |
|---|---|---|---|---|
| MCNN | Density map | 30.79 | 46.92 | 26.4 |
| CSRNet | Density map | 10.46 | 16.90 | 10.6 |
| BL | Density map | 8.10 | 13.45 | 7.7 |
| DM-Count | Density map | 8.85 | 13.64 | 7.4 |
| P2PNet | Point detection | 9.26 | 16.53 | 6.7 |
| APGCC | Point detection | 10.26 | 16.92 | 6.0 |
| VGG16+FC | Regression | 16.03 | 24.95 | — |
| ResNet50+FC | Regression | 22.46 | 40.57 | — |
| YOLO11m-head | Detection | 40.20 | 72.93 | — |

---

## Evaluation Metric

$$\text{MAE} = \frac{1}{N}\sum_{i=1}^{N}|z_i - \hat{z}_i|, \quad \text{MSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(z_i - \hat{z}_i)^2}$$

where $z_i$ is the ground-truth count and $\hat{z}_i$ is the predicted count for image $i$.

---

## Acknowledgements

This project builds on publicly available implementations:
- [MCNN](https://github.com/svishwa/crowdcount-mcnn)
- [CSRNet](https://github.com/leeyeehoo/CSRNet-pytorch)
- [Bayesian Loss](https://github.com/ZhihengCV/Bayesian-Crowd-Counting)
- [DM-Count](https://github.com/cvlab-stonybrook/DM-Count)
- [P2PNet](https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet)
- [APGCC](https://github.com/AaronCIH/APGCC)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
