# Crowd Counting — A Comparative Study

> Bachelor-level course project comparing ten crowd counting models spanning three methodological families: **density map estimation**, **point-based detection**, and **global regression**.

---

## Abstract

Crowd counting — estimating the number of people in an image — is a core task in intelligent video surveillance, event management, and public safety. This project benchmarks a representative set of models from three paradigms:

- **Density map estimation** (MCNN, CSRNet, BL, DM-Count, STEERER): predict a per-pixel density map whose integral equals the crowd count. Ground-truth maps are generated from head annotations via adaptive Gaussian kernels.
- **Point-based detection** (P2PNet, CLTR): directly regress head coordinates, then derive the count by aggregation.
- **Global regression** (TransCrowd, VGG16+FC, ResNet50+FC): map the whole image directly to a scalar count without spatial supervision.

All ten models are evaluated on the two ShanghaiTech benchmarks: **Part A** (dense urban crowds, 33–3,139 people) and **Part B** (sparse suburban crowds, 9–578 people). The two datasets are complementary in difficulty and are the standard shared benchmark across virtually all crowd counting publications, enabling direct comparison to reported baselines.

---

## Model Overview

| # | Model | Family | Publication | Backbone |
|---|---|---|---|---|
| 1 | MCNN | Density map | CVPR 2016 | Multi-column CNN |
| 2 | CSRNet | Density map | CVPR 2018 | VGG16 + dilated conv |
| 3 | BL (Bayesian Loss) | Density map | ICCV 2019 | VGG19 |
| 4 | DM-Count | Density map | NeurIPS 2020 | VGG19 |
| 5 | P2PNet | Point detection | ICCV 2021 | VGG16 |
| 6 | CLTR | Point detection | ECCV 2022 | ResNet-50 + Transformer |
| 7 | STEERER | Density map | ICCV 2023 | ConvNeXt (multi-scale) |
| 8 | TransCrowd | Regression | IJCAI 2022 | DeiT-Base ViT |
| 9 | VGG16+FC *(ours)* | Regression | — | VGG16 |
| 10 | ResNet50+FC *(ours)* | Regression | — | ResNet-50 |

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
| JHU-Crowd++ | 4,372 (2272+500+1600) | 0–25,791 | Head-center + size (`.txt`) | ❌ Excluded |

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
├── data/                    # All raw datasets (not in repo)
│   ├── ShanghaiTech/
│   ├── UCF-QNRF/
│   ├── UCF-CC-50/
│   ├── Unidata/
│   ├── mall_dataset/
│   └── jhu_crowd/
├── preprocess/              # Shared preprocessing scripts
│   ├── convert_unidata.py       # JSON → numpy point arrays
│   ├── gen_density_maps.py      # Adaptive Gaussian density maps (.npy)
│   ├── gen_h5_density.py        # CSRNet-format .h5 density maps
│   ├── gen_point_npy.py         # BL / DM-Count point npy files
│   ├── gen_cltr_h5.py           # CLTR-format .h5 files
│   └── create_unified_split.py  # Merge datasets → train/val/test npy lists
├── MCNN/
├── CSRNet/
├── Bayesian-Loss/
├── DM-Count/
├── P2PNet/
├── CLTR/
├── STEERER/
└── TransCrowd/              # Also hosts VGG16+FC and ResNet50+FC
    └── Networks/
        └── models.py        # ViT + VGG16CountNet + ResNet50CountNet
```

---

## Setup

```bash
conda create -n crowd python=3.9 -y
conda activate crowd

# Core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install scipy h5py timm opencv-python-headless Pillow tqdm

# Per-model extras (install as needed)
pip install mmcv-full         # STEERER
pip install nni               # TransCrowd (NNI for hyperparameter search)
```

---

## Preprocessing & Training

See [EXPERIMENTS.md](EXPERIMENTS.md) for all preprocessing commands, implementation details, and progress tracking.

Each model is trained and evaluated twice — once on SHA and once on SHB — using the standard train/test splits from the original dataset releases. All ten models share the same training protocol: Adam optimizer, early stopping with patience=50, best checkpoint retained.

---

## Results

### ShanghaiTech Part A (MAE / MSE)

| Model | Family | MAE ↓ | MSE ↓ | Published MAE |
|---|---|---|---|---|
| MCNN | Density map | | | 110.2 |
| CSRNet | Density map | | | 68.2 |
| BL | Density map | | | 62.8 |
| DM-Count | Density map | | | 59.7 |
| STEERER | Density map | | | 56.0 |
| P2PNet | Point detection | | | 52.7 |
| CLTR | Point detection | | | 56.9 |
| TransCrowd | Regression | | | 66.1 |
| VGG16+FC | Regression | | | — |
| ResNet50+FC | Regression | | | — |

### ShanghaiTech Part B (MAE / MSE)

| Model | Family | MAE ↓ | MSE ↓ | Published MAE |
|---|---|---|---|---|
| MCNN | Density map | | | 26.4 |
| CSRNet | Density map | | | 10.6 |
| BL | Density map | | | 7.7 |
| DM-Count | Density map | | | 7.4 |
| STEERER | Density map | | | 6.5 |
| P2PNet | Point detection | | | 6.7 |
| CLTR | Point detection | | | 6.5 |
| TransCrowd | Regression | | | 8.1 |
| VGG16+FC | Regression | | | — |
| ResNet50+FC | Regression | | | — |

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
- [CLTR](https://github.com/dk-liang/CLTR)
- [STEERER](https://github.com/taohan10200/STEERER)
- [TransCrowd](https://github.com/dk-liang/TransCrowd)
