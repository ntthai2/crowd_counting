# Crowd Counting — A Comparative Study

> Bachelor-level course project comparing ten crowd counting models spanning three methodological families: **density map estimation**, **point-based detection**, and **global regression**.

---

## Abstract

Crowd counting — estimating the number of people in an image — is a core task in intelligent video surveillance, event management, and public safety. This project benchmarks a representative set of models from three paradigms:

- **Density map estimation** (MCNN, CSRNet, BL, DM-Count, STEERER): predict a per-pixel density map whose integral equals the crowd count. Ground-truth maps are generated from head annotations via adaptive Gaussian kernels.
- **Point-based detection** (P2PNet, CLTR): directly regress head coordinates, then derive the count by aggregation.
- **Global regression** (TransCrowd, VGG16+FC, ResNet50+FC): map the whole image directly to a scalar count without spatial supervision.

Experiments are run in two phases: (1) a **merged unified dataset** covering ShanghaiTech A/B, UCF-QNRF, Unidata, and mall, plus (2) **individual standard benchmarks** to allow comparison with reported numbers in the literature.

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

| Dataset | Images | Count range | Annotation | Used for |
|---|---|---|---|---|
| ShanghaiTech A | 482 (300+182) | 33–3139 | Point (`.mat`) | Phase 1 (merged) + Phase 2 |
| ShanghaiTech B | 716 (400+316) | 9–578 | Point (`.mat`) | Phase 1 (merged) + Phase 2 |
| UCF-QNRF | 1,535 (1201+334) | 49–12,865 | Point (`_ann.mat`) | Phase 1 (merged) + Phase 2 |
| UCF-CC-50 | 50 (5-fold CV) | 94–4,543 | Point (`_ann.mat`) | Phase 2 only |
| Unidata | ~500 | varies | JSON keypoint | Phase 1 (merged) |
| mall | 2,000 | 13–53 | Point-per-frame (`.mat`) | Phase 1 (merged) |
| JHU-Crowd++ | 4,372 (2272+500+1600) | 0–25,791 | Head-center + size (`.txt`) | Phase 3 (deferred) |

> UCF-CC-50 is excluded from the merged split because its 5-fold cross-validation protocol would be distorted by mixing with other datasets. JHU-Crowd++ is deferred to Phase 3: the dataset is very large (4,372 images, up to 25,791 heads) and the annotations include many occluded/low-confidence heads that reduce training signal reliability.

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

Experiments are run in two phases:
- **Phase 1**: merged unified dataset (ShanghaiTech A+B, UCF-QNRF, Unidata, mall), 70/15/15 split, seed 42.
- **Phase 2**: each model re-trained on individual standard splits (SHA, SHB, QNRF, UCF-CC-50 5-fold).
- **Phase 3** *(future)*: JHU-Crowd++, YOLO, RF-DETR, RT-DETR.

---

## Results

### Phase 1 — Unified dataset (MAE / MSE)

| Model | MAE ↓ | MSE ↓ |
|---|---|---|
| MCNN | | |
| CSRNet | | |
| BL | | |
| DM-Count | | |
| P2PNet | | |
| CLTR | | |
| STEERER | | |
| TransCrowd (token) | | |
| TransCrowd (gap) | | |
| VGG16+FC | | |
| ResNet50+FC | | |

### Phase 2 — Individual benchmarks (MAE / MSE)

#### ShanghaiTech Part A

| Model | MAE ↓ | MSE ↓ |
|---|---|---|
| MCNN | | |
| CSRNet | | |
| BL | | |
| DM-Count | | |
| P2PNet | | |
| CLTR | | |
| STEERER | | |
| TransCrowd | | |
| VGG16+FC | | |
| ResNet50+FC | | |

#### ShanghaiTech Part B

| Model | MAE ↓ | MSE ↓ |
|---|---|---|
| MCNN | | |
| CSRNet | | |
| BL | | |
| DM-Count | | |
| P2PNet | | |
| CLTR | | |
| STEERER | | |
| TransCrowd | | |
| VGG16+FC | | |
| ResNet50+FC | | |

#### UCF-QNRF

| Model | MAE ↓ | MSE ↓ |
|---|---|---|
| MCNN | | |
| CSRNet | | |
| BL | | |
| DM-Count | | |
| P2PNet | | |
| CLTR | | |
| STEERER | | |
| TransCrowd | | |
| VGG16+FC | | |
| ResNet50+FC | | |

#### UCF-CC-50 (5-fold mean ± std)

| Model | MAE ↓ | MSE ↓ |
|---|---|---|
| MCNN | | |
| CSRNet | | |
| BL | | |
| DM-Count | | |
| P2PNet | | |
| CLTR | | |
| STEERER | | |
| TransCrowd | | |
| VGG16+FC | | |
| ResNet50+FC | | |

#### JHU-Crowd++

| Model | MAE ↓ | MSE ↓ |
|---|---|---|
| MCNN | | |
| CSRNet | | |
| BL | | |
| DM-Count | | |
| P2PNet | | |
| CLTR | | |
| STEERER | | |
| TransCrowd | | |
| VGG16+FC | | |
| ResNet50+FC | | |

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
