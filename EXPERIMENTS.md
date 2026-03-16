# Experiments Log
>
> Python: `/home/team_cam_ai/miniconda3/envs/ntt_det/bin/python` (3.13.5)
> PyTorch 2.7.1 · torchvision 0.22.1+cu128 · CUDA 12.8 · single GPU
> Working dir: `/ssd1/team_cam_ai/ntthai/crowd_counting`
> **NEVER** use `conda run` — it buffers stdout and kills nohup logging.

---

## Strategy

Train **8 crowd-counting models** on **ShanghaiTech A (SHA)** and **ShanghaiTech B (SHB)** separately, producing two result tables for direct comparison against published baselines.

**Models dropped:** CLTR, STEERER, TransCrowd (transformer-based; unstable training, modest results).

| Dataset | Images | Count range | Density |
|---|---|---|---|
| ShanghaiTech A | 482 (300 train + 182 test) | 33 – 3,139 | Dense urban |
| ShanghaiTech B | 716 (400 train + 316 test) | 9 – 578 | Sparse suburban |

---

## Models

| # | Model | Family | Dir | Entry point |
|---|---|---|---|---|
| 1 | MCNN | Density map | `MCNN/` | `MCNN/train.py` |
| 2 | CSRNet | Density map | `CSRNet/` | `CSRNet/train.py` |
| 3 | BL (Bayesian Loss) | Density map | `Bayesian-Loss/` | `Bayesian-Loss/train.py` |
| 4 | DM-Count | Density map | `DM-Count/` | `DM-Count/train.py` |
| 5 | P2PNet | Point detection | `P2PNet/` | `P2PNet/train.py` |
| 6 | VGG16+FC | Regression (CNN) | root | `train_regressor.py --model-type vgg16` |
| 7 | ResNet50+FC | Regression (CNN) | root | `train_regressor.py --model-type resnet50` |
| 8 | APGCC | Point detection | `APGCC/apgcc/` | `APGCC/apgcc/main.py` |

---

## Critical Notes — Read Before Running Anything

1. **BL on SHA requires `--crop-size 128`.**
   93 SHA images are smaller than 512×512 px (the default crop size). The trainer asserts `image_size >= crop_size` and will crash without this flag. SHB images are all large enough — do not add this flag for SHB.

2. **DM-Count `--data-dir` points to the raw dataset folder, not a subfolder.**
   Use `data/ShanghaiTech/part_A` (not `part_A/dm`). The loader reads `train/` and `val/` subdirectories with `.npy` density sidecars from the raw path.

3. **P2PNet requires pre-creating the output directory before running.**
   The script opens a log file inside `--output_dir` without creating it first. Always run `mkdir -p logs/p2pnet_sha_ckpts` before the training command.

4. **VAL log format is uniform across all models:**
   ```
   VAL epoch=XXX mae=XX.XX mse=XX.XX best_mae=XX.XX
   ```
   Monitor progress with: `grep "^VAL" logs/<model>.log | tail -5`

5. **Early stopping** is enabled with patience=50 in all models.

6. **APGCC list-file compatibility was patched.**
  APGCC now auto-detects `shanghai_tech_part_a_{train,test}.list` / `shanghai_tech_part_b_{train,test}.list` when `train.list` / `test.list` do not exist. This allows direct training on the same SHA/SHB preprocessing already used by P2PNet.

---

## Checkpoint File Locations

| Model | Best checkpoint path |
|---|---|
| MCNN | `logs/mcnn_<ds>_ckpts/best_model.h5` |
| CSRNet | `logs/csrnet_<ds>_ckpts/model_best.pth.tar` |
| BL | `logs/bl_<ds>_ckpts/best_model.pth` |
| DM-Count | `logs/dmcount_<ds>_ckpts/best_model.pth` |
| P2PNet | `logs/p2pnet_<ds>_ckpts/best_mae.pth` |
| VGG16+FC | `logs/vgg16_<ds>_ckpts/model_best.pth` |
| ResNet50+FC | `logs/resnet50_<ds>_ckpts/model_best.pth` |
| APGCC | `logs/apgcc_<ds>_ckpts/best.pth` |

---

## Hyperparameters

| Model | LR | Batch | Epochs | Optimizer | Notes |
|---|---|---|---|---|---|
| MCNN | 1e-5 | 1 | 400 | Adam | From original paper |
| CSRNet | 1e-6 | 1 | 400 | SGD | Small LR for VGG16 backbone |
| BL | 1e-5 | 5 | 500 | Adam | crop=128 for SHA; default 512 for SHB |
| DM-Count | 1e-4 | 8 | 500 | Adam | VGG19 backbone |
| P2PNet | 1e-4 (backbone 1e-5) | 8 | 3500 | AdamW | lr_drop=3500 |
| VGG16+FC | 1e-4 | 8 | 1000 | Adam | ImageNet pretrained; input 448×448; MSE loss |
| ResNet50+FC | 1e-4 | 8 | 1000 | Adam | ImageNet pretrained; input 448×448; MSE loss |
| APGCC | 1e-4 (backbone 1e-5) | 8 | 3500 | Adam | IFI decoder + APG |

---

## SHA Training Commands

All commands from: `/ssd1/team_cam_ai/ntthai/crowd_counting`

```bash
PYTHON=/home/team_cam_ai/miniconda3/envs/ntt_det/bin/python
BASE=/ssd1/team_cam_ai/ntthai/crowd_counting

# ─── 1. MCNN SHA ─────────────────────────────────────────────────────────────
nohup $PYTHON -u $BASE/MCNN/train.py \
  --dataset shanghaiA \
  --data-dir $BASE/data/ShanghaiTech/part_A \
  --output-dir $BASE/logs/mcnn_sha_ckpts \
  --epochs 400 --lr 1e-5 --gpu 0 \
  > $BASE/logs/mcnn_sha.log 2>&1 &

# ─── 2. CSRNet SHA ───────────────────────────────────────────────────────────
nohup $PYTHON -u $BASE/CSRNet/train.py \
  $BASE/CSRNet/part_A_train.json \
  $BASE/CSRNet/part_A_test.json \
  0 sha \
  --epochs 400 \
  --ckpt-dir $BASE/logs/csrnet_sha_ckpts \
  > $BASE/logs/csrnet_sha.log 2>&1 &

# ─── 3. BL SHA ───────────────────────────────────────────────────────────────
# NOTE: --crop-size 128 is required for SHA
nohup $PYTHON -u $BASE/Bayesian-Loss/train.py \
  --data-dir $BASE/data/ShanghaiTech/part_A/bl \
  --save-dir $BASE/logs/bl_sha_ckpts \
  --max-epoch 500 --val-epoch 1 --val-start 1 \
  --lr 1e-5 --device 0 --crop-size 128 \
  > $BASE/logs/bl_sha.log 2>&1 &

# ─── 4. DM-Count SHA ─────────────────────────────────────────────────────────
# NOTE: --data-dir is the raw part_A/ folder, not part_A/dm/
nohup $PYTHON -u $BASE/DM-Count/train.py \
  --dataset sha \
  --data-dir $BASE/data/ShanghaiTech/part_A \
  --save-dir $BASE/logs/dmcount_sha_ckpts \
  --max-epoch 500 --val-epoch 1 --val-start 1 \
  --lr 1e-4 --device 0 \
  > $BASE/logs/dmcount_sha.log 2>&1 &

# ─── 5. P2PNet SHA ───────────────────────────────────────────────────────────
# NOTE: mkdir required — the script does not auto-create output_dir
mkdir -p $BASE/logs/p2pnet_sha_ckpts
nohup $PYTHON -u $BASE/P2PNet/train.py \
  --dataset_file SHHA \
  --data_root $BASE/data/ShanghaiTech/part_A \
  --output_dir $BASE/logs/p2pnet_sha_ckpts \
  --checkpoints_dir $BASE/logs/p2pnet_sha_ckpts \
  --epochs 3500 --lr 1e-4 --gpu_id 0 \
  > $BASE/logs/p2pnet_sha.log 2>&1 &

# ─── 6. VGG16+FC SHA ─────────────────────────────────────────────────────────
nohup $PYTHON -u $BASE/train_regressor.py \
  --dataset ShanghaiA --data-dir $BASE/data/ShanghaiTech/part_A \
  --save-dir $BASE/logs/vgg16_sha_ckpts --model-type vgg16 \
  --epochs 1000 --lr 1e-4 --batch-size 8 --gpu 0 \
  > $BASE/logs/vgg16_sha.log 2>&1 &

# ─── 7. ResNet50+FC SHA ─────────────────────────────────────────────────────
nohup $PYTHON -u $BASE/train_regressor.py \
  --dataset ShanghaiA --data-dir $BASE/data/ShanghaiTech/part_A \
  --save-dir $BASE/logs/resnet50_sha_ckpts --model-type resnet50 \
  --epochs 1000 --lr 1e-4 --batch-size 8 --gpu 0 \
  > $BASE/logs/resnet50_sha.log 2>&1 &

# ─── 8. APGCC SHA ────────────────────────────────────────────────────────────
mkdir -p $BASE/logs/apgcc_sha_ckpts
nohup $PYTHON -u $BASE/APGCC/apgcc/main.py \
  -c $BASE/APGCC/apgcc/configs/SHHA_IFI.yml \
  DATASETS.DATA_ROOT $BASE/data/ShanghaiTech/part_A \
  DATASETS.DATASET SHHA \
  OUTPUT_DIR $BASE/logs/apgcc_sha_ckpts \
  GPU_ID 0 \
  > $BASE/logs/apgcc_sha.log 2>&1 &
```

---

## SHB Training Commands

```bash
PYTHON=/home/team_cam_ai/miniconda3/envs/ntt_det/bin/python
BASE=/ssd1/team_cam_ai/ntthai/crowd_counting

# ─── 1. MCNN SHB ─────────────────────────────────────────────────────────────
nohup $PYTHON -u $BASE/MCNN/train.py \
  --dataset shanghaiB \
  --data-dir $BASE/data/ShanghaiTech/part_B \
  --output-dir $BASE/logs/mcnn_shb_ckpts \
  --epochs 400 --lr 1e-5 --gpu 0 \
  > $BASE/logs/mcnn_shb.log 2>&1 &

# ─── 2. CSRNet SHB ───────────────────────────────────────────────────────────
nohup $PYTHON -u $BASE/CSRNet/train.py \
  $BASE/CSRNet/part_B_train.json \
  $BASE/CSRNet/part_B_test.json \
  0 shb \
  --epochs 400 \
  --ckpt-dir $BASE/logs/csrnet_shb_ckpts \
  > $BASE/logs/csrnet_shb.log 2>&1 &

# ─── 3. BL SHB ───────────────────────────────────────────────────────────────
# No --crop-size flag needed for SHB (all images are >= 512px)
nohup $PYTHON -u $BASE/Bayesian-Loss/train.py \
  --data-dir $BASE/data/ShanghaiTech/part_B/bl \
  --save-dir $BASE/logs/bl_shb_ckpts \
  --max-epoch 500 --val-epoch 1 --val-start 1 \
  --lr 1e-5 --device 0 \
  > $BASE/logs/bl_shb.log 2>&1 &

# ─── 4. DM-Count SHB ─────────────────────────────────────────────────────────
nohup $PYTHON -u $BASE/DM-Count/train.py \
  --dataset shb \
  --data-dir $BASE/data/ShanghaiTech/part_B \
  --save-dir $BASE/logs/dmcount_shb_ckpts \
  --max-epoch 500 --val-epoch 1 --val-start 1 \
  --lr 1e-4 --device 0 \
  > $BASE/logs/dmcount_shb.log 2>&1 &

# ─── 5. P2PNet SHB ───────────────────────────────────────────────────────────
mkdir -p $BASE/logs/p2pnet_shb_ckpts
nohup $PYTHON -u $BASE/P2PNet/train.py \
  --dataset_file SHHB \
  --data_root $BASE/data/ShanghaiTech/part_B \
  --output_dir $BASE/logs/p2pnet_shb_ckpts \
  --checkpoints_dir $BASE/logs/p2pnet_shb_ckpts \
  --epochs 3500 --lr 1e-4 --gpu_id 0 \
  > $BASE/logs/p2pnet_shb.log 2>&1 &

# ─── 6. VGG16+FC SHB ─────────────────────────────────────────────────────────
nohup $PYTHON -u $BASE/train_regressor.py \
  --dataset ShanghaiB --data-dir $BASE/data/ShanghaiTech/part_B \
  --save-dir $BASE/logs/vgg16_shb_ckpts --model-type vgg16 \
  --epochs 1000 --lr 1e-4 --batch-size 8 --gpu 0 \
  > $BASE/logs/vgg16_shb.log 2>&1 &

# ─── 7. ResNet50+FC SHB ─────────────────────────────────────────────────────
nohup $PYTHON -u $BASE/train_regressor.py \
  --dataset ShanghaiB --data-dir $BASE/data/ShanghaiTech/part_B \
  --save-dir $BASE/logs/resnet50_shb_ckpts --model-type resnet50 \
  --epochs 1000 --lr 1e-4 --batch-size 8 --gpu 0 \
  > $BASE/logs/resnet50_shb.log 2>&1 &

# ─── 8. APGCC SHB ────────────────────────────────────────────────────────────
mkdir -p $BASE/logs/apgcc_shb_ckpts
nohup $PYTHON -u $BASE/APGCC/apgcc/main.py \
  -c $BASE/APGCC/apgcc/configs/SHHB_IFI.yml \
  DATASETS.DATA_ROOT $BASE/data/ShanghaiTech/part_B \
  DATASETS.DATASET SHHB \
  OUTPUT_DIR $BASE/logs/apgcc_shb_ckpts \
  GPU_ID 0 \
  > $BASE/logs/apgcc_shb.log 2>&1 &
```

---

## Resume Commands

These extend the training commands with a checkpoint flag. Append to whichever dataset run you need to continue.

| Model | Resume flag | Checkpoint file |
|---|---|---|
| MCNN | `--resume <ckpt>` | `logs/mcnn_<ds>_ckpts/best_model.h5` |
| CSRNet | `--pre <ckpt>` | `logs/csrnet_<ds>_ckpts/model_best.pth.tar` |
| BL | `--resume <ckpt>` | `logs/bl_<ds>_ckpts/best_model.pth` |
| DM-Count | `--resume <ckpt>` | `logs/dmcount_<ds>_ckpts/best_model.pth` |
| P2PNet | `--resume <ckpt>` | `logs/p2pnet_<ds>_ckpts/best_mae.pth` |
| VGG16+FC | `--resume <ckpt>` | `logs/vgg16_<ds>_ckpts/checkpoint.pth` |
| ResNet50+FC | `--resume <ckpt>` | `logs/resnet50_<ds>_ckpts/checkpoint.pth` |
| APGCC | `RESUME True RESUME_PATH <ckpt>` | `logs/apgcc_<ds>_ckpts/best.pth` |

**Example — resume P2PNet SHA:**
```bash
BASE=/ssd1/team_cam_ai/ntthai/crowd_counting
PYTHON=/home/team_cam_ai/miniconda3/envs/ntt_det/bin/python
mkdir -p $BASE/logs/p2pnet_sha_ckpts
nohup $PYTHON -u $BASE/P2PNet/train.py \
  --dataset_file SHHA \
  --data_root $BASE/data/ShanghaiTech/part_A \
  --output_dir $BASE/logs/p2pnet_sha_ckpts \
  --checkpoints_dir $BASE/logs/p2pnet_sha_ckpts \
  --epochs 3500 --lr 1e-4 --gpu_id 0 \
  --resume $BASE/logs/p2pnet_sha_ckpts/best_mae.pth \
  > $BASE/logs/p2pnet_sha.log 2>&1 &
```

---

## Monitoring

```bash
BASE=/ssd1/team_cam_ai/ntthai/crowd_counting
PYTHON=/home/team_cam_ai/miniconda3/envs/ntt_det/bin/python

# List running training processes
ps aux | grep -E "[p]ython.*train" | awk '{print $2, $12, $13, $14}'

# GPU memory and utilization
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader

# All VAL lines across all logs, sorted by model
grep "^VAL" $BASE/logs/*.log | sort

# Follow a single log live
tail -f $BASE/logs/mcnn_sha.log
```

---

## Results

Lower is better for both MAE and MSE.

### ShanghaiTech A

| Model | MAE | MSE | Published MAE |
|---|---|---|---|
| MCNN | 131.47 | 202.50 | 110.2 |
| CSRNet | 70.15 | 109.17 | 68.2 |
| BL | 66.34 | 100.65 | 62.8 |
| DM-Count | 65.88 | 104.70 | 59.7 |
| P2PNet | 58.09 | 95.27 | 52.7 |
| VGG16+FC | 113.51 | 168.23 | — |
| ResNet50+FC | 135.47 | 200.70 | — |
| YOLO11m-head | 236.30 | 392.29 | — |

### ShanghaiTech B

| Model | MAE | MSE | Published MAE |
|---|---|---|---|
| MCNN | 30.79 | 46.92 | 26.4 |
| CSRNet | 10.46 | 16.90 | 10.6 |
| BL | 8.10 | 13.45 | 7.7 |
| DM-Count | 8.85 | 13.64 | 7.4 |
| P2PNet | 9.26 | 16.53 | 6.7 |
| VGG16+FC | 16.03 | 24.95 | — |
| ResNet50+FC | 22.46 | 40.57 | — |
| YOLO11m-head | 40.20 | 72.93 | — |

---

## Codebase Notes

### Bug Fixes Applied

| File | Fix |
|---|---|
| `P2PNet/util/misc.py` | `float(torchvision.__version__[:3])` → tuple comparison; fixes crash with torchvision 0.22 |
| `P2PNet/models/vgg_.py` | Fall back to `torch.hub.load_state_dict_from_url` when hardcoded private weight path is absent |
| `DM-Count/train_helper.py` | Moved VAL print line to after `self.best_mae` update so it shows current best, not stale value |

### Data Layout Reference

| Model | SHA data path | SHB data path |
|---|---|---|
| MCNN | `data/ShanghaiTech/part_A/*/gt_density_map/*.npy` | `part_B/` same layout |
| CSRNet | `data/ShanghaiTech/part_A/*/ground_truth/*.h5` | `part_B/` same layout |
| BL | `data/ShanghaiTech/part_A/bl/train/` + `val/` | `part_B/bl/` same |
| DM-Count | `data/ShanghaiTech/part_A/train/` + `val/` | `part_B/` same |
| P2PNet | `P2PNet/crowd_datasets/SHHA/SHHA.list` | `SHHB/SHHB.list` |
| VGG16+FC / ResNet50+FC | `data/ShanghaiTech/part_A/` (raw) | `data/ShanghaiTech/part_B/` (raw) |
| APGCC | `data/ShanghaiTech/part_A/shanghai_tech_part_a_train.list` | `data/ShanghaiTech/part_B/shanghai_tech_part_b_train.list` |

### Preprocessing Scripts (`preprocess/`)

These have all been run already. Only re-run if data is lost.

| Script | Purpose |
|---|---|
| `gen_density_maps.py` | Gaussian `.npy` density maps for MCNN |
| `gen_h5_density.py` | `.h5` density maps for CSRNet |
| `gen_point_npy.py` | BL `(N,3)` + DM-Count `(N,2)` `.npy` point files |
| `gen_csrnet_json.py` | CSRNet JSON file lists |
| `gen_p2pnet_data.py` | P2PNet `.list` annotation files |
| `reorganise_bl_dm.py` | Reorganizes raw data into `train/` + `val/` for BL/DM-Count |

---

## Project Narrative

> This section records what was actually done — decisions made, problems encountered, and how they were resolved — for use in the final report.

### Background and Motivation

The project aims to benchmark crowd-counting methods spanning three architectural families: **density-map regression**, **point detection**, and **global count regression**. Seven models were ultimately evaluated:

- *Density-map* (MCNN, CSRNet, BL, DM-Count) — the mainstream approach; the network predicts a spatial density map whose integral gives the count.
- *Point detection* (P2PNet) — newer paradigm that predicts a discrete set of head locations instead of a smooth map.
- *Count regression* (VGG16+FC, ResNet50+FC) — global regression directly from image features; simpler baseline but lower accuracy.

Three additional transformer-based models (CLTR, STEERER, TransCrowd) were attempted but dropped: training was unstable and after hundreds of epochs results could not approach the density-map baselines, making the additional complexity unjustifiable within the project scope.

ShanghaiTech A and B were chosen as the evaluation benchmarks because they are the universally-accepted standard for crowd counting, they are small enough to train on a single GPU in a reasonable time, and published MAE/MSE numbers are available for every model so comparisons are meaningful.

### Dataset Strategy

Initial work included downloading UCF-QNRF (4.3 GB raw + 45 GB preprocessed), the mall dataset (4.6 GB), and Unidata (502 MB) alongside ShanghaiTech. After evaluating time constraints and the scope of the project, the decision was made to focus exclusively on SHA and SHB. The other datasets were downloaded and preprocessed but ultimately unused. The preprocessed QNRF data alone (45 GB) will be deleted once code cleanup is complete.

### Environment Setup

All models were run under a single shared Conda environment (`ntt_det`) with Python 3.13.5, PyTorch 2.7.1, and CUDA 12.8.

### Code Fixes Required Before Training

Several bugs in the original model repositories had to be patched:

**P2PNet torchvision version check** (`P2PNet/util/misc.py`): The code did `float(torchvision.__version__[:3])` which crashes when the version string starts with `0.22` (only one digit before the decimal in the substring). Fixed by comparing version tuples instead.

**P2PNet VGG backbone weights** (`P2PNet/models/vgg_.py`): The model tried to load weights from a hardcoded private server path (`/public/home/...`). When that path does not exist on the training machine it raised a `FileNotFoundError` at import time. Fixed by adding a fallback to `torch.hub.load_state_dict_from_url` using the standard `torchvision` URL for VGG16-BN.

**BL crop-size assertion on SHA** (`Bayesian-Loss/train.py`): The Bayesian-Loss trainer asserts that every image is at least as large as the crop size. 93 images in ShanghaiTech A are smaller than the default 512×512. Solved by adding `--crop-size 128` for SHA training only. SHB images are uniformly larger and do not need this flag.

**DM-Count `--data-dir` confusion**: The DM-Count trainer expects the raw dataset root (containing `train/` and `val/` subdirectories populated by `reorganise_bl_dm.py`), not the `dm/` subdirectory that might be expected from the variable name. This was discovered after the first training run produced an empty dataset error.

**DM-Count best-MAE display bug** (`DM-Count/train_helper.py`): The `VAL` log line was printed before `self.best_mae` was updated, so it always showed the previous epoch's best. Fixed by moving the print statement two lines down.

### Preprocessing Decisions

**Density maps (adaptive Gaussian):** Rather than using fixed-sigma Gaussian density maps (as in the original CSRNet paper), an adaptive per-head sigma was used for all density-map models. The sigma for each annotated head is computed as the mean distance to its three nearest neighbours multiplied by 0.3, clamped to [2, 20] pixels. This is the standard practice for dense scenes. The same density map generation code (`gen_density_maps.py`, `gen_h5_density.py`) was used for all relevant models to ensure consistency.

**BL point format:** Bayesian-Loss requires a 3-column `.npy` file `[x, y, knn_dist]` where the third column is the mean k-NN distance used in the Bayesian uncertainty loss. This was computed in `gen_point_npy.py` with k=3, matching the BL paper.

**DM-Count needs `train/` / `val/` split directories:** DM-Count's dataset loader globs for images in `<root>/train/` and `<root>/val/`. ShanghaiTech ships as `train_data/` and `test_data/` (no validation set is provided separately; researchers use the test set as validation). The `reorganise_bl_dm.py` script creates the expected structure by symlinking images and generating `.npy` annotation files into `train/` and `val/` subdirectories.

### Uniform Validation Logging

All 7 models output validation metrics in a consistent format:
```
VAL epoch=XXX mae=XX.XX mse=XX.XX best_mae=XX.XX
```
This required patching several models' validation loops. The format was chosen to allow a single `grep "^VAL"` command to aggregate results across all log files regardless of model, and also to feed the `plot_training.py` script which generates training curves.

### Early Stopping

A patience-50 early stopping rule was added uniformly across all models. Training halts if validation MAE does not improve for 50 consecutive epochs. This prevents over-training on the small SHA/SHB datasets (300 / 400 training images respectively) and frees the GPU for the next model without manual monitoring.

### Infrastructure

All training jobs are launched with `nohup ... &` to allow logging out. `conda run` was discovered to buffer stdout aggressively, making log files appear empty until the process ends. The fix is to activate the environment in the shell before launching, or to call the Python binary by its absolute path (`/home/team_cam_ai/miniconda3/envs/ntt_det/bin/python`) directly without `conda run`.
