# Experiments Log — Crowd Counting Course Project

> **This file is the primary memory and progress tracker. Read it first every session.**
>
> Environment: conda env `ntt_det` — always `conda activate ntt_det` before any command.
> Working dir: `/ssd1/team_cam_ai/ntthai/crowd_counting`

---

## Current State (as of 2026-03-10)

### ✅ Done
| # | Task |
|---|---|
| 1 | Project planning: 10 models, 3 families, 2-phase + future Phase 3 |
| 2 | `README.md` — overview + results tables only; all commands live in this file |
| 3 | `preprocess/convert_unidata.py` written and run → `data/Unidata/processed/labels/**/*.npy` |
| 4 | `preprocess/gen_density_maps.py` written and run for all 5 datasets (JHU deferred) |
| 5 | `DM-Count/preprocess_dataset.py` run → `data/UCF-QNRF-processed/` (train/val/test + `.npy` points) |
| 6 | All 5 preprocess scripts support QNRF processed layout (JHU support kept for Phase 3) |
| 7 | `TransCrowd/` extended: `VGG16CountNet`, `ResNet50CountNet`, unified dataset support |
| 8 | `MCNN/src/` fully ported to Python 3 — imports verified working |
| 9 | `preprocess/gen_h5_density.py` run for all 5 datasets → 4,753 `.h5` files ✅ |
| 10 | `preprocess/gen_point_npy.py` run for all 5 datasets → BL + DM-Count `.npy` files ✅ |
| 11 | `preprocess/gen_cltr_h5.py` run for all 5 datasets → CLTR `.h5` files ✅ |
| 12 | `preprocess/create_unified_split.py` run → 3327/713/713 train/val/test (4,753 total) ✅ |
| 13 | `TransCrowd/Networks/models.py` — fixed deprecated timm imports ✅ |
| 14 | `TransCrowd/train.py` — `Resize(384,384)`; lazy `pre_data()`; NNI bypass ✅ |
| 15 | `TransCrowd/dataset.py` — small-image guard; lazy `__getitem__` (28GB→1.6GB RAM) ✅ |
| 16 | `CSRNet/image.py` — Python 3 div fix; QNRF path fallback; `import os` ✅ |
| 17 | `preprocess/gen_csrnet_json.py` written and run → 7 JSON list files for CSRNet ✅ |
| 18 | `preprocess/reorganise_bl_dm.py` written and run → BL/DM `train/`+`val/` subdirs ✅ |
| 19 | DM-Count SHA/SHB symlinks: `train_data→train`, `test_data→val` ✅ |
| 20 | P2PNet: SHA/SHB `.txt` GT + `.list` files; `SHHB` class; configurable list names ✅ |
| 21 | CLTR: `image.py` path fix; sha/shb/qnrf/unified npy lists; NNI bypass; `local_rank=0` fix ✅ |
| 22 | STEERER: JSON GT + split txts + `SHHA_our.py`/`SHHB_our.py`/`QNRF_our.py` configs ✅ |
| 23 | `MCNN/train.py` — removed old module-level code (lines 148–287) that caused crash after epoch 200 ✅ |
| 24 | **All 7 trainable models** now have early stopping (patience=50) + uniform VAL log output ✅ |
| 25 | **All checkpoints** centralized under `logs/<model>_<dataset>_ckpts/` ✅ |
| 26 | **`plot_training.py`** created, tested — parses VAL log lines, plots MAE+MSE curves ✅ |

### ⏳ In Progress / Next Steps
1. **CSRNet SHA**: running PID 1293610, ep 0+, training from scratch — patience=50
2. **MCNN SHA**: running PID 1295440, ep 0+, training from scratch — patience=50
3. **VGG16+FC unified**: running PID 1298174, ep 0+, training from scratch — patience=50
4. After above 3 finish: launch **BL SHA**, **DM-Count SHA**
5. After those: launch **P2PNet SHA**, **CLTR SHA**, **ResNet50+FC**, **TransCrowd**
6. **STEERER**: ⚠️ BLOCKED — needs `hrnetv2_w48_imagenet_pretrained.pth`

---

## Preprocessing Status

### Step 1 — .npy Density Maps ✅ DONE

| Dataset | Split | Files |
|---|---|---|
| ShanghaiTech A | train_data/gt_density_map | 300 |
| ShanghaiTech A | test_data/gt_density_map | 182 |
| ShanghaiTech B | train_data/gt_density_map | 400 |
| ShanghaiTech B | test_data/gt_density_map | 316 |
| UCF-QNRF-processed | train/gt_density_map | 1081 |
| UCF-QNRF-processed | val/gt_density_map | 120 |
| UCF-QNRF-processed | test/gt_density_map | 334 |
| Unidata/processed | gt_density_map | 20 |
| mall_dataset | gt_density_map | 2000 |
| **Total** | | **4,753** |

### Step 2 — CSRNet .h5 Density Maps ✅ DONE (4,753 files)

| Dataset | Files |
|---|---|
| ShanghaiTech A (train+test) | 482 |
| ShanghaiTech B (train+test) | 716 |
| UCF-QNRF-processed (train/val/test) | 1,535 |
| Unidata/processed | 20 |
| mall_dataset | 2,000 |
| **Total** | **4,753** |

### Steps 3–5 — ALL DONE ✅

| Step | Output | Files |
|---|---|---|
| 3 — BL point npy `(N,3)` | `<dataset>/bl/*.npy` | 4,753 |
| 3 — DM point npy `(N,2)` | `<dataset>/dm/*.npy` | 4,753 |
| 4 — CLTR h5 | `<dataset>/gt_detr_map/*.h5` | 4,753 |
| 5 — Unified split | `TransCrowd/npydata/unified_{train,val,test}.npy` | 3327/713/713 |

---

## Project Goal

Benchmark **10 crowd counting models** across three methodological families on multiple datasets, as part of a bachelor-level university course project.

---

## Models

| # | Model | Family | Notes |
|---|---|---|---|
| 1 | MCNN | Density map | CVPR 2016 — `MCNN/` cloned from svishwa/crowdcount-mcnn, fully Python 3 |
| 2 | CSRNet | Density map | CVPR 2018 — `CSRNet/` |
| 3 | BL (Bayesian Loss) | Density map | ICCV 2019 — `Bayesian-Loss/` |
| 4 | DM-Count | Density map | NeurIPS 2020 — `DM-Count/` |
| 5 | P2PNet | Point detection | ICCV 2021 — `P2PNet/` |
| 6 | CLTR | Point detection | ECCV 2022 — `CLTR/` |
| 7 | STEERER | Density map | ICCV 2023 — `STEERER/` |
| 8 | TransCrowd | Regression (ViT) | IJCAI 2022 — `TransCrowd/` |
| 9 | VGG16+FC | Regression (CNN) | New — `TransCrowd/Networks/models.py` |
| 10 | ResNet50+FC | Regression (CNN) | New — `TransCrowd/Networks/models.py` |

---

## Datasets

| Dataset | Images | Count range | Annotation format | Phase |
|---|---|---|---|---|
| ShanghaiTech A | 482 (300+182) | 33–3139 | `.mat` → `image_info[0,0][0,0][0]` → `(N,2)` | 1+2 |
| ShanghaiTech B | 716 (400+316) | 9–578 | same as SHA | 1+2 |
| UCF-QNRF | 1535 (1201+334) | 49–12,865 | DM-Count processed → `(N,2)` `.npy` per image | 1+2 |
| UCF-CC-50 | 50 (5-fold CV) | 94–4,543 | `_ann.mat` → `annPoints` → `(N,2)` | 2 only |
| Unidata | 20 | varies | JSON keypoint → `(N,2)` `.npy` | 1 |
| mall | 2000 | 13–53 | `mall_gt.mat` → `frame[i]['loc']` → `(N,2)` | 1 |
| JHU-Crowd++ | 4372 (2272/500/1600) | 0–25,791 | `.txt` `x y w h o b` (x,y = head centers) | **3 (deferred)** |

---

## Experiment Phases

### Phase 1 — Unified merged dataset
Pool SHA + SHB + QNRF + Unidata + mall → 70/15/15 train/val/test, seed=42. JHU excluded: too large, annotation reliability concerns (occluded/low-confidence heads).

### Phase 2 — Individual standard benchmarks
SHA, SHB, QNRF (standard splits), UCF-CC-50 (5-fold CV).

### Phase 3 (future / not scoped)
JHU-Crowd++, YOLO, RF-DETR, RT-DETR.

---

## Data Format Matrix

| Model | GT format needed | Path convention |
|---|---|---|
| MCNN | `(H,W)` `.npy` density map | `<split>/gt_density_map/<stem>.npy` |
| CSRNet | `.h5` key `density` `(H,W)` | `<split>/ground_truth/<stem>.h5` |
| BL | `(N,3)` `.npy` [x,y,nn_dist] | `<split>/bl/<stem>.npy` + symlinked images |
| DM-Count | `(N,2)` `.npy` [x,y] | `<split>/dm/<stem>.npy` + symlinked images |
| P2PNet | `.list` path file + `.mat` GT | list-driven (to be written) |
| CLTR | `.h5` keys `image`+`kpoint` `(H,W)` | `<split>/gt_detr_map/<stem>.h5` |
| STEERER | JSON list path file (NWPU-style) | config-driven (to be written) |
| TransCrowd / VGG16+FC / ResNet50+FC | scalar count | `.npy` path list + sidecar `_counts.npy` |

---

## Files Created / Modified

### Preprocess scripts (`preprocess/`) — all support sha/shb/qnrf/unidata/mall/jhu

| File | Purpose | Status |
|---|---|---|
| `convert_unidata.py` | JSON → `(N,2)` `.npy` | ✅ Run |
| `gen_density_maps.py` | Gaussian `.npy` density maps | ✅ Run (5 datasets, no JHU) |
| `gen_h5_density.py` | CSRNet `.h5` density maps | ✅ Run (5 datasets, 4,753 files) |
| `gen_point_npy.py` | BL `(N,3)` + DM-Count `(N,2)` `.npy` | ✅ Run (5 datasets) |
| `gen_cltr_h5.py` | CLTR `.h5` (image+kpoint) | ✅ Run (5 datasets) |
| `create_unified_split.py` | Merged 70/15/15 split (no JHU) | ✅ Run (3327/713/713) |

**Density map algorithm**: adaptive Gaussian, σ = 0.3 × mean(k-NN dist, k=3), clamped [2,20]px; fallback σ=15 if N≤3.

### TransCrowd modifications

| File | Change |
|---|---|
| `TransCrowd/Networks/models.py` | `VGG16CountNet`, `ResNet50CountNet`, factory functions |
| `TransCrowd/config.py` | `--model_type` choices; **`--patience` (default 50)** |
| `TransCrowd/train.py` | Model dispatch; `unified` dataset; **early stopping + VAL log** |
| `TransCrowd/image.py` | `load_data()` accepts `gt_count=` kwarg → skips `.h5` lookup |

### Early Stopping Design (all models, patience=50)

| Model | Script | Val frequency | Patience counter |
|---|---|---|---|
| CSRNet | `CSRNet/train.py` | every epoch | increments by 1 per epoch |
| MCNN | `MCNN/train.py` | every 2 epochs | increments by 2 per val call |
| VGG16+FC / ResNet50+FC / TransCrowd | `TransCrowd/train.py` | every 5 epochs | increments by 5 |
| Bayesian-Loss | `Bayesian-Loss/utils/regression_trainer.py` | every `--val-epoch` epochs | increments by val-epoch |
| DM-Count | `DM-Count/train_helper.py` | every `--val-epoch` epochs | increments by val-epoch |
| P2PNet | `P2PNet/train.py` | every `--eval_freq` epochs | increments by eval_freq |
| CLTR | `CLTR/train_distributed.py` | every `--test_per_epoch` epochs | increments by test_per_epoch |

**VAL log format** (same for ALL models, printed to stdout → captured in `.log`):
```
VAL epoch=XXX mae=XX.XX mse=XX.XX best_mae=XX.XX
```

### Checkpoint Directories

All checkpoints go under `logs/<model>_<dataset>_ckpts/`. Only `best_model.pth` (or `best_model.h5` for MCNN) is kept; no per-epoch files.

| Model | Checkpoint Dir |
|---|---|
| CSRNet SHA | `logs/csrnet_sha_ckpts/` |
| MCNN SHA | `logs/mcnn_sha_ckpts/` |
| Bayesian-Loss SHA | `logs/bl_sha_ckpts/` |
| DM-Count SHA | `logs/dmcount_sha_ckpts/` |
| P2PNet SHA | `logs/p2pnet_sha_ckpts/` |
| CLTR SHA | `logs/cltr_sha_ckpts/` |
| VGG16+FC Unified | `logs/vgg16_unified_ckpts/` |
| ResNet50+FC Unified | `logs/resnet50_unified_ckpts/` |
| TransCrowd Unified | `logs/transcrowd_unified_ckpts/` |

### MCNN Python 3 port (`MCNN/src/`)

| File | Change |
|---|---|
| `data_loader.py` | Full rewrite: Python 3, reads `.npy`, `//` integer division |
| `network.py` | Removed `Variable`, `volatile=True` → `.detach()` |
| `models.py` | `from .network import Conv2d` |
| `crowd_count.py` | Relative imports |
| `evaluate_model.py` | Relative imports |
| `train.py` | Full rewrite: argparse CLI |
| `test.py` | Full rewrite: argparse CLI |

---

## Full Command Reference

```bash
# ─── SETUP ───────────────────────────────────────────────────────────────────
# All commands from: /ssd1/team_cam_ai/ntthai/crowd_counting
# NEVER use `conda run` — it buffers stdout. Always use direct Python path.
PYTHON=/home/team_cam_ai/miniconda3/envs/ntt_det/bin/python
BASE=/ssd1/team_cam_ai/ntthai/crowd_counting

# ─── CURRENTLY RUNNING (2026-03-10 restart) ──────────────────────────────────
# CSRNet SHA   PID 1293610  ep 0+  from scratch  patience=50
# MCNN SHA     PID 1295440  ep 0+  from scratch  patience=50
# VGG16+FC     PID 1298174  ep 0+  from scratch  patience=50

# ─── MONITOR ────────────────────────────────────────────────────────────────
ps aux | grep -E "[p]ython.*train" | awk '{print $2, $12, $13, $14}'
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
tail -5 $BASE/logs/csrnet_sha.log $BASE/logs/mcnn_sha.log $BASE/logs/vgg16_unified.log
grep "^VAL " $BASE/logs/csrnet_sha.log | tail -3   # structured val lines

# ─── PLOT TRAINING CURVES ───────────────────────────────────────────────────
# Requires at least some VAL log lines to exist in logs/*.log
cd $BASE
$PYTHON plot_training.py --log-dir logs/ --output plots/training_curves.png
# Plots only specific models:
$PYTHON plot_training.py --log-dir logs/ --models csrnet_sha mcnn_sha vgg16_unified --output plots/density_models.png

# ─── RELAUNCH IF KILLED ─────────────────────────────────────────────────────
# CSRNet SHA (from scratch, or with --pre to resume):
nohup bash -c "cd $BASE/CSRNet && $PYTHON -u train.py part_A_train.json part_A_test.json 0 SHA --ckpt-dir ../logs/csrnet_sha_ckpts --patience 50 --epochs 400" > $BASE/logs/csrnet_sha.log 2>&1 &

# CSRNet SHA (resume from best):
nohup bash -c "cd $BASE/CSRNet && $PYTHON -u train.py part_A_train.json part_A_test.json 0 SHA --ckpt-dir ../logs/csrnet_sha_ckpts --pre ../logs/csrnet_sha_ckpts/model_best.pth.tar --patience 50 --epochs 400" > $BASE/logs/csrnet_sha.log 2>&1 &

# MCNN SHA (from scratch):
nohup $PYTHON -u $BASE/MCNN/train.py --dataset shanghaiA --data-dir $BASE/data/ShanghaiTech/part_A --output-dir $BASE/logs/mcnn_sha_ckpts --epochs 2000 --lr 1e-5 --gpu 0 --patience 50 > $BASE/logs/mcnn_sha.log 2>&1 &

# MCNN SHA (resume from best):
nohup $PYTHON -u $BASE/MCNN/train.py --dataset shanghaiA --data-dir $BASE/data/ShanghaiTech/part_A --output-dir $BASE/logs/mcnn_sha_ckpts --epochs 2000 --lr 1e-5 --gpu 0 --patience 50 --resume $BASE/logs/mcnn_sha_ckpts/best_model.h5 > $BASE/logs/mcnn_sha.log 2>&1 &

# VGG16+FC unified (from scratch):
nohup bash -c "cd $BASE/TransCrowd && $PYTHON -u train.py --dataset unified --model_type vgg16 --save_path ../logs/vgg16_unified_ckpts --gpu_id 0 --lr 1e-5 --epochs 500 --batch_size 8 --print_freq 50 --patience 50" > $BASE/logs/vgg16_unified.log 2>&1 &

# VGG16+FC unified (resume from best):
nohup bash -c "cd $BASE/TransCrowd && $PYTHON -u train.py --dataset unified --model_type vgg16 --save_path ../logs/vgg16_unified_ckpts --gpu_id 0 --lr 1e-5 --epochs 500 --batch_size 8 --print_freq 50 --patience 50 --pre ../logs/vgg16_unified_ckpts/model_best.pth" > $BASE/logs/vgg16_unified.log 2>&1 &

# ─── QUEUED — LAUNCH WHEN ABOVE 3 FINISH ────────────────────────────────────

# BL SHA (VGG19, ~3 GB VRAM, batch=5, crop=128, val starts at ep 200)
nohup bash -c "cd $BASE/Bayesian-Loss && $PYTHON -u train.py --data-dir ../data/ShanghaiTech/part_A/bl --save-dir ../logs/bl_sha_ckpts --device 0 --batch-size 5 --crop-size 128 --max-epoch 2000 --val-start 200 --val-epoch 5 --patience 50 --num-workers 8" > $BASE/logs/bl_sha.log 2>&1 &

# DM-Count SHA (VGG19, ~3 GB VRAM, batch=8)
nohup bash -c "cd $BASE/DM-Count && $PYTHON -u train.py --dataset sha --data-dir ../data/ShanghaiTech/part_A --save-dir ../logs/dmcount_sha_ckpts --device 0 --batch-size 8 --max-epoch 2000 --val-start 50 --val-epoch 5 --patience 50 --num-workers 4" > $BASE/logs/dmcount_sha.log 2>&1 &

# P2PNet SHA (VGG16_bn + transformer, ~6 GB)
# First create output dir:
mkdir -p $BASE/logs/p2pnet_sha_ckpts
nohup bash -c "cd $BASE/P2PNet && $PYTHON -u train.py --dataset_file SHHA --data_root ../data/ShanghaiTech/part_A --epochs 3500 --lr_drop 3500 --output_dir ../logs/p2pnet_sha_ckpts --checkpoints_dir ../logs/p2pnet_sha_ckpts --patience 50 --num_workers 4" > $BASE/logs/p2pnet_sha.log 2>&1 &

# CLTR SHA (ResNet50+CDETR, ~6-8 GB) — needs --save flag to save checkpoints
nohup bash -c "cd $BASE/CLTR && $PYTHON -u train_distributed.py --dataset sha --gpu_id 0 --epochs 2000 --lr 1e-4 --crop_size 256 --batch_size 4 --save_path ../logs/cltr_sha_ckpts --patience 50 --save" > $BASE/logs/cltr_sha.log 2>&1 &

# ResNet50+FC unified (after VGG16 finishes):
nohup bash -c "cd $BASE/TransCrowd && $PYTHON -u train.py --dataset unified --model_type resnet50 --save_path ../logs/resnet50_unified_ckpts --gpu_id 0 --lr 1e-4 --epochs 500 --batch_size 16 --print_freq 50 --patience 50" > $BASE/logs/resnet50_unified.log 2>&1 &

# TransCrowd token unified (after above finishes):
nohup bash -c "cd $BASE/TransCrowd && $PYTHON -u train.py --dataset unified --model_type token --save_path ../logs/transcrowd_unified_ckpts --gpu_id 0 --lr 1e-5 --epochs 500 --batch_size 8 --print_freq 50 --patience 50" > $BASE/logs/transcrowd_unified.log 2>&1 &

# STEERER — ⚠️ BLOCKED until hrnetv2_w48_imagenet_pretrained.pth is provided
# Place at: $BASE/PretrainedModels/hrnetv2_w48_imagenet_pretrained.pth
# Then: nohup bash -c "cd $BASE/STEERER && $PYTHON tools/train_cc.py configs/SHHA_our.py" > $BASE/logs/steerer_sha.log 2>&1 &
```

---

## Model Audit Status

| Model | Audit | Training Status |
|---|---|---|
| CSRNet | ✅ | ▶ RUNNING PID 1293610 — SHA from scratch, patience=50 |
| MCNN | ✅ | ▶ RUNNING PID 1295440 — SHA from scratch, patience=50 |
| VGG16+FC | ✅ | ▶ RUNNING PID 1298174 — Unified from scratch, patience=50 |
| Bayesian-Loss | ✅ | ⏳ QUEUED |
| DM-Count | ✅ | ⏳ QUEUED |
| P2PNet | ✅ | ⏳ QUEUED |
| CLTR | ✅ | ⏳ QUEUED |
| ResNet50+FC | ✅ | ⏳ QUEUED (after VGG16 finishes) |
| TransCrowd | ✅ | ⏳ QUEUED (after ResNet50 finishes) |
| STEERER | ⚠️ | ❌ BLOCKED — needs `hrnetv2_w48_imagenet_pretrained.pth` |

## STEERER Blocker — Requires User Action

STEERER needs the HRNet-W48 pretrained backbone. Config points to:
```
/ssd1/team_cam_ai/ntthai/crowd_counting/PretrainedModels/hrnetv2_w48_imagenet_pretrained.pth
```
**Action**: Download `hrnetv2_w48_imagenet_pretrained.pth` from the HRNet-Image-Classification GitHub releases and place it at the path above.

## BL & DM-Count Directory Structure

BL trainer expects `<data_dir>/train/` and `<data_dir>/val/` with `*.jpg` + `replace('.jpg','.npy')` sidecar:

| Dataset | BL/DM train | BL/DM val |
|---|---|---|
| ShanghaiTech A | 300 jpg+npy | 182 jpg+npy |
| ShanghaiTech B | 400 jpg+npy | 316 jpg+npy |
| UCF-QNRF | 1081 jpg+npy | 120+334 jpg+npy |
| mall | 1400 jpg+npy | 600 jpg+npy |
| Unidata | 14 jpg+npy | 6 jpg+npy |

DM-Count SHA/SHB symlinks: `train_data→train`, `test_data→val` (needed by `Crowd_sh` class).
