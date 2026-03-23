# Experiments Log

> Python: `/home/team_cam_ai/miniconda3/envs/ntt_det/bin/python` (3.13.5)
> PyTorch 2.7.1 · torchvision 0.22.1+cu128 · CUDA 12.8 · single GPU
> Working dir: `/ssd1/team_cam_ai/ntthai/crowd_counting`
> **NEVER** use `conda run` — it buffers stdout and kills nohup logging.

---

## Strategy

Train **8 crowd-counting models** on **ShanghaiTech A (SHA)** and **ShanghaiTech B (SHB)** separately, then evaluate a **YOLO11m head-detector counting baseline** on the same test sets. Additionally: backbone swap experiments on the global regressor, and real-world video inference on 12 internet videos.

**Models dropped:** CLTR, STEERER, TransCrowd (transformer-based; unstable training, modest results).

| Dataset | Images | Count range | Density |
|---|---|---|---|
| ShanghaiTech A | 482 (300 train + 182 test) | 33–3,139 | Dense urban |
| ShanghaiTech B | 716 (400 train + 316 test) | 9–578 | Sparse suburban |

---

## Checkpoint File Locations

| Model | SHA checkpoint | SHB checkpoint |
|---|---|---|
| CSRNet | `logs/csrnet_sha_ckpts/model_best.pth.tar` | `logs/csrnet_shb_ckpts/model_best.pth.tar` |
| P2PNet | `logs/p2pnet_sha_ckpts/best_mae.pth` | `logs/p2pnet_shb_ckpts/best_mae.pth` |
| EfficientNet-B0+FC | `logs/b0_sha_ckpts/model_best.pth` | `logs/b0_shb_ckpts/model_best.pth` |
| YOLO11m | `runs/head_detection/train/weights/best.pt` | (same — no SHA/SHB distinction) |

---

## Hyperparameters

| Model | LR | Batch | Epochs | Optimizer | Notes |
|---|---|---|---|---|---|
| CSRNet | 1e-6 | 1 | 400 | SGD | Hardcoded in script — not a CLI flag |
| P2PNet | 1e-4 (backbone 1e-5) | 8 | 3500 | AdamW | lr_drop=3500 |
| EfficientNet-B0+FC | 1e-4 | 8 | 1000 | Adam | input 448×448 |

---

## SHA Training Commands

```bash
PYTHON=/home/team_cam_ai/miniconda3/envs/ntt_det/bin/python
BASE=/ssd1/team_cam_ai/ntthai/crowd_counting

# ─── CSRNet SHA ───────────────────────────────────────────────────────────────
nohup $PYTHON -u $BASE/CSRNet/train.py \
  $BASE/CSRNet/part_A_train.json $BASE/CSRNet/part_A_test.json 0 sha \
  --epochs 400 --patience 50 \
  --ckpt-dir $BASE/logs/csrnet_sha_ckpts \
  > $BASE/logs/csrnet_sha.log 2>&1 &

# ─── P2PNet SHA ───────────────────────────────────────────────────────────────
mkdir -p $BASE/logs/p2pnet_sha_ckpts
nohup $PYTHON -u $BASE/P2PNet/train.py \
  --dataset_file SHHA \
  --data_root $BASE/data/ShanghaiTech/part_A \
  --output_dir $BASE/logs/p2pnet_sha_ckpts \
  --checkpoints_dir $BASE/logs/p2pnet_sha_ckpts \
  --epochs 3500 --lr 1e-4 --patience 200 --backbone vgg16_bn --gpu_id 0 \
  > $BASE/logs/p2pnet_sha.log 2>&1 &

# ─── EfficientNet-B0+FC SHA ───────────────────────────────────────────────────
nohup $PYTHON -u $BASE/train_regressor.py \
  --dataset ShanghaiA --data-dir $BASE/data/ShanghaiTech/part_A \
  --save-dir $BASE/logs/b0_sha_ckpts --model-type efficientnet_b0 \
  --epochs 1000 --lr 1e-4 --batch-size 8 --patience 50 \
  > $BASE/logs/b0_sha.log 2>&1 &
```

## SHB Training Commands

```bash
# ─── CSRNet SHB ───────────────────────────────────────────────────────────────
nohup $PYTHON -u $BASE/CSRNet/train.py \
  $BASE/CSRNet/part_B_train.json $BASE/CSRNet/part_B_test.json 0 shb \
  --epochs 400 --patience 50 \
  --ckpt-dir $BASE/logs/csrnet_shb_ckpts \
  > $BASE/logs/csrnet_shb.log 2>&1 &

# ─── P2PNet SHB ───────────────────────────────────────────────────────────────
mkdir -p $BASE/logs/p2pnet_shb_ckpts
nohup $PYTHON -u $BASE/P2PNet/train.py \
  --dataset_file SHHB \
  --data_root $BASE/data/ShanghaiTech/part_B \
  --output_dir $BASE/logs/p2pnet_shb_ckpts \
  --checkpoints_dir $BASE/logs/p2pnet_shb_ckpts \
  --epochs 3500 --lr 1e-4 --patience 200 --backbone vgg16_bn --gpu_id 0 \
  > $BASE/logs/p2pnet_shb.log 2>&1 &

# ─── EfficientNet-B0+FC SHB ───────────────────────────────────────────────────
nohup $PYTHON -u $BASE/train_regressor.py \
  --dataset ShanghaiB --data-dir $BASE/data/ShanghaiTech/part_B \
  --save-dir $BASE/logs/b0_shb_ckpts --model-type efficientnet_b0 \
  --epochs 1000 --lr 1e-4 --batch-size 8 --patience 50 \
  > $BASE/logs/b0_shb.log 2>&1 &
```

---

## Video Inference Commands

```bash
mkdir -p results

# Run all models on all videos, both SHA and SHB checkpoints
for video in videos/*.mp4; do
  name=$(basename "$video" .mp4)
  python video_inference.py --model all --trained_on sha \
    --video "$video" --output_csv "results/${name}_sha.csv" --every_n 5
  python video_inference.py --model all --trained_on shb \
    --video "$video" --output_csv "results/${name}_shb.csv" --every_n 5
done

# Extract summary from all CSVs
for f in results/*.csv; do
  echo "=== $f ==="
  python -c "
import pandas as pd
df = pd.read_csv('$f')
cols = [c for c in df.columns if c not in ['frame','timestamp_s']]
for c in cols:
    print(f'  {c}: mean={df[c].mean():.1f} std={df[c].std():.1f}')
"
done
```


## Data Layout Reference

| Model | SHA data path | SHB data path |
|---|---|---|
| CSRNet | `CSRNet/part_A_train.json` + `.h5` density maps | `CSRNet/part_B_train.json` |
| P2PNet | `P2PNet/crowd_datasets/SHHA/SHHA.list` | `SHHB/SHHB.list` |
| Regressor | `data/ShanghaiTech/part_A/` (raw) | `data/ShanghaiTech/part_B/` (raw) |

---

## Project Narrative

### Background and Motivation

The project benchmarks crowd-counting methods spanning four architectural families. Eight models were fully trained and evaluated, plus YOLO as a detection-count baseline. Three transformer-based models (CLTR, STEERER, TransCrowd) were attempted but dropped due to unstable training.

### Backbone Swap Experiment

Backbone swap was evaluated on the global regression model to answer: *which backbone properties matter for crowd counting?* Six backbones were tested. Key findings:

- **EfficientNet family outperforms VGG16** despite being 10–25x smaller. EfficientNet-B0 (5.3M params) achieved MAE 91.67 vs VGG16's 113.51.
- **ImageNet accuracy is not a reliable predictor** — EfficientNet-B0 (77.7% acc) beats ResNet50 (80.9% acc) and ConvNeXt-Tiny (82.5% acc). Feature efficiency matters more than raw classification accuracy.
- **ResNet50 underperforms VGG16** — skip connections and depth do not help global regression on this task.
- **ConvNeXt-Tiny failed to converge** with lr=1e-4 — likely requires different lr schedule (LayerNorm + GELU architecture).
- **CSRNet backbone swap failed** — lr is hardcoded at 1e-6 for VGG16; non-VGG backbones need higher lr but no CLI flag exists. All non-VGG variants plateaued at MAE ~390.
- **P2PNet FPN is architecture-sensitive** — only ResNet50 converged (MAE 114 vs VGG16-BN's 58); MobileNetV3 and EfficientNet produced degenerate results due to FPN channel mismatch.

### Video Inference and Domain Gap

Twelve internet videos were collected (7 dense mall, 5 sparse retail/CCTV) and evaluated with SHA-trained and SHB-trained checkpoints for each model. Three videos were manually counted for ground truth validation.

**Key findings:**
- P2PNet SHA was closest on medium-density scenes (~150 people).
- CSRNet SHA was closest on high-density scenes (~500 people).
- P2PNet SHB collapsed on sparse scenes (predicted 2–6 people when 10+ were present).
- Regressor (EfficientNet-B0) showed high temporal instability (high std) due to sensitivity to visual texture — shelves, bottles, and decorative lighting caused large frame-to-frame variance.
- YOLO consistently failed on crowds above 20 people due to occlusion and small object scale.
- Domain gap was clearest in the regressor: SHA-trained predicted 714 on a dense mall (GT ~500) while SHB-trained predicted only 140 on the same scene.
