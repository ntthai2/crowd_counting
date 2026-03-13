#!/usr/bin/env bash
set -euo pipefail

BASE=/ssd1/team_cam_ai/ntthai/crowd_counting
PYTHON=/home/team_cam_ai/miniconda3/envs/ntt_det/bin/python
APGCC_DIR="$BASE/APGCC/apgcc"

mkdir -p "$BASE/logs/apgcc_sha_ckpts" "$BASE/logs/apgcc_shb_ckpts"

cd "$APGCC_DIR"

# SHA
nohup "$PYTHON" -u main.py \
  -c ./configs/SHHA_IFI.yml \
  DATASETS.DATA_ROOT "$BASE/data/ShanghaiTech/part_A" \
  DATASETS.DATASET SHHA \
  OUTPUT_DIR "$BASE/logs/apgcc_sha_ckpts" \
  GPU_ID 0 \
  > "$BASE/logs/apgcc_sha.log" 2>&1 &

# SHB
nohup "$PYTHON" -u main.py \
  -c ./configs/SHHB_IFI.yml \
  DATASETS.DATA_ROOT "$BASE/data/ShanghaiTech/part_B" \
  DATASETS.DATASET SHHB \
  OUTPUT_DIR "$BASE/logs/apgcc_shb_ckpts" \
  GPU_ID 0 \
  > "$BASE/logs/apgcc_shb.log" 2>&1 &

printf "Started APGCC SHA + SHB training in background.\n"
printf "Logs:\n  %s\n  %s\n" "$BASE/logs/apgcc_sha.log" "$BASE/logs/apgcc_shb.log"
