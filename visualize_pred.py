"""
Visualize YOLO detections and GT counts for best/worst predictions on SHA/SHB.

Usage:
  PYTHON=/home/team_cam_ai/miniconda3/envs/ntt_det/bin/python
  BASE=/ssd1/team_cam_ai/ntthai/crowd_counting
  $PYTHON -u $BASE/visualize_pred.py \
    --weights $BASE/runs/head_detection/train/weights/best.pt \
    --csv-sha $BASE/runs/head_detection/eval_csv/yolo_sha.csv \
    --csv-shb $BASE/runs/head_detection/eval_csv/yolo_shb.csv \
    --sha-root $BASE/data/ShanghaiTech/part_A \
    --shb-root $BASE/data/ShanghaiTech/part_B \
    --out-dir $BASE/runs/head_detection/vis
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import scipy.io
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize YOLO best/worst predictions")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--csv-sha", type=str, required=True)
    parser.add_argument("--csv-shb", type=str, required=True)
    parser.add_argument("--sha-root", type=str, required=True)
    parser.add_argument("--shb-root", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--num", type=int, default=5, help="Number of best/worst to visualize per split")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", type=str, default="0")
    return parser.parse_args()

def load_csv(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["abs_error"] = float(row["abs_error"])
            rows.append(row)
    return rows

def get_samples(rows, num=1):
    best = sorted(rows, key=lambda r: r["abs_error"])[:num]
    worst = sorted(rows, key=lambda r: r["abs_error"], reverse=True)[:num]
    return best, worst

def get_image_path(img_name, root):
    return Path(root) / "test_data" / "images" / img_name

def get_gt_points(gt_path):
    mat = scipy.io.loadmat(str(gt_path))
    pts = mat["image_info"][0, 0][0, 0][0]
    return np.array(pts)

def draw_boxes_and_points(img_path, boxes, gt_pts, pred_count, gt_count, error, out_path):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    # Draw YOLO boxes
    for box in boxes:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    # Draw GT points
    for pt in gt_pts:
        x, y = pt
        draw.ellipse([x-3, y-3, x+3, y+3], fill="lime", outline="black")
    # Add text
    text = f"Pred: {pred_count} | GT: {gt_count} | Error: {error}"
    draw.text((10, 10), text, fill="yellow")
    img.save(out_path)

def visualize(rows, root, model, out_dir, num=1):
    best, worst = get_samples(rows, num)
    for tag, samples in [("best", best), ("worst", worst)]:
        for row in samples:
            img_name = row["image"]
            img_path = get_image_path(img_name, root)
            gt_path = Path(root) / "test_data" / "ground-truth" / f"GT_{Path(img_name).stem}.mat"
            gt_pts = get_gt_points(gt_path)
            result = model.predict(
                source=str(img_path),
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
                classes=[0],
            )[0]
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
            out_path = Path(out_dir) / f"{tag}_{img_name}"
            draw_boxes_and_points(img_path, boxes, gt_pts, row["pred_count"], row["gt_count"], row["error"], out_path)
            print(f"Saved {tag} {img_name} to {out_path}")

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    model = YOLO(args.weights)
    # SHA
    sha_rows = load_csv(args.csv_sha)
    visualize(sha_rows, args.sha_root, model, args.out_dir, num=args.num)
    # SHB
    shb_rows = load_csv(args.csv_shb)
    visualize(shb_rows, args.shb_root, model, args.out_dir, num=args.num)