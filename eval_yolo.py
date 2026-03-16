"""
Evaluate a YOLO head detector as a crowd-counting model on ShanghaiTech.

For each test image, predicted count = number of detected boxes.
Ground-truth count is read from ShanghaiTech .mat point annotations.

Usage:
  PYTHON=/home/team_cam_ai/miniconda3/envs/ntt_det/bin/python
  BASE=/ssd1/team_cam_ai/ntthai/crowd_counting
    $PYTHON -u $BASE/eval_yolo.py \
        --weights $BASE/runs/head_detection/train/weights/best.pt \
        --dataset both --device 0 \
        --conf 0.25 --imgsz 1280 \
        --save-csv-dir $BASE/runs/head_detection/eval_csv
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import scipy.io
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLO detection-count MAE/RMSE on SHA/SHB")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(ROOT / "runs" / "head_detection" / "train" / "weights" / "best.pt"),
        help="Path to YOLO weights (.pt)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="both",
        choices=["sha", "shb", "both"],
        help="Dataset to evaluate",
    )
    parser.add_argument(
        "--sha-root",
        type=str,
        default=str(ROOT / "data" / "ShanghaiTech" / "part_A"),
        help="Path to ShanghaiTech Part A root",
    )
    parser.add_argument(
        "--shb-root",
        type=str,
        default=str(ROOT / "data" / "ShanghaiTech" / "part_B"),
        help="Path to ShanghaiTech Part B root",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--max-det", type=int, default=5000, help="Max detections per image")
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device for inference (e.g. 0, 0,1, cpu)",
    )
    parser.add_argument(
        "--save-csv-dir",
        type=str,
        default=None,
        help="Optional output directory for per-image prediction CSV files",
    )
    return parser.parse_args()


def load_shanghaitech_count(mat_path: Path) -> int:
    """Read the standard ShanghaiTech point annotation format from .mat."""
    mat = scipy.io.loadmat(str(mat_path))
    if "image_info" not in mat:
        raise KeyError(f"Missing 'image_info' in {mat_path}")

    try:
        pts = mat["image_info"][0, 0][0, 0][0]
        return int(len(pts))
    except Exception as exc:
        raise ValueError(f"Could not parse ShanghaiTech annotation: {mat_path}") from exc


def list_test_samples(dataset_root: Path):
    img_dir = dataset_root / "test_data" / "images"
    gt_dir = dataset_root / "test_data" / "ground-truth"

    if not img_dir.exists():
        raise FileNotFoundError(f"Missing images directory: {img_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"Missing ground-truth directory: {gt_dir}")

    samples = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        gt_path = gt_dir / f"GT_{img_path.stem}.mat"
        if not gt_path.exists():
            continue
        samples.append((img_path, gt_path))

    if not samples:
        raise RuntimeError(f"No test samples found under: {dataset_root}")
    return samples


def count_detections(model: YOLO, image_path: Path, args) -> int:
    result = model.predict(
        source=str(image_path),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        max_det=args.max_det,
        device=args.device,
        verbose=False,
        classes=[0],
    )[0]

    if result.boxes is None:
        return 0
    return int(result.boxes.xyxy.shape[0])


def evaluate_dataset(model: YOLO, name: str, dataset_root: Path, args):
    samples = list_test_samples(dataset_root)
    mae_sum = 0.0
    mse_sum = 0.0
    rows = []

    for idx, (img_path, gt_path) in enumerate(samples, start=1):
        gt = load_shanghaitech_count(gt_path)
        pred = count_detections(model, img_path, args)
        err = pred - gt

        mae_sum += abs(err)
        mse_sum += err * err
        rows.append((img_path.name, gt, pred, err, abs(err), err * err))

        if idx % 50 == 0 or idx == len(samples):
            print(f"[{name}] processed {idx}/{len(samples)}", flush=True)

    n = len(samples)
    mae = mae_sum / n
    mse = math.sqrt(mse_sum / n)

    if args.save_csv_dir:
        out_dir = Path(args.save_csv_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"yolo_{name.lower()}.csv"
        with out_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "gt_count", "pred_count", "error", "abs_error", "sq_error"])
            writer.writerows(rows)
        print(f"[{name}] saved per-image CSV: {out_csv}")

    print(f"EVAL dataset={name} images={n} mae={mae:.2f} mse={mse:.2f}", flush=True)
    return mae, mse, n


def main():
    args = parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    print(f"[INFO] Loading YOLO weights: {weights}")
    model = YOLO(str(weights))

    targets = []
    if args.dataset in {"sha", "both"}:
        targets.append(("SHA", Path(args.sha_root)))
    if args.dataset in {"shb", "both"}:
        targets.append(("SHB", Path(args.shb_root)))

    summary = {}
    for name, root in targets:
        mae, mse, n = evaluate_dataset(model, name, root, args)
        summary[name] = (mae, mse, n)

    print("\n=== Summary ===")
    for name in ["SHA", "SHB"]:
        if name in summary:
            mae, mse, n = summary[name]
            print(f"{name}: images={n}, MAE={mae:.2f}, MSE={mse:.2f}")


if __name__ == "__main__":
    main()
