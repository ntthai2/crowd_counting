"""
convert_unidata.py
------------------
Convert Unidata JSON keypoint annotations to per-image numpy point arrays (.npy)
compatible with the format used by ShanghaiTech/UCF-QNRF loaders.

Each output file `<stem>.npy` contains a float32 array of shape (N, 2)
where each row is [x, y] of one annotated head.

Usage
-----
python preprocess/convert_unidata.py \
    --data-dir data/Unidata \
    --out-dir  data/Unidata/processed
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from pathlib import Path


def load_points_from_json(json_path: str) -> np.ndarray:
    """Return (N, 2) float32 array of [x, y] head keypoints from a Unidata JSON."""
    with open(json_path, "r") as f:
        data = json.load(f)

    points = []
    for ann in data.get("annotations", []):
        kp = ann.get("keypoint", {})
        x = kp.get("x")
        y = kp.get("y")
        if x is not None and y is not None:
            points.append([float(x), float(y)])

    return np.array(points, dtype=np.float32).reshape(-1, 2)


def main():
    parser = argparse.ArgumentParser(description="Convert Unidata JSON annotations to .npy point arrays")
    parser.add_argument("--data-dir", required=True,
                        help="Root of Unidata (contains crowds_counting.csv and images/, labels/)")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory; mirrors the images/ sub-structure")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)

    csv_path = data_dir / "crowds_counting.csv"
    df = pd.read_csv(csv_path)

    processed, skipped = 0, 0
    for _, row in df.iterrows():
        img_rel  = row["image"]   # e.g. images/0-1000/0.jpg
        lbl_rel  = row["label"]   # e.g. labels/0-1000/0.json

        json_path = data_dir / lbl_rel
        img_path  = data_dir / img_rel

        if not json_path.exists():
            print(f"[WARN] Missing annotation: {json_path}")
            skipped += 1
            continue

        # Preserve sub-directory structure under out_dir/labels/
        stem      = Path(img_rel).stem          # "0"
        bucket    = Path(img_rel).parent.name   # "0-1000"
        out_subdir = out_dir / "labels" / bucket
        out_subdir.mkdir(parents=True, exist_ok=True)

        points = load_points_from_json(str(json_path))
        out_npy = out_subdir / f"{stem}.npy"
        np.save(str(out_npy), points)

        # Also create a symlink / copy path list item for images
        img_out_dir = out_dir / "images" / bucket
        img_out_dir.mkdir(parents=True, exist_ok=True)
        img_link = img_out_dir / Path(img_rel).name
        if not img_link.exists():
            try:
                img_link.symlink_to(img_path.resolve())
            except OSError:
                pass  # network fs or Windows: skip symlink

        processed += 1

    print(f"Done. Processed: {processed}, Skipped: {skipped}")
    print(f"Point arrays written to: {out_dir / 'labels'}")


if __name__ == "__main__":
    main()
