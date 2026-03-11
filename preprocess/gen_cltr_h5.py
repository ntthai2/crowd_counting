"""
gen_cltr_h5.py
--------------
Generate CLTR-compatible .h5 files.

CLTR's image.py expects:
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'gt_detr_map')

Each .h5 file must contain two datasets:
    'image'   — uint8  (H, W, 3)   the RGB image itself
    'kpoint'  — float32 (H, W)     binary point map  (1.0 at each head location)

This combined format is what CLTR's make_npydata-style preprocessing creates.
The kpoint map is NOT a Gaussian density map — it is a discrete 0/1 map with
a single 1.0 at the rounded (x, y) coordinate of each annotated head.

Usage
-----
python preprocess/gen_cltr_h5.py --dataset shanghaiA \
    --data-dir data/ShanghaiTech/part_A

python preprocess/gen_cltr_h5.py --dataset shanghaiB \
    --data-dir data/ShanghaiTech/part_B
"""

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
import scipy.io
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Point → kpoint map
# ---------------------------------------------------------------------------

def _make_kpoint_map(points: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
    """Return (H, W) float32 map with 1.0 at each annotated head location."""
    kmap = np.zeros((img_h, img_w), dtype=np.float32)
    if len(points) == 0:
        return kmap
    for x, y in points:
        xi = int(round(x))
        yi = int(round(y))
        xi = max(0, min(img_w - 1, xi))
        yi = max(0, min(img_h - 1, yi))
        kmap[yi, xi] = 1.0
    return kmap


# ---------------------------------------------------------------------------
# Dataset iterators  →  (img_path, points (N,2), out_dir)
# ---------------------------------------------------------------------------

def _iter_shanghaitech(data_dir: Path, split: str):
    img_dir = data_dir / split / "images"
    gt_dir  = data_dir / split / "ground-truth"
    out_dir = data_dir / split / "gt_detr_map"
    for img_file in sorted(img_dir.glob("*.jpg")):
        gt_file = gt_dir / f"GT_{img_file.stem}.mat"
        if not gt_file.exists():
            continue
        mat = scipy.io.loadmat(str(gt_file))
        pts = mat["image_info"][0, 0][0, 0][0].astype(np.float32)
        yield img_file, pts, out_dir


# ---------------------------------------------------------------------------
# Core writer
# ---------------------------------------------------------------------------

def process(iterator, desc: str):
    for img_path, points, out_dir in tqdm(iterator, desc=desc):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        h, w   = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if len(points) > 0:
            mask   = (points[:, 0] >= 0) & (points[:, 0] < w) & \
                     (points[:, 1] >= 0) & (points[:, 1] < h)
            points = points[mask]

        kmap = _make_kpoint_map(points, h, w)

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (img_path.stem + ".h5")

        with h5py.File(str(out_path), "w") as f:
            f.create_dataset("image",  data=img_rgb,  compression="gzip")
            f.create_dataset("kpoint", data=kmap,     compression="gzip")


def main():
    parser = argparse.ArgumentParser(description="Generate CLTR-format .h5 files")
    parser.add_argument("--dataset", required=True,
                        choices=["shanghaiA", "shanghaiB"])
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    for split in ("train_data", "test_data"):
        process(_iter_shanghaitech(data_dir, split), f"{args.dataset}/{split}")

    print("Done.")


if __name__ == "__main__":
    main()
