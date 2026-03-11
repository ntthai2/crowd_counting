"""
gen_h5_density.py
-----------------
Generate CSRNet-compatible .h5 density map files.

CSRNet's image.py expects:
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
    target = np.asarray(gt_file['density'])   # (H, W) density map

The density map is then downsampled by 8x inside the network, so we save at
full resolution here and let CSRNet handle the downsampling.

Usage
-----
python preprocess/gen_h5_density.py --dataset shanghaiA \
    --data-dir data/ShanghaiTech/part_A

python preprocess/gen_h5_density.py --dataset shanghaiB \
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
# Density map helper (same adaptive Gaussian as gen_density_maps.py)
# ---------------------------------------------------------------------------

def _adaptive_gaussian_density(points: np.ndarray, img_h: int, img_w: int,
                                k: int = 3, sigma_min: float = 2.0,
                                sigma_max: float = 20.0,
                                fallback_sigma: float = 15.0) -> np.ndarray:
    density = np.zeros((img_h, img_w), dtype=np.float32)

    if len(points) == 0:
        return density

    points = points.astype(np.float32)

    if len(points) > k:
        diff = points[:, None, :] - points[None, :, :]
        sq_dist = (diff ** 2).sum(axis=2)
        np.fill_diagonal(sq_dist, np.inf)
        sorted_dist = np.sort(sq_dist, axis=1)
        sigmas = np.sqrt(sorted_dist[:, :k].mean(axis=1)) * 0.3
        sigmas = np.clip(sigmas, sigma_min, sigma_max)
    else:
        sigmas = np.full(len(points), fallback_sigma, dtype=np.float32)

    for (x, y), sigma in zip(points, sigmas):
        xi, yi = int(round(x)), int(round(y))
        xi = max(0, min(img_w - 1, xi))
        yi = max(0, min(img_h - 1, yi))

        r = int(3 * sigma) + 1
        x0, x1 = max(0, xi - r), min(img_w, xi + r + 1)
        y0, y1 = max(0, yi - r), min(img_h, yi + r + 1)

        xs = np.arange(x0, x1, dtype=np.float32) - xi
        ys = np.arange(y0, y1, dtype=np.float32) - yi
        gx = np.exp(-xs ** 2 / (2 * sigma ** 2))
        gy = np.exp(-ys ** 2 / (2 * sigma ** 2))
        patch = np.outer(gy, gx)
        patch_sum = patch.sum()
        if patch_sum > 0:
            patch /= patch_sum
        density[y0:y1, x0:x1] += patch

    return density


# ---------------------------------------------------------------------------
# Dataset iterators  (img_path, points)
# ---------------------------------------------------------------------------

def _iter_shanghaitech(data_dir: Path, split: str):
    img_dir = data_dir / split / "images"
    gt_dir  = data_dir / split / "ground-truth"
    for img_file in sorted(img_dir.glob("*.jpg")):
        gt_file = gt_dir / f"GT_{img_file.stem}.mat"
        if not gt_file.exists():
            continue
        mat = scipy.io.loadmat(str(gt_file))
        pts = mat["image_info"][0, 0][0, 0][0].astype(np.float32)
        yield img_file, pts, data_dir / split / "ground_truth"


# ---------------------------------------------------------------------------
# Core writer
# ---------------------------------------------------------------------------

def process(iterator, desc: str):
    for img_path, points, out_dir in tqdm(iterator, desc=desc):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        if len(points) > 0:
            mask = (points[:, 0] >= 0) & (points[:, 0] < w) & \
                   (points[:, 1] >= 0) & (points[:, 1] < h)
            points = points[mask]

        density = _adaptive_gaussian_density(points, h, w)

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (img_path.stem + ".h5")

        with h5py.File(str(out_path), "w") as f:
            f.create_dataset("density", data=density, compression="gzip")


def main():
    parser = argparse.ArgumentParser(description="Generate CSRNet-format .h5 density maps")
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
