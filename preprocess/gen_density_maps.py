"""
gen_density_maps.py
-------------------
Generate per-image adaptive Gaussian density maps from head point annotations.

Adaptive sigma per head = mean distance to the k nearest other heads (k=3).
For sparse images (fewer than k+1 heads) a fixed fallback sigma is used.

The density map D satisfies: sum(D) ≈ number of annotated heads.

Outputs: one `.npy` float32 density map per image, placed in a
`gt_density_map/` directory alongside the `images/` directory of each split.

Supported --dataset values:
  shanghaiA, shanghaiB

Usage
-----
python preprocess/gen_density_maps.py --dataset shanghaiA \
    --data-dir data/ShanghaiTech/part_A

python preprocess/gen_density_maps.py --dataset shanghaiB \
    --data-dir data/ShanghaiTech/part_B
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import scipy.io
import scipy.ndimage
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Density map generation helpers
# ---------------------------------------------------------------------------

def _adaptive_gaussian_density(points: np.ndarray, img_h: int, img_w: int,
                                k: int = 3, sigma_min: float = 2.0,
                                sigma_max: float = 20.0,
                                fallback_sigma: float = 15.0) -> np.ndarray:
    """
    Generate adaptive Gaussian density map.

    Parameters
    ----------
    points : (N, 2) float array  [x, y] coordinates
    img_h, img_w : output map dimensions (same as image)
    k : number of nearest neighbours for sigma estimation
    sigma_min, sigma_max : clamp adaptive sigma to this range
    fallback_sigma : used when N <= k (sparse crowd)
    """
    density = np.zeros((img_h, img_w), dtype=np.float32)

    if len(points) == 0:
        return density

    points = points.astype(np.float32)

    # Compute per-head adaptive sigma via k-NN distances
    if len(points) > k:
        # Pairwise squared distances
        diff = points[:, None, :] - points[None, :, :]          # (N, N, 2)
        sq_dist = (diff ** 2).sum(axis=2)                        # (N, N)
        np.fill_diagonal(sq_dist, np.inf)
        sorted_dist = np.sort(sq_dist, axis=1)                   # (N, N)
        sigmas = np.sqrt(sorted_dist[:, :k].mean(axis=1)) * 0.3  # (N,)
        sigmas = np.clip(sigmas, sigma_min, sigma_max)
    else:
        sigmas = np.full(len(points), fallback_sigma, dtype=np.float32)

    # Render each head as a Gaussian blob
    for (x, y), sigma in zip(points, sigmas):
        xi, yi = int(round(x)), int(round(y))
        # Clamp to valid image coordinates
        xi = max(0, min(img_w - 1, xi))
        yi = max(0, min(img_h - 1, yi))
        density[yi, xi] += 1.0

    # Single-pass Gaussian filter approximation:
    # Use a spatially varying approach by accumulating scaled impulses.
    # For correctness we render each head individually with its own sigma.
    density = np.zeros((img_h, img_w), dtype=np.float32)
    for (x, y), sigma in zip(points, sigmas):
        xi, yi = int(round(x)), int(round(y))
        xi = max(0, min(img_w - 1, xi))
        yi = max(0, min(img_h - 1, yi))

        # Local bounding box (±3σ)
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
# Dataset-specific loaders
# ---------------------------------------------------------------------------

def _iter_shanghaitech(data_dir: Path, split: str):
    """Yield (img_path, points_array) pairs for one ShanghaiTech split."""
    img_dir = data_dir / split / "images"
    gt_dir  = data_dir / split / "ground-truth"
    for img_file in sorted(img_dir.glob("*.jpg")):
        stem   = img_file.stem                      # IMG_100
        gt_file = gt_dir / f"GT_{stem}.mat"
        if not gt_file.exists():
            print(f"[WARN] Missing GT: {gt_file}")
            continue
        mat = scipy.io.loadmat(str(gt_file))
        pts = mat["image_info"][0, 0][0, 0][0].astype(np.float32)  # (N, 2) x,y
        yield img_file, pts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_for_split(iterator, out_root: Path, desc: str):
    out_root.mkdir(parents=True, exist_ok=True)
    for img_path, points in tqdm(iterator, desc=desc):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Cannot read image: {img_path}")
            continue
        h, w = img.shape[:2]

        # Filter out-of-bounds points
        if len(points) > 0:
            mask = (points[:, 0] >= 0) & (points[:, 0] < w) & \
                   (points[:, 1] >= 0) & (points[:, 1] < h)
            points = points[mask]

        density = _adaptive_gaussian_density(points, h, w)

        out_path = out_root / (img_path.stem + ".npy")
        np.save(str(out_path), density)


def main():
    parser = argparse.ArgumentParser(description="Generate adaptive Gaussian density maps")
    parser.add_argument("--dataset", required=True,
                        choices=["shanghaiA", "shanghaiB"],
                        help="Dataset identifier")
    parser.add_argument("--data-dir", required=True,
                        help="Root directory of the dataset")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    for split in ("train_data", "test_data"):
        out_root = data_dir / split / "gt_density_map"
        generate_for_split(
            _iter_shanghaitech(data_dir, split), out_root,
            desc=f"{args.dataset}/{split}"
        )

    print("Density map generation complete.")


if __name__ == "__main__":
    main()
