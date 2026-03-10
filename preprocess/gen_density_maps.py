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
  shanghaiA, shanghaiB, qnrf, unidata, mall, jhu

Usage
-----
# ShanghaiTech A
python preprocess/gen_density_maps.py --dataset shanghaiA \
    --data-dir data/ShanghaiTech/part_A

# UCF-QNRF (after preprocess_dataset.py resize)
python preprocess/gen_density_maps.py --dataset qnrf \
    --data-dir data/UCF-QNRF-processed

# Unidata (after convert_unidata.py)
python preprocess/gen_density_maps.py --dataset unidata \
    --data-dir data/Unidata/processed

# mall
python preprocess/gen_density_maps.py --dataset mall \
    --data-dir data/mall_dataset
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


def _iter_qnrf(data_dir: Path, split: str):
    """Yield (img_path, points_array) for UCF-QNRF.

    Supports both the original layout (split names: Train/Test, annotations: _ann.mat)
    and the DM-Count preprocessed layout (split names: train/test/val,
    annotations: <stem>.npy with shape (N,2)).
    """
    split_dir = data_dir / split
    if not split_dir.exists():
        return
    for img_file in sorted(split_dir.glob("*.jpg")):
        # Try DM-Count-processed .npy first
        npy_file = split_dir / (img_file.stem + ".npy")
        if npy_file.exists():
            pts = np.load(str(npy_file)).astype(np.float32)
            yield img_file, pts
            continue
        # Fall back to original _ann.mat
        ann_file = split_dir / (img_file.stem + "_ann.mat")
        if not ann_file.exists():
            print(f"[WARN] Missing annotation: {ann_file}")
            continue
        mat = scipy.io.loadmat(str(ann_file))
        pts = mat["annPoints"].astype(np.float32)
        yield img_file, pts


def _iter_unidata(data_dir: Path):
    """Yield (img_path, points_array) for converted Unidata."""
    img_root = data_dir / "images"
    lbl_root = data_dir / "labels"
    for img_file in sorted(img_root.rglob("*.jpg")):
        bucket = img_file.parent.name
        npy_file = lbl_root / bucket / f"{img_file.stem}.npy"
        if not npy_file.exists():
            print(f"[WARN] Missing annotation: {npy_file}")
            continue
        pts = np.load(str(npy_file))                # (N, 2) x,y
        yield img_file, pts


def _iter_mall(data_dir: Path):
    """Yield (img_path, points_array) for the mall dataset."""
    gt_mat = scipy.io.loadmat(str(data_dir / "mall_gt.mat"))
    frames_dir = data_dir / "frames"
    frame_data = gt_mat["frame"][0]                 # 2000-element object array

    for idx, fr in enumerate(frame_data):
        loc = fr["loc"][0, 0].astype(np.float32)    # (N, 2) x,y
        img_file = frames_dir / f"seq_{idx + 1:06d}.jpg"
        if not img_file.exists():
            print(f"[WARN] Missing frame: {img_file}")
            continue
        yield img_file, loc


def _iter_jhu(data_dir: Path, split: str):
    """Yield (img_path, points_array) for one JHU-Crowd++ split.

    GT format per line: x y w h o b  (x,y are head center coordinates).
    """
    img_dir = data_dir / split / "images"
    gt_dir  = data_dir / split / "gt"
    for img_file in sorted(img_dir.glob("*.jpg")):
        gt_file = gt_dir / (img_file.stem + ".txt")
        if not gt_file.exists():
            print(f"[WARN] Missing GT: {gt_file}")
            continue
        pts = []
        with open(gt_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    pts.append([float(parts[0]), float(parts[1])])
        yield img_file, np.array(pts, dtype=np.float32).reshape(-1, 2)


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
                        choices=["shanghaiA", "shanghaiB", "qnrf", "unidata", "mall", "jhu"],
                        # jhu: data/jhu_crowd  (train/val/test splits are all processed)
                        help="Dataset identifier")
    parser.add_argument("--data-dir", required=True,
                        help="Root directory of the dataset")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if args.dataset in ("shanghaiA", "shanghaiB"):
        for split in ("train_data", "test_data"):
            out_root = data_dir / split / "gt_density_map"
            generate_for_split(
                _iter_shanghaitech(data_dir, split), out_root,
                desc=f"{args.dataset}/{split}"
            )

    elif args.dataset == "qnrf":
        # Support both processed (train/val/test) and original (Train/Test) layouts
        for split in ("train", "val", "test", "Train", "Test"):
            if (data_dir / split).exists():
                out_root = data_dir / split / "gt_density_map"
                generate_for_split(
                    _iter_qnrf(data_dir, split), out_root,
                    desc=f"qnrf/{split}"
                )

    elif args.dataset == "unidata":
        out_root = data_dir / "gt_density_map"
        generate_for_split(_iter_unidata(data_dir), out_root, desc="unidata")

    elif args.dataset == "mall":
        out_root = data_dir / "gt_density_map"
        generate_for_split(_iter_mall(data_dir), out_root, desc="mall")

    elif args.dataset == "jhu":
        for split in ("train", "val", "test"):
            out_root = data_dir / split / "gt_density_map"
            generate_for_split(
                _iter_jhu(data_dir, split), out_root,
                desc=f"jhu/{split}"
            )

    print("Density map generation complete.")


if __name__ == "__main__":
    main()
