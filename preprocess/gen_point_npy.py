"""
gen_point_npy.py
----------------
Generate per-image point annotation numpy files in the formats expected by
Bayesian-Loss (BL) and DM-Count.

BL  (Bayesian-Loss/datasets/crowd.py):
    Expects  img_path.replace('jpg', 'npy')  →  (N, 3) float32  [x, y, nn_dist]
    The 3rd column is the mean distance to the k nearest neighbours,
    used by the Bayesian crowd counting loss.

DM-Count (DM-Count/datasets/crowd.py):
    Expects  img_path.replace('jpg', 'npy')  →  (N, 2) float32  [x, y]

Because both conventions map to the same filename, we write to separate
output directories:
    --out-dir-bl         e.g.  data/unified_bl/
    --out-dir-dm         e.g.  data/unified_dm/

Images are symlinked (or optionally copied) next to the .npy files so that
the models' glob-based loaders can find both.

Usage
-----
python preprocess/gen_point_npy.py --dataset shanghaiA \
    --data-dir  data/ShanghaiTech/part_A \
    --out-dir-bl data/ShanghaiTech/part_A/bl \
    --out-dir-dm data/ShanghaiTech/part_A/dm

python preprocess/gen_point_npy.py --dataset unified \
    --npy-list  TransCrowd/npydata/unified_train.npy \
               TransCrowd/npydata/unified_val.npy \
               TransCrowd/npydata/unified_test.npy \
    --npy-points-dir data/Unidata/processed/labels \
    --out-dir-bl data/unified_bl \
    --out-dir-dm data/unified_dm
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import scipy.io
from tqdm import tqdm


K_NN = 3   # nearest neighbours for sigma / distance computation


def _knn_distances(points: np.ndarray, k: int = K_NN) -> np.ndarray:
    """Return per-point mean k-NN distance (float32, shape (N,))."""
    if len(points) <= k:
        return np.full(len(points), 15.0, dtype=np.float32)
    diff    = points[:, None, :] - points[None, :, :]   # (N, N, 2)
    sq_dist = (diff ** 2).sum(axis=2)                    # (N, N)
    np.fill_diagonal(sq_dist, np.inf)
    sorted_d = np.sort(sq_dist, axis=1)
    return np.sqrt(sorted_d[:, :k].mean(axis=1)).astype(np.float32)


def _save(img_path: Path, points_xy: np.ndarray, out_bl: Path, out_dm: Path):
    """Write BL (N,3) and DM-Count (N,2) npy files and symlink the image."""
    points_xy = points_xy.astype(np.float32).reshape(-1, 2)

    for out_dir in (out_bl, out_dm):
        out_dir.mkdir(parents=True, exist_ok=True)
        img_link = out_dir / img_path.name
        if not img_link.exists():
            try:
                img_link.symlink_to(img_path.resolve())
            except OSError:
                pass

    # BL: (N, 3) — [x, y, nn_dist]
    nn_dist = _knn_distances(points_xy)
    pts_bl  = np.concatenate([points_xy, nn_dist[:, None]], axis=1)
    np.save(str(out_bl / (img_path.stem + ".npy")), pts_bl)

    # DM-Count: (N, 2) — [x, y]
    np.save(str(out_dm / (img_path.stem + ".npy")), points_xy)


# ---------------------------------------------------------------------------
# Dataset iterators  →  (img_path, (N,2) points)
# ---------------------------------------------------------------------------

def _iter_shanghaitech(data_dir: Path):
    for split in ("train_data", "test_data"):
        img_dir = data_dir / split / "images"
        gt_dir  = data_dir / split / "ground-truth"
        for img_file in sorted(img_dir.glob("*.jpg")):
            gt_file = gt_dir / f"GT_{img_file.stem}.mat"
            if not gt_file.exists():
                continue
            mat = scipy.io.loadmat(str(gt_file))
            pts = mat["image_info"][0, 0][0, 0][0].astype(np.float32)
            yield img_file, pts


def _iter_qnrf(data_dir: Path):
    """Supports both processed (.npy) and original (_ann.mat) QNRF layouts."""
    for split in ("train", "val", "test", "Train", "Test"):
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        for img_file in sorted(split_dir.glob("*.jpg")):
            npy_file = split_dir / (img_file.stem + ".npy")
            if npy_file.exists():
                yield img_file, np.load(str(npy_file)).astype(np.float32)
                continue
            ann_file = split_dir / (img_file.stem + "_ann.mat")
            if not ann_file.exists():
                continue
            mat = scipy.io.loadmat(str(ann_file))
            yield img_file, mat["annPoints"].astype(np.float32)


def _iter_unidata(data_dir: Path):
    img_root = data_dir / "images"
    lbl_root = data_dir / "labels"
    for img_file in sorted(img_root.rglob("*.jpg")):
        bucket   = img_file.parent.name
        npy_file = lbl_root / bucket / f"{img_file.stem}.npy"
        if not npy_file.exists():
            continue
        yield img_file, np.load(str(npy_file))


def _iter_mall(data_dir: Path):
    gt_mat     = scipy.io.loadmat(str(data_dir / "mall_gt.mat"))
    frames_dir = data_dir / "frames"
    for idx, fr in enumerate(gt_mat["frame"][0]):
        loc      = fr["loc"][0, 0].astype(np.float32)
        img_file = frames_dir / f"seq_{idx + 1:06d}.jpg"
        if not img_file.exists():
            continue
        yield img_file, loc


def _iter_jhu(data_dir: Path):
    """GT format per line: x y w h o b  (x,y = head centers)."""
    for split in ("train", "val", "test"):
        img_dir = data_dir / split / "images"
        gt_dir  = data_dir / split / "gt"
        for img_file in sorted(img_dir.glob("*.jpg")):
            gt_file = gt_dir / (img_file.stem + ".txt")
            if not gt_file.exists():
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

def main():
    parser = argparse.ArgumentParser(description="Generate BL/DM-Count point npy files")
    parser.add_argument("--dataset", required=True,
                        choices=["shanghaiA", "shanghaiB", "qnrf", "unidata", "mall", "jhu"])
    parser.add_argument("--data-dir",    required=True)
    parser.add_argument("--out-dir-bl",  required=True,
                        help="Output root for BL (N,3) npy + symlinked images")
    parser.add_argument("--out-dir-dm",  required=True,
                        help="Output root for DM-Count (N,2) npy + symlinked images")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_bl   = Path(args.out_dir_bl)
    out_dm   = Path(args.out_dir_dm)

    datasets = {
        "shanghaiA": _iter_shanghaitech,
        "shanghaiB": _iter_shanghaitech,
        "qnrf":      _iter_qnrf,
        "unidata":   _iter_unidata,
        "mall":      _iter_mall,
        "jhu":       _iter_jhu,
    }
    iterator = datasets[args.dataset](data_dir)

    for img_path, pts in tqdm(iterator, desc=args.dataset):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        if len(pts) > 0:
            mask = (pts[:, 0] >= 0) & (pts[:, 0] < w) & \
                   (pts[:, 1] >= 0) & (pts[:, 1] < h)
            pts = pts[mask]
        _save(img_path, pts, out_bl, out_dm)

    print(f"Done.\n  BL  → {out_bl}\n  DM  → {out_dm}")


if __name__ == "__main__":
    main()
