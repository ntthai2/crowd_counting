"""
create_unified_split.py
-----------------------
Pool images from multiple crowd counting datasets into a single
train / val / test split and write npy file lists compatible with
TransCrowd's npydata loader format.

Sources included (all have dense point annotations):
  - ShanghaiTech Part A  (train_data + test_data)
  - ShanghaiTech Part B  (train_data + test_data)
  - UCF-QNRF             (Train + Test, after preprocess resize)
  - Unidata              (all images, after convert_unidata.py)
  - mall                 (all 2000 frames)

UCF-CC-50 is deliberately excluded (5-fold protocol, small set).

Output npy files contain lists of image absolute paths.
A paired `<stem>_counts.npy` list of integer counts is also saved
(used by the regression models; density-map models derive count from sum).

Usage
-----
python preprocess/create_unified_split.py \
    --shanghaiA  data/ShanghaiTech/part_A \
    --shanghaiB  data/ShanghaiTech/part_B \
    --qnrf       data/UCF-QNRF-processed \
    --unidata    data/Unidata/processed \
    --mall       data/mall_dataset \
    --out-dir    TransCrowd/npydata \
    --split      0.70 0.15 0.15 \
    --seed       42
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import scipy.io
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers: collect (img_path, count) pairs per dataset
# ---------------------------------------------------------------------------

def collect_shanghaitech(data_dir: Path) -> list:
    items = []
    for split in ("train_data", "test_data"):
        img_dir = data_dir / split / "images"
        gt_dir  = data_dir / split / "ground-truth"
        for img_file in sorted(img_dir.glob("*.jpg")):
            stem    = img_file.stem
            gt_file = gt_dir / f"GT_{stem}.mat"
            if not gt_file.exists():
                continue
            mat = scipy.io.loadmat(str(gt_file))
            pts = mat["image_info"][0, 0][0, 0][0]
            items.append((str(img_file.resolve()), len(pts)))
    return items


def collect_qnrf(data_dir: Path) -> list:
    items = []
    for split in ("train", "val", "test", "Train", "Test"):
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        for img_file in sorted(split_dir.glob("*.jpg")):
            # Try DM-Count-processed .npy first, then original _ann.mat
            npy_file = split_dir / (img_file.stem + ".npy")
            if npy_file.exists():
                pts = np.load(str(npy_file))
                items.append((str(img_file.resolve()), len(pts)))
                continue
            ann_file = split_dir / (img_file.stem + "_ann.mat")
            if not ann_file.exists():
                continue
            mat = scipy.io.loadmat(str(ann_file))
            pts = mat["annPoints"]
            items.append((str(img_file.resolve()), len(pts)))
    return items


def collect_unidata(data_dir: Path) -> list:
    """Expects convert_unidata.py to have run; reads .npy point files."""
    items = []
    img_root = data_dir / "images"
    lbl_root = data_dir / "labels"
    for img_file in sorted(img_root.rglob("*.jpg")):
        bucket   = img_file.parent.name
        npy_file = lbl_root / bucket / f"{img_file.stem}.npy"
        if not npy_file.exists():
            continue
        pts = np.load(str(npy_file))
        items.append((str(img_file.resolve()), len(pts)))
    return items


def collect_jhu(data_dir: Path) -> list:
    """JHU-Crowd++ GT: space-separated x y w h o b (x,y = head centers)."""
    items = []
    for split in ("train", "val", "test"):
        img_dir = data_dir / split / "images"
        gt_dir  = data_dir / split / "gt"
        for img_file in sorted(img_dir.glob("*.jpg")):
            gt_file = gt_dir / (img_file.stem + ".txt")
            if not gt_file.exists():
                continue
            count = sum(
                1 for line in open(gt_file)
                if len(line.strip().split()) >= 2
            )
            items.append((str(img_file.resolve()), count))
    return items


def collect_mall(data_dir: Path) -> list:
    items = []
    gt_mat     = scipy.io.loadmat(str(data_dir / "mall_gt.mat"))
    frames_dir = data_dir / "frames"
    frame_data = gt_mat["frame"][0]
    counts     = gt_mat["count"].flatten()          # (2000,)
    for idx, count in enumerate(counts):
        img_file = frames_dir / f"seq_{idx + 1:06d}.jpg"
        if not img_file.exists():
            continue
        items.append((str(img_file.resolve()), int(count)))
    return items


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Create unified crowd counting split")
    parser.add_argument("--shanghaiA", default=None, help="ShanghaiTech Part A root")
    parser.add_argument("--shanghaiB", default=None, help="ShanghaiTech Part B root")
    parser.add_argument("--qnrf",      default=None, help="UCF-QNRF root (after resize)")
    parser.add_argument("--unidata",   default=None, help="Unidata processed root")
    parser.add_argument("--mall",      default=None, help="mall_dataset root")
    parser.add_argument("--jhu",       default=None, help="JHU-Crowd++ root (data/jhu_crowd)")
    parser.add_argument("--out-dir",   required=True, help="Output directory for npy lists")
    parser.add_argument("--split", nargs=3, type=float, default=[0.70, 0.15, 0.15],
                        help="Train/val/test fractions (must sum to 1)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    assert abs(sum(args.split) - 1.0) < 1e-6, "Split fractions must sum to 1"

    # Collect all items
    all_items: list[tuple[str, int]] = []

    if args.shanghaiA:
        items = collect_shanghaitech(Path(args.shanghaiA))
        print(f"ShanghaiTech A: {len(items)} images")
        all_items.extend(items)

    if args.shanghaiB:
        items = collect_shanghaitech(Path(args.shanghaiB))
        print(f"ShanghaiTech B: {len(items)} images")
        all_items.extend(items)

    if args.qnrf:
        items = collect_qnrf(Path(args.qnrf))
        print(f"UCF-QNRF: {len(items)} images")
        all_items.extend(items)

    if args.unidata:
        items = collect_unidata(Path(args.unidata))
        print(f"Unidata: {len(items)} images")
        all_items.extend(items)

    if args.mall:
        items = collect_mall(Path(args.mall))
        print(f"mall: {len(items)} images")
        all_items.extend(items)

    if args.jhu:
        items = collect_jhu(Path(args.jhu))
        print(f"JHU-Crowd++: {len(items)} images")
        all_items.extend(items)

    if not all_items:
        raise ValueError("No dataset paths provided or no images found.")

    print(f"Total: {len(all_items)} images, "
          f"count range [{min(c for _,c in all_items)}, {max(c for _,c in all_items)}]")

    # Shuffle and split
    rng = random.Random(args.seed)
    rng.shuffle(all_items)

    n      = len(all_items)
    n_tr   = int(round(n * args.split[0]))
    n_val  = int(round(n * args.split[1]))
    n_test = n - n_tr - n_val

    splits = {
        "train": all_items[:n_tr],
        "val":   all_items[n_tr: n_tr + n_val],
        "test":  all_items[n_tr + n_val:],
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_items in splits.items():
        img_paths = [p for p, _ in split_items]
        counts    = [c for _, c in split_items]

        npy_path   = out_dir / f"unified_{split_name}.npy"
        count_path = out_dir / f"unified_{split_name}_counts.npy"

        np.save(str(npy_path),   img_paths)
        np.save(str(count_path), counts)
        print(f"  {split_name:5s}: {len(split_items):5d} images → {npy_path}")

    print("Done.")


if __name__ == "__main__":
    main()
