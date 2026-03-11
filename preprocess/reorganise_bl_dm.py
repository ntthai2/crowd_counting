"""
Reorganise already-generated BL/DM point npy directories into the
train/val layout expected by Bayesian-Loss and DM-Count trainers.

For SHA / SHB:
    bl/train/  ← symlinks to train_data/images + BL npy computed fresh
    bl/val/    ← symlinks to test_data/images  + BL npy computed fresh

This script *replaces* the existing flat bl/ and dm/ directories.
"""

import argparse
import os
import shutil
from pathlib import Path
import numpy as np
import scipy.io
from glob import glob
from tqdm import tqdm

K_NN = 3


def knn_dist(pts: np.ndarray, k: int = K_NN) -> np.ndarray:
    if len(pts) <= k:
        return np.full(len(pts), 15.0, dtype=np.float32)
    diff    = pts[:, None, :] - pts[None, :, :]
    sq_dist = (diff ** 2).sum(axis=2)
    np.fill_diagonal(sq_dist, np.inf)
    sorted_d = np.sort(sq_dist, axis=1)
    return np.sqrt(sorted_d[:, :k].mean(axis=1)).astype(np.float32)


def save_pair(img_path: Path, pts_xy: np.ndarray, bl_dir: Path, dm_dir: Path):
    pts_xy = pts_xy.astype(np.float32).reshape(-1, 2)
    for out_dir in (bl_dir, dm_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        lnk = out_dir / img_path.name
        if not lnk.exists():
            lnk.symlink_to(img_path.resolve())

    nn_d   = knn_dist(pts_xy)
    pts_bl = np.concatenate([pts_xy, nn_d[:, None]], axis=1)
    np.save(str(bl_dir / (img_path.stem + '.npy')), pts_bl)
    np.save(str(dm_dir / (img_path.stem + '.npy')), pts_xy)


# ─── SHA / SHB ───────────────────────────────────────────────────────────────

def process_sha_shb(data_dir: Path, out_base: Path, suffix: str):
    """data_dir = data/ShanghaiTech/part_A or part_B"""
    for split_name, img_split, gt_split in [
        ('train', 'train_data', 'train_data'),
        ('val',   'test_data',  'test_data'),
    ]:
        img_dir = data_dir / img_split / 'images'
        gt_dir  = data_dir / img_split / 'ground-truth'
        bl_dir  = out_base / 'bl' / split_name
        dm_dir  = out_base / 'dm' / split_name

        imgs = sorted(img_dir.glob('*.jpg'))
        print(f'  {suffix}/{split_name}: {len(imgs)} images')
        for img_path in tqdm(imgs, leave=False):
            gt_file = gt_dir / f'GT_{img_path.stem}.mat'
            mat = scipy.io.loadmat(str(gt_file))
            pts = mat['image_info'][0, 0][0, 0][0].astype(np.float32)
            save_pair(img_path, pts, bl_dir, dm_dir)


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shanghaiA', type=str)
    parser.add_argument('--shanghaiB', type=str)

    args = parser.parse_args()

    if args.shanghaiA:
        p = Path(args.shanghaiA)
        print(f'Processing ShanghaiTech A ...')
        # Remove old flat dirs
        for d in ('bl', 'dm'):
            flat = p / d
            if flat.exists() and not (flat / 'train').exists():
                print(f'  Removing old flat {flat}')
                shutil.rmtree(flat)
        process_sha_shb(p, p, 'SHA')

    if args.shanghaiB:
        p = Path(args.shanghaiB)
        print(f'Processing ShanghaiTech B ...')
        for d in ('bl', 'dm'):
            flat = p / d
            if flat.exists() and not (flat / 'train').exists():
                shutil.rmtree(flat)
        process_sha_shb(p, p, 'SHB')

    print('\nDone.')

if __name__ == '__main__':
    main()
