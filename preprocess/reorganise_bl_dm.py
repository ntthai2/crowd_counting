"""
Reorganise already-generated BL/DM point npy directories into the
train/val layout expected by Bayesian-Loss and DM-Count trainers.

For SHA / SHB:
    bl/train/  ← symlinks to train_data/images + BL npy computed fresh
    bl/val/    ← symlinks to test_data/images  + BL npy computed fresh

For QNRF-processed (lowercase split dirs already present):
    bl/train/  bl/val/  bl/test/  ← already correct, just check

For mall / Unidata (no natural split, 80/20 used):
    bl/train/  bl/val/

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


# ─── QNRF-processed ──────────────────────────────────────────────────────────

def process_qnrf(data_dir: Path, out_base: Path):
    for split in ('train', 'val', 'test'):
        img_dir = data_dir / split
        if not img_dir.exists():
            continue
        bl_dir = out_base / 'bl' / split
        dm_dir = out_base / 'dm' / split

        imgs = sorted(img_dir.glob('*.jpg'))
        print(f'  qnrf/{split}: {len(imgs)} images')
        for img_path in tqdm(imgs, leave=False):
            npy_path = img_path.with_suffix('.npy')
            if not npy_path.exists():
                continue
            pts = np.load(str(npy_path)).astype(np.float32).reshape(-1, 2)
            save_pair(img_path, pts, bl_dir, dm_dir)


# ─── mall ────────────────────────────────────────────────────────────────────

def process_mall(data_dir: Path, out_base: Path, val_frac: float = 0.2):
    import scipy.io as sio
    gt = sio.loadmat(str(data_dir / 'mall_gt.mat'))
    locations = gt['frame'][0]  # list of per-frame struct

    imgs = sorted((data_dir / 'frames').glob('*.jpg'))
    n_val = max(1, int(len(imgs) * val_frac))
    splits = {
        'val':   imgs[-n_val:],
        'train': imgs[:-n_val],
    }
    for split_name, img_list in splits.items():
        bl_dir = out_base / 'bl' / split_name
        dm_dir = out_base / 'dm' / split_name
        print(f'  mall/{split_name}: {len(img_list)} images')
        for img_path in tqdm(img_list, leave=False):
            idx = int(img_path.stem.replace('seq_', '')) - 1
            loc = locations[idx][0][0][0]                 # (N, 2)
            pts = loc.astype(np.float32)[:, :2]
            save_pair(img_path, pts, bl_dir, dm_dir)


# ─── Unidata ─────────────────────────────────────────────────────────────────

def process_unidata(data_dir: Path, out_base: Path, val_frac: float = 0.2):
    label_dir = data_dir / 'labels'
    imgs = sorted(data_dir.glob('images/*.jpg'))
    if not imgs:
        imgs = sorted(data_dir.glob('*.jpg'))
    n_val = max(1, int(len(imgs) * val_frac))
    splits = {
        'val':   imgs[-n_val:],
        'train': imgs[:-n_val],
    }
    for split_name, img_list in splits.items():
        bl_dir = out_base / 'bl' / split_name
        dm_dir = out_base / 'dm' / split_name
        print(f'  unidata/{split_name}: {len(img_list)} images')
        for img_path in tqdm(img_list, leave=False):
            npy_path = label_dir / (img_path.stem + '.npy')
            if not npy_path.exists():
                npy_path = img_path.with_suffix('.npy')
            if not npy_path.exists():
                continue
            pts = np.load(str(npy_path)).astype(np.float32).reshape(-1, 2)
            save_pair(img_path, pts, bl_dir, dm_dir)


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shanghaiA', type=str)
    parser.add_argument('--shanghaiB', type=str)
    parser.add_argument('--qnrf',      type=str)
    parser.add_argument('--mall',      type=str)
    parser.add_argument('--unidata',   type=str)
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

    if args.qnrf:
        p = Path(args.qnrf)
        print(f'Processing UCF-QNRF ...')
        for d in ('bl', 'dm'):
            flat = p / d
            if flat.exists() and not (flat / 'train').exists():
                shutil.rmtree(flat)
        process_qnrf(p, p)

    if args.mall:
        p = Path(args.mall)
        print(f'Processing mall ...')
        for d in ('bl', 'dm'):
            flat = p / d
            if flat.exists() and not (flat / 'train').exists():
                shutil.rmtree(flat)
        process_mall(p, p)

    if args.unidata:
        p = Path(args.unidata)
        print(f'Processing Unidata ...')
        for d in ('bl', 'dm'):
            flat = p / d
            if flat.exists() and not (flat / 'train').exists():
                shutil.rmtree(flat)
        process_unidata(p, p)

    print('\nDone.')

if __name__ == '__main__':
    main()
