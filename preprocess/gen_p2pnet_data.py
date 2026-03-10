"""Generate P2PNet annotation .txt files and .list files for SHA and SHB.

Usage:
    python preprocess/gen_p2pnet_data.py
"""
import os
import glob
import numpy as np
import scipy.io as io

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASETS = {
    'shanghaiA': {
        'root': os.path.join(BASE, 'data/ShanghaiTech/part_A'),
        'splits': {
            'train': 'train_data',
            'test': 'test_data',
        },
        'gt_prefix': 'GT_',
        'list_name': 'sha',
    },
    'shanghaiB': {
        'root': os.path.join(BASE, 'data/ShanghaiTech/part_B'),
        'splits': {
            'train': 'train_data',
            'test': 'test_data',
        },
        'gt_prefix': 'GT_',
        'list_name': 'shb',
    },
}


def read_sha_mat(mat_path):
    """Read SHA/SHB ground truth mat file -> (N, 2) float32 array of (x, y)."""
    mat = io.loadmat(mat_path)
    points = mat['image_info'][0, 0][0, 0][0]  # (N, 2): x, y
    return points.astype(np.float32)


def process_dataset(name, cfg):
    root = cfg['root']
    ann_dir = os.path.join(root, 'annotations')
    os.makedirs(ann_dir, exist_ok=True)

    list_lines = {'train': [], 'test': []}

    for split, subdir in cfg['splits'].items():
        img_dir = os.path.join(root, subdir, 'images')
        gt_dir = os.path.join(root, subdir, 'ground-truth')
        ann_split_dir = os.path.join(ann_dir, split)
        os.makedirs(ann_split_dir, exist_ok=True)

        imgs = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
        print(f"  {name}/{split}: {len(imgs)} images")

        for img_path in imgs:
            stem = os.path.splitext(os.path.basename(img_path))[0]  # e.g. IMG_1
            mat_name = cfg['gt_prefix'] + stem + '.mat'
            mat_path = os.path.join(gt_dir, mat_name)

            points = read_sha_mat(mat_path)

            # Write annotation .txt (x y per line)
            ann_path = os.path.join(ann_split_dir, stem + '.txt')
            with open(ann_path, 'w') as f:
                for pt in points:
                    f.write(f"{pt[0]:.4f} {pt[1]:.4f}\n")

            # Relative paths (relative to root, for .list file)
            rel_img = os.path.relpath(img_path, root)
            rel_ann = os.path.relpath(ann_path, root)
            list_lines[split].append(f"{rel_img} {rel_ann}\n")

    # Write .list files into root (where --data_root points)
    list_name = cfg['list_name']
    for split, lines in list_lines.items():
        fname = os.path.join(root, f'{list_name}_{split}.list')
        with open(fname, 'w') as f:
            f.writelines(lines)
        print(f"  Wrote: {fname} ({len(lines)} entries)")


if __name__ == '__main__':
    for name, cfg in DATASETS.items():
        print(f"\nProcessing {name}...")
        process_dataset(name, cfg)
    print("\nDone.")
