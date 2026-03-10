"""Generate CSRNet JSON list files for all datasets."""
import json
import glob
import os

BASE = '/ssd1/team_cam_ai/ntthai/crowd_counting'
CSR  = f'{BASE}/CSRNet'

datasets = {
    'part_A_train': f'{BASE}/data/ShanghaiTech/part_A/train_data/images/*.jpg',
    'part_A_test':  f'{BASE}/data/ShanghaiTech/part_A/test_data/images/*.jpg',
    'part_B_train': f'{BASE}/data/ShanghaiTech/part_B/train_data/images/*.jpg',
    'part_B_test':  f'{BASE}/data/ShanghaiTech/part_B/test_data/images/*.jpg',
    'qnrf_train':   f'{BASE}/data/UCF-QNRF-processed/train/*.jpg',
    'qnrf_val':     f'{BASE}/data/UCF-QNRF-processed/val/*.jpg',
    'qnrf_test':    f'{BASE}/data/UCF-QNRF-processed/test/*.jpg',
}

for name, pattern in datasets.items():
    paths = sorted(glob.glob(pattern))
    out = f'{CSR}/{name}.json'
    with open(out, 'w') as f:
        json.dump(paths, f)
    print(f'{name}: {len(paths)} paths → {out}')
