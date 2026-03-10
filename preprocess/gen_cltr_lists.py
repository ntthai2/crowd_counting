"""Generate CLTR .npy image-path list files for SHA, SHB, QNRF, and unified.

CLTR expects:
  np.save('./npydata/<name>_train.npy', list_of_jpg_paths)
  np.save('./npydata/<name>_val.npy',   list_of_jpg_paths)

image.py resolves gt_detr_map h5 from image path (fixed to handle both layouts).

Usage:
    python preprocess/gen_cltr_lists.py
"""
import os
import glob
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLTR_DIR = os.path.join(BASE, 'CLTR')
NPY_DIR = os.path.join(CLTR_DIR, 'npydata')
os.makedirs(NPY_DIR, exist_ok=True)

TRANSCROWD_NPY = os.path.join(BASE, 'TransCrowd', 'npydata')


def gather_jpgs(img_dir):
    """Return sorted list of absolute jpg paths in img_dir."""
    return sorted(glob.glob(os.path.join(img_dir, '*.jpg')))


# ── SHA ───────────────────────────────────────────────────────────────────────
sha_root = os.path.join(BASE, 'data/ShanghaiTech/part_A')
sha_train = gather_jpgs(os.path.join(sha_root, 'train_data', 'images'))
sha_test  = gather_jpgs(os.path.join(sha_root, 'test_data', 'images'))

np.save(os.path.join(NPY_DIR, 'sha_train.npy'), sha_train)
np.save(os.path.join(NPY_DIR, 'sha_val.npy'),   sha_test)
print(f"SHA: train={len(sha_train)}, val={len(sha_test)}")

# ── SHB ───────────────────────────────────────────────────────────────────────
shb_root = os.path.join(BASE, 'data/ShanghaiTech/part_B')
shb_train = gather_jpgs(os.path.join(shb_root, 'train_data', 'images'))
shb_test  = gather_jpgs(os.path.join(shb_root, 'test_data', 'images'))

np.save(os.path.join(NPY_DIR, 'shb_train.npy'), shb_train)
np.save(os.path.join(NPY_DIR, 'shb_val.npy'),   shb_test)
print(f"SHB: train={len(shb_train)}, val={len(shb_test)}")

# ── QNRF ──────────────────────────────────────────────────────────────────────
qnrf_root = os.path.join(BASE, 'data/UCF-QNRF-processed')
qnrf_train = gather_jpgs(os.path.join(qnrf_root, 'train'))
qnrf_val   = gather_jpgs(os.path.join(qnrf_root, 'val'))
qnrf_test  = gather_jpgs(os.path.join(qnrf_root, 'test'))

np.save(os.path.join(NPY_DIR, 'qnrf_train.npy'), qnrf_train)
np.save(os.path.join(NPY_DIR, 'qnrf_val.npy'),   qnrf_val)
np.save(os.path.join(NPY_DIR, 'qnrf_test.npy'),  qnrf_test)
print(f"QNRF: train={len(qnrf_train)}, val={len(qnrf_val)}, test={len(qnrf_test)}")

# ── Unified (from TransCrowd split npy files) ──────────────────────────────
# TransCrowd unified_{train,val,test}.npy contain image paths (first column of each entry)
def load_transcrowd_paths(npy_path):
    data = np.load(npy_path, allow_pickle=True)
    paths = []
    for item in data:
        # Each item is a dict with 'path' key (from lazy loading rewrite)
        if isinstance(item, dict):
            paths.append(item['path'])
        else:
            paths.append(str(item))
    return paths

try:
    uni_train = load_transcrowd_paths(os.path.join(TRANSCROWD_NPY, 'unified_train.npy'))
    uni_val   = load_transcrowd_paths(os.path.join(TRANSCROWD_NPY, 'unified_val.npy'))
    uni_test  = load_transcrowd_paths(os.path.join(TRANSCROWD_NPY, 'unified_test.npy'))

    np.save(os.path.join(NPY_DIR, 'unified_train.npy'), uni_train)
    np.save(os.path.join(NPY_DIR, 'unified_val.npy'),   uni_val)
    np.save(os.path.join(NPY_DIR, 'unified_test.npy'),  uni_test)
    print(f"Unified: train={len(uni_train)}, val={len(uni_val)}, test={len(uni_test)}")
except Exception as e:
    print(f"Unified: skipped ({e})")

print("Done. NPY files saved to", NPY_DIR)
