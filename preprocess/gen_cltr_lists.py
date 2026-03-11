"""Generate CLTR .npy image-path list files for SHA and SHB.

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

print("Done. NPY files saved to", NPY_DIR)
