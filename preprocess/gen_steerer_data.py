"""Generate STEERER dataset structure for SHA, SHB, QNRF.

STEERER expects per-dataset:
  {root}/images/{id}.jpg      (or symlink)
  {root}/jsons/{id}.json      {"points": [[x1,y1], [x2,y2], ...]}
  {root}/train.txt            → "id count class_id\n" per line
  {root}/val.txt              (same format)
  {root}/test.txt             (same format)

Then a Python config file in STEERER/configs/{dataset}_final.py.

Usage:
    python preprocess/gen_steerer_data.py
"""
import os
import glob
import json
import hashlib
import numpy as np
import scipy.io as io

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEERER_DIR = os.path.join(BASE, 'STEERER')
PROCESSED_BASE = os.path.join(BASE, 'data', 'steerer')


def read_sha_points(mat_path, prefix='GT_'):
    """Read SHA/SHB mat -> (N,2) float32 (x,y)."""
    mat = io.loadmat(mat_path)
    pts = mat['image_info'][0, 0][0, 0][0].astype(np.float32)
    return pts  # (N,2): x,y


def read_qnrf_points(npy_path):
    """Read QNRF point npy (N,2) float32 (x,y)."""
    pts = np.load(npy_path).astype(np.float32)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 2)
    return pts


def make_steerer_dataset(name, out_root, splits_data):
    """
    splits_data: dict of split -> list of (img_path, points_array)
    Creates {out_root}/images/, {out_root}/jsons/, and split txt files.
    """
    img_out = os.path.join(out_root, 'images')
    jsn_out = os.path.join(out_root, 'jsons')
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(jsn_out, exist_ok=True)

    split_files = {}
    for split, entries in splits_data.items():
        lines = []
        for img_path, pts in entries:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            # Symlink image
            link = os.path.join(img_out, stem + '.jpg')
            if not os.path.exists(link):
                os.symlink(img_path, link)
            # Write JSON GT
            json_path = os.path.join(jsn_out, stem + '.json')
            with open(json_path, 'w') as f:
                json.dump({'points': pts.tolist()}, f)
            count = len(pts)
            lines.append(f"{stem} {count} 0\n")

        split_files[split] = lines
        txt_path = os.path.join(out_root, f'{split}.txt')
        with open(txt_path, 'w') as f:
            f.writelines(lines)
        print(f"  {name}/{split}: {len(lines)} entries → {txt_path}")

    return split_files


def write_config(name, out_root, config_name=None):
    if config_name is None:
        config_name = name.upper()
    cfg_path = os.path.join(STEERER_DIR, 'configs', f'{config_name}_our.py')
    rel_root = os.path.relpath(out_root, STEERER_DIR)
    cfg_content = f"""# Auto-generated STEERER config for {name}
gpus = (0,)
log_dir = 'exp'
workers = 4
print_freq = 30
seed = 42

network = dict(
    backbone="MocHRBackbone",
    sub_arch='hrnet48',
    counter_type='withMOE',
    resolution_num=[0, 1, 2, 3],
    loss_weight=[1., 1./2, 1./4, 1./8],
    sigma=[4],
    gau_kernel_size=11,
    baseline_loss=False,
    pretrained_backbone="../PretrainedModels/hrnetv2_w48_imagenet_pretrained.pth",
    head=dict(
        type='CountingHead',
        fuse_method='cat',
        in_channels=96,
        stages_channel=[384, 192, 96, 48],
        inter_layer=[64, 32, 16],
        out_channels=1,
    )
)

dataset = dict(
    name='{config_name}',
    root='{rel_root}',
    test_set='test.txt',
    train_set='train.txt',
    loc_gt='test_gt_loc.txt',
    num_classes=len(network['resolution_num']),
    den_factor=100,
    extra_train_set=None,
)

optimizer = dict(
    NAME='adamw',
    BASE_LR=1e-5,
    BETAS=(0.9, 0.999),
    WEIGHT_DECAY=1e-2,
    EPS=1.0e-08,
    MOMENTUM=0.9,
    AMSGRAD=False,
    NESTEROV=True,
)

lr_config = dict(
    NAME='cosine',
    WARMUP_METHOD='linear',
    DECAY_EPOCHS=250,
    DECAY_RATE=0.1,
    WARMUP_EPOCHS=10,
    WARMUP_LR=5.0e-07,
    MIN_LR=1.0e-07,
)

log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook')],
)

train = dict(
    counter='normal',
    image_size=(768, 768),
    route_size=(256, 256),
    base_size=2048,
    batch_size_per_gpu=4,
    shuffle=True,
    begin_epoch=0,
    end_epoch=400,
    extra_epoch=0,
    extra_lr=0,
    resume_path=None,
    flip=True,
    multi_scale=True,
    scale_factor=(0.5, 1/0.5),
    val_span=[-400, -200, -100, -50, -50],
    downsamplerate=1,
    ignore_label=255,
)

test = dict(
    image_size=(1024, 2048),
    base_size=2048,
    loc_base_size=(768, 2048),
    loc_threshold=0.2,
    batch_size_per_gpu=1,
    patch_batch_size=16,
    flip_test=False,
    multi_scale=False,
    model_file='',
)

CUDNN = dict(BENCHMARK=True, DETERMINISTIC=False, ENABLED=True)
"""
    with open(cfg_path, 'w') as f:
        f.write(cfg_content)
    print(f"  Config written: {cfg_path}")


# ── ShanghaiTech A ────────────────────────────────────────────────────────────
print("\nSHA:")
sha_root = os.path.join(BASE, 'data/ShanghaiTech/part_A')
sha_out  = os.path.join(PROCESSED_BASE, 'SHHA')

sha_splits = {}
for split, subdir in [('train', 'train_data'), ('test', 'test_data')]:
    entries = []
    img_dir = os.path.join(sha_root, subdir, 'images')
    gt_dir  = os.path.join(sha_root, subdir, 'ground-truth')
    for img_path in sorted(glob.glob(os.path.join(img_dir, '*.jpg'))):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        mat_path = os.path.join(gt_dir, 'GT_' + stem + '.mat')
        pts = read_sha_points(mat_path)
        entries.append((img_path, pts))
    sha_splits[split] = entries

make_steerer_dataset('sha', sha_out, sha_splits)
write_config('sha', sha_out, config_name='SHHA')

# ── ShanghaiTech B ────────────────────────────────────────────────────────────
print("\nSHB:")
shb_root = os.path.join(BASE, 'data/ShanghaiTech/part_B')
shb_out  = os.path.join(PROCESSED_BASE, 'SHHB')

shb_splits = {}
for split, subdir in [('train', 'train_data'), ('test', 'test_data')]:
    entries = []
    img_dir = os.path.join(shb_root, subdir, 'images')
    gt_dir  = os.path.join(shb_root, subdir, 'ground-truth')
    for img_path in sorted(glob.glob(os.path.join(img_dir, '*.jpg'))):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        mat_path = os.path.join(gt_dir, 'GT_' + stem + '.mat')
        pts = read_sha_points(mat_path)
        entries.append((img_path, pts))
    shb_splits[split] = entries

make_steerer_dataset('shb', shb_out, shb_splits)
write_config('shb', shb_out, config_name='SHHB')

# ── UCF-QNRF ─────────────────────────────────────────────────────────────────
print("\nQNRF:")
qnrf_root = os.path.join(BASE, 'data/UCF-QNRF-processed')
qnrf_out  = os.path.join(PROCESSED_BASE, 'QNRF')

qnrf_splits = {}
for split in ['train', 'val', 'test']:
    entries = []
    split_dir = os.path.join(qnrf_root, split)
    for img_path in sorted(glob.glob(os.path.join(split_dir, '*.jpg'))):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        # QNRF has dm/<stem>.npy (x,y) points
        npy_path = os.path.join(qnrf_root, split, 'dm', stem + '.npy')
        if not os.path.exists(npy_path):
            # fallback: the direct .npy alongside jpg
            npy_path = img_path.replace('.jpg', '.npy')
        pts = read_qnrf_points(npy_path)
        entries.append((img_path, pts))
    qnrf_splits[split] = entries

make_steerer_dataset('qnrf', qnrf_out, qnrf_splits)
write_config('qnrf', qnrf_out, config_name='QNRF')

print("\nDone.")
