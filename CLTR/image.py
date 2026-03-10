import os
import cv2
import h5py
import numpy as np
from PIL import Image

def _gt_path_from_img(img_path):
    """Return the CLTR h5 GT path for a given image path.
    Handles two layouts:
      - SHA/SHB: …/images/IMG_xx.jpg  →  …/gt_detr_map/IMG_xx.h5
      - QNRF/mall/flat: …/train/img_xx.jpg  →  …/train/gt_detr_map/img_xx.h5
    """
    stem = os.path.splitext(os.path.basename(img_path))[0]
    img_dir = os.path.dirname(img_path)
    if os.path.basename(img_dir) == 'images':
        base_dir = os.path.dirname(img_dir)
    else:
        base_dir = img_dir
    return os.path.join(base_dir, 'gt_detr_map', stem + '.h5')

def load_data(img_path, args, train=True):
    gt_path = _gt_path_from_img(img_path)

    while True:
        try:
            gt_file = h5py.File(gt_path)
            k = np.asarray(gt_file['kpoint'])
            img = np.asarray(gt_file['image'])
            img = Image.fromarray(img, mode='RGB')
            break
        except OSError:
            #print("path is wrong", gt_path)
            cv2.waitKey(1000)  # Wait a bit
    img = img.copy()
    k = k.copy()

    return img, k


def load_data_test(img_path, args, train=True):

    img = Image.open(img_path).convert('RGB')

    return img

