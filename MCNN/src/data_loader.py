import numpy as np
import cv2
import os
import random


class ImageDataLoader():
    def __init__(self, data_path, gt_path, shuffle=False, gt_downsample=False, pre_load=False):
        # pre_load: if True, all images are loaded into CPU RAM for faster processing.
        self.data_path = data_path
        self.gt_path = gt_path
        self.gt_downsample = gt_downsample
        self.pre_load = pre_load
        self.data_files = sorted([
            f for f in os.listdir(data_path)
            if os.path.isfile(os.path.join(data_path, f))
        ])
        self.shuffle = shuffle
        if shuffle:
            random.seed(2468)
        self.num_samples = len(self.data_files)
        self.blob_list = {}
        self.id_list = list(range(self.num_samples))
        if self.pre_load:
            print('Pre-loading the data. This may take a while...')
            for idx, fname in enumerate(self.data_files):
                blob = self._load_blob(fname)
                self.blob_list[idx] = blob
                if (idx + 1) % 100 == 0:
                    print(f'Loaded {idx + 1} / {self.num_samples} files')
            print(f'Completed loading {self.num_samples} files')

    def _load_blob(self, fname):
        img = cv2.imread(os.path.join(self.data_path, fname), 0)  # grayscale
        img = img.astype(np.float32)
        ht, wd = img.shape
        ht_1 = (ht // 4) * 4
        wd_1 = (wd // 4) * 4
        img = cv2.resize(img, (wd_1, ht_1))
        img = img.reshape((1, 1, img.shape[0], img.shape[1]))

        # Load .npy density map (stem matches image filename stem)
        stem = os.path.splitext(fname)[0]
        npy_path = os.path.join(self.gt_path, stem + '.npy')
        den = np.load(npy_path).astype(np.float32)

        if self.gt_downsample:
            wd_out = wd_1 // 4
            ht_out = ht_1 // 4
            den = cv2.resize(den, (wd_out, ht_out))
            den = den * ((wd * ht) / (wd_out * ht_out))
        else:
            den = cv2.resize(den, (wd_1, ht_1))
            den = den * ((wd * ht) / (wd_1 * ht_1))

        den = den.reshape((1, 1, den.shape[0], den.shape[1]))
        return {'data': img, 'gt_density': den, 'fname': fname}

    def __iter__(self):
        if self.shuffle:
            if self.pre_load:
                random.shuffle(self.id_list)
            else:
                random.shuffle(self.data_files)

        for idx in self.id_list:
            if self.pre_load:
                blob = self.blob_list[idx]
                blob['idx'] = idx
            else:
                blob = self._load_blob(self.data_files[idx])
            yield blob

    def get_num_samples(self):
        return self.num_samples
