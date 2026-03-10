import torch
from torch.utils.data import Dataset
import os
import random
from image import *
import numpy as np
import numbers
from torchvision import datasets, transforms
import torch.nn.functional as F


class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None, train=False, seen=0, batch_size=1,
                 num_workers=4, args=None):
        if train:
            random.shuffle(root)

        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.args = args

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        blob = self.lines[index]
        fname     = blob['fname']
        img_path  = blob['path']
        gt_count_preloaded = blob.get('gt_count', None)

        # Lazy image load
        from PIL import Image as PIL_Image
        img = PIL_Image.open(img_path).convert('RGB')

        # Ground-truth count
        if gt_count_preloaded is not None:
            import numpy as _np
            gt_count = _np.array(float(gt_count_preloaded), dtype=_np.float32)
        else:
            import h5py, numpy as _np
            gt_path = img_path.replace('.jpg', '.h5').replace('images', 'gt_density_map')
            with h5py.File(gt_path, 'r') as f:
                gt_count = _np.array(f['gt_count'], dtype=_np.float32)

        '''data augmention'''
        if self.train == True:
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            # if random.random() > self.args['random_noise']:
            #     proportion = random.uniform(0.004, 0.015)
            #     width, height = img.size[0], img.size[1]
            #     num = int(height * width * proportion)
            #     for i in range(num):
            #         w = random.randint(0, width - 1)
            #         h = random.randint(0, height - 1)
            #         if random.randint(0, 1) == 0:
            #             img.putpixel((w, h), (0, 0, 0))
            #         else:
            #             img.putpixel((w, h), (255, 255, 255))

        gt_count = gt_count.copy()
        img = img.copy()

        if self.train == True:
            if self.transform is not None:
                img = self.transform(img)

            return fname, img, gt_count

        else:
            if self.transform is not None:
                img = self.transform(img)

            width, height = img.shape[2], img.shape[1]

            m = int(width / 384)
            n = int(height / 384)

            # If image is smaller than 384 in either dim, use the whole image
            if m == 0 or n == 0:
                return fname, img.cuda().unsqueeze(0), gt_count

            for i in range(0, m):
                for j in range(0, n):

                    if i == 0 and j == 0:
                        img_return = img[:, j * 384: 384 * (j + 1), i * 384:(i + 1) * 384].cuda().unsqueeze(0)
                    else:
                        crop_img = img[:, j * 384: 384 * (j + 1), i * 384:(i + 1) * 384].cuda().unsqueeze(0)

                        img_return = torch.cat([img_return, crop_img], 0).cuda()
            return fname, img_return, gt_count
