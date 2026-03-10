import os
import argparse
import torch
import numpy as np

from src.crowd_count import CrowdCounter
from src import network
from src.data_loader import ImageDataLoader
from src import utils


def main():
    parser = argparse.ArgumentParser(description='Test MCNN for crowd counting')
    parser.add_argument('--model',    required=True, help='Path to .h5 model file')
    parser.add_argument('--data-dir', required=True, help='Path to test images directory')
    parser.add_argument('--gt-dir',   required=True, help='Path to test density maps directory')
    parser.add_argument('--gpu',      type=int, default=0)
    parser.add_argument('--save-output', action='store_true')
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.enabled  = True
    torch.backends.cudnn.benchmark = False

    output_dir = './output/'
    model_name = os.path.basename(args.model).split('.')[0]
    os.makedirs(output_dir, exist_ok=True)
    file_results = os.path.join(output_dir, f'results_{model_name}.txt')
    density_dir  = os.path.join(output_dir, f'density_maps_{model_name}')
    if args.save_output:
        os.makedirs(density_dir, exist_ok=True)

    net = CrowdCounter()
    network.load_net(args.model, net)
    net.cuda()
    net.eval()

    data_loader = ImageDataLoader(args.data_dir, args.gt_dir,
                                  shuffle=False, gt_downsample=True,
                                  pre_load=True)
    mae = 0.0
    mse = 0.0
    with torch.no_grad():
        for blob in data_loader:
            im_data  = blob['data']
            gt_data  = blob['gt_density']
            density_map = net(im_data, gt_data)
            density_map = density_map.data.cpu().numpy()
            gt_count  = np.sum(gt_data)
            et_count  = np.sum(density_map)
            mae += abs(gt_count - et_count)
            mse += (gt_count - et_count) ** 2
            if args.save_output:
                utils.save_density_map(
                    density_map, density_dir,
                    'output_' + blob['fname'].split('.')[0] + '.png'
                )

    n   = data_loader.get_num_samples()
    mae = mae / n
    mse = np.sqrt(mse / n)
    print(f'\nMAE: {mae:.2f}, MSE: {mse:.2f}')

    with open(file_results, 'w') as f:
        f.write(f'MAE: {mae:.2f}, MSE: {mse:.2f}\n')


if __name__ == '__main__':
    main()

f.close()