import os
import sys
import argparse
import torch
import numpy as np

from src.crowd_count import CrowdCounter
from src import network
from src.data_loader import ImageDataLoader
from src.timer import Timer
from src import utils
from src.evaluate_model import evaluate_model

try:
    from termcolor import cprint
except ImportError:
    cprint = None


def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)


# ---------------------------------------------------------------------------
# Dataset path resolution
# ---------------------------------------------------------------------------

DATASET_SPLITS = {
    # dataset_name: (train_img_subdir, train_gt_subdir, val_img_subdir, val_gt_subdir)
    'shanghaiA': ('train_data/images', 'train_data/gt_density_map',
                  'test_data/images',  'test_data/gt_density_map'),
    'shanghaiB': ('train_data/images', 'train_data/gt_density_map',
                  'test_data/images',  'test_data/gt_density_map'),
}


def main():
    parser = argparse.ArgumentParser(description='Train MCNN for crowd counting')
    parser.add_argument('--dataset',    required=True,
                        choices=list(DATASET_SPLITS.keys()),
                        help='Dataset identifier')
    parser.add_argument('--data-dir',   required=True,
                        help='Root directory of the dataset')
    parser.add_argument('--output-dir', default='../logs/mcnn_sha_ckpts',
                        help='Directory to save model checkpoints')
    parser.add_argument('--epochs',     type=int, default=200)
    parser.add_argument('--lr',         type=float, default=1e-5)
    parser.add_argument('--gpu',        type=int, default=0)
    parser.add_argument('--pre-load',   action='store_true',
                        help='Pre-load all images into RAM')
    parser.add_argument('--resume',     type=str, default=None,
                        help='Path to an h5 checkpoint to resume from (weights only)')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Epoch to start counting from (use with --resume)')
    parser.add_argument('--best-mae',   type=float, default=None,
                        help='Known best MAE so far (use with --resume)')
    parser.add_argument('--patience',   type=int, default=50,
                        help='Stop if val MAE does not improve for this many epochs (default: 50)')
    parser.add_argument('--backbone',   type=str, default='mcnn',
                        help='mcnn (default) or any torchvision classification model name')
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)

    data_dir = args.data_dir
    tr_img, tr_gt, val_img, val_gt = DATASET_SPLITS[args.dataset]
    train_path    = os.path.join(data_dir, tr_img)
    train_gt_path = os.path.join(data_dir, tr_gt)
    val_path      = os.path.join(data_dir, val_img)
    val_gt_path   = os.path.join(data_dir, val_gt)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    rand_seed = 64678
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)

    net = CrowdCounter(backbone=args.backbone)
    network.weights_normal_init(net, dev=0.01)
    net.cuda()
    net.train()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr
    )

    data_loader     = ImageDataLoader(train_path, train_gt_path,
                                      shuffle=True,  gt_downsample=True,
                                      pre_load=args.pre_load)
    data_loader_val = ImageDataLoader(val_path, val_gt_path,
                                      shuffle=False, gt_downsample=True,
                                      pre_load=False)

    best_mae = sys.maxsize
    best_mse = 0.0
    best_model = ''
    disp_interval = 500

    if args.resume:
        print(f'Resuming from {args.resume}')
        network.load_net(args.resume, net)
        if args.start_epoch:
            print(f'  start_epoch={args.start_epoch}')
        if args.best_mae is not None:
            best_mae = args.best_mae
            print(f'  best_mae={best_mae}')

    epochs_no_improve = 0
    t = Timer()
    t.tic()

    for epoch in range(args.start_epoch, args.epochs + 1):
        step = -1
        train_loss = 0.0
        re_cnt = False

        for blob in data_loader:
            step += 1
            im_data  = blob['data']
            gt_data  = blob['gt_density']
            density_map = net(im_data, gt_data)
            loss = net.loss
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % disp_interval == 0:
                duration = t.toc(average=False)
                fps = (step + 1) / max(duration, 1e-6)
                gt_count  = np.sum(gt_data)
                et_count  = np.sum(density_map.data.cpu().numpy())
                log_text = ('epoch: %4d, step %4d, Time: %.4fs, '
                            'gt_cnt: %4.1f, et_cnt: %4.1f'
                            % (epoch, step, 1. / fps, gt_count, et_count))
                log_print(log_text, color='green', attrs=['bold'])
                re_cnt = True

            if re_cnt:
                t.tic()
                re_cnt = False

        if epoch % 2 == 0:
            save_name = os.path.join(output_dir, 'current.h5')
            network.save_net(save_name, net)
            mae, mse = evaluate_model(save_name, data_loader_val)
            is_best = mae < best_mae
            if is_best:
                best_mae   = mae
                best_mse   = mse
                best_model = 'best_model.h5'
                network.save_net(os.path.join(output_dir, 'best_model.h5'), net)
            log_print(f'EPOCH: {epoch}, MAE: {mae:.1f}, MSE: {mse:.1f}',
                      color='green', attrs=['bold'])
            log_print(f'BEST MAE: {best_mae:.1f}, BEST MSE: {best_mse:.1f}, '
                      f'BEST MODEL: {best_model}',
                      color='green', attrs=['bold'])
            print(f'VAL epoch={epoch} mae={mae:.2f} mse={mse:.2f} best_mae={best_mae:.2f}')

            if is_best:
                epochs_no_improve = 0
            else:
                epochs_no_improve += 2  # eval runs every 2 epochs
            if epochs_no_improve >= args.patience:
                print(f'Early stopping: val MAE has not improved for {args.patience} epochs. '
                      f'Best MAE: {best_mae:.1f}')
                break


if __name__ == '__main__':
    main()
