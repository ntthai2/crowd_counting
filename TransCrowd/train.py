from __future__ import division
import warnings
from Networks.models import base_patch16_384_token, base_patch16_384_gap, vgg16_count, resnet50_count
import torch.nn as nn
from torchvision import transforms
import dataset
import math
from utils import save_checkpoint, setup_seed
import torch
import os
import logging
import nni
from nni.utils import merge_parameter
from config import return_args, args
import numpy as np
from image import load_data

warnings.filterwarnings('ignore')
import time

setup_seed(args.seed)

logger = logging.getLogger('mnist_AutoML')


def main(args):
    if args['dataset'] == 'ShanghaiA':
        train_file = './npydata/ShanghaiA_train.npy'
        test_file = './npydata/ShanghaiA_test.npy'
    elif args['dataset'] == 'ShanghaiB':
        train_file = './npydata/ShanghaiB_train.npy'
        test_file = './npydata/ShanghaiB_test.npy'
    else:
        raise ValueError(f"Unknown dataset: {args['dataset']}. Choose 'ShanghaiA' or 'ShanghaiB'.")

    with open(train_file, 'rb') as outfile:
        train_list = np.load(outfile, allow_pickle=True).tolist()
    with open(test_file, 'rb') as outfile:
        val_list = np.load(outfile, allow_pickle=True).tolist()

    print(len(train_list), len(val_list))

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu_id']

    if args['model_type'] == 'token':
        model = base_patch16_384_token(pretrained=True)
    elif args['model_type'] == 'gap':
        model = base_patch16_384_gap(pretrained=True)
    elif args['model_type'] == 'vgg16':
        model = vgg16_count(pretrained=True)
        print('Using VGG16+FC regression model')
    elif args['model_type'] == 'resnet50':
        model = resnet50_count(pretrained=True)
        print('Using ResNet50+FC regression model')
    else:
        raise ValueError(f"Unknown model_type: {args['model_type']}")

    model = nn.DataParallel(model, device_ids=[0])
    model = model.cuda()

    criterion = nn.L1Loss(size_average=False).cuda()

    optimizer = torch.optim.Adam(
        [  #
            {'params': model.parameters(), 'lr': args['lr']},
        ], lr=args['lr'], weight_decay=args['weight_decay'])

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300], gamma=0.1, last_epoch=-1)
    print(args['pre'])

    # args['save_path'] = args['save_path'] + str(args['rdt'])
    print(args['save_path'])
    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])

    if args['pre']:
        if os.path.isfile(args['pre']):
            print("=> loading checkpoint '{}'".format(args['pre']))
            checkpoint = torch.load(args['pre'])
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            args['start_epoch'] = checkpoint['epoch']
            args['best_pred'] = checkpoint['best_prec1']
        else:
            print("=> no checkpoint found at '{}'".format(args['pre']))

    torch.set_num_threads(args['workers'])

    print(args['best_pred'], args['start_epoch'])

    # Counts are pre-saved in a sidecar _counts.npy file (no per-image h5 lookup needed).
    count_list_train, count_list_val = None, None
    if args['dataset'] in ('ShanghaiA', 'ShanghaiB'):
        count_file_train = train_file.replace('.npy', '_counts.npy')
        count_file_val   = test_file.replace('.npy', '_counts.npy')
        if os.path.exists(count_file_train):
            with open(count_file_train, 'rb') as f:
                count_list_train = np.load(f, allow_pickle=True).tolist()
        if os.path.exists(count_file_val):
            with open(count_file_val, 'rb') as f:
                count_list_val = np.load(f, allow_pickle=True).tolist()

    train_data = pre_data(train_list, args, train=True,  count_list=count_list_train)
    test_data  = pre_data(val_list,   args, train=False, count_list=count_list_val)

    epochs_no_improve = 0

    for epoch in range(args['start_epoch'], args['epochs']):

        start = time.time()
        train(train_data, model, criterion, optimizer, epoch, args, scheduler)
        end1 = time.time()

        if epoch % 5 == 0 and epoch >= 10:
            prec1 = validate(test_data, model, args)
            end2 = time.time()
            val_mae, val_mse = prec1
            is_best = val_mae < args['best_pred']
            args['best_pred'] = min(val_mae, args['best_pred'])

            print(f"VAL epoch={epoch} mae={val_mae:.2f} mse={val_mse:.2f} best_mae={args['best_pred']:.2f}")

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args['pre'],
                'state_dict': model.state_dict(),
                'best_prec1': args['best_pred'],
                'optimizer': optimizer.state_dict(),
            }, is_best, args['save_path'])

            if is_best:
                epochs_no_improve = 0
            else:
                epochs_no_improve += 5  # val runs every 5 epochs
            if epochs_no_improve >= args['patience']:
                print(f"Early stopping: val MAE has not improved for {args['patience']} epochs. "
                      f"Best MAE: {args['best_pred']:.3f}")
                break




def pre_data(train_list, args, train, count_list=None):
    """Build a lightweight index — paths + pre-computed counts only.
    Images are NOT loaded here; loading happens lazily in __getitem__.
    """
    print("Building dataset index (%d samples) ..." % len(train_list))
    data_keys = {}
    for j, img_path in enumerate(train_list):
        blob = {
            'path':     str(img_path),
            'fname':    os.path.basename(str(img_path)),
            'gt_count': count_list[j] if count_list is not None else None,
        }
        data_keys[j] = blob
    return data_keys


def train(Pre_data, model, criterion, optimizer, epoch, args, scheduler):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(Pre_data, args['save_path'],
                            shuffle=True,
                            transform=transforms.Compose([
                                transforms.Resize((384, 384)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]),
                            ]),
                            train=True,
                            batch_size=args['batch_size'],
                            num_workers=args['workers'],
                            args=args),
        batch_size=args['batch_size'], drop_last=False)
    args['lr'] = optimizer.param_groups[0]['lr']
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args['lr']))

    model.train()
    end = time.time()

    for i, (fname, img, gt_count) in enumerate(train_loader):

        data_time.update(time.time() - end)
        img = img.cuda()

        out1 = model(img)
        gt_count = gt_count.type(torch.FloatTensor).cuda().unsqueeze(1)

        # print(out1.shape, kpoint.shape)
        loss = criterion(out1, gt_count)

        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args['print_freq'] == 0:
            print('4_Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))

    scheduler.step()


def validate(Pre_data, model, args):
    print('begin test')
    batch_size = 1
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(Pre_data, args['save_path'],
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),

                            ]),
                            args=args, train=False),
        batch_size=1)

    model.eval()

    mae = 0.0
    mse = 0.0
    visi = []
    index = 0

    for i, (fname, img, gt_count) in enumerate(test_loader):

        img = img.cuda()
        if len(img.shape) == 5:
            img = img.squeeze(0)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            out1 = model(img)
            count = torch.sum(out1).item()

        gt_count = torch.sum(gt_count).item()
        mae += abs(gt_count - count)
        mse += abs(gt_count - count) * abs(gt_count - count)

        if i % 15 == 0:
            print('{fname} Gt {gt:.2f} Pred {pred}'.format(fname=fname[0], gt=gt_count, pred=count))

    mae = mae * 1.0 / (len(test_loader) * batch_size)
    mse = math.sqrt(mse / (len(test_loader)) * batch_size)

    nni.report_intermediate_result(mae)
    print(' \n* MAE {mae:.3f}\n'.format(mae=mae), '* MSE {mse:.3f}'.format(mse=mse))

    return mae, mse


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    # Use NNI when running inside an experiment, fall back to CLI args otherwise.
    try:
        tuner_params = nni.get_next_parameter()
    except Exception:
        tuner_params = {}
    logger.debug(tuner_params)
    params = vars(merge_parameter(return_args, tuner_params))
    print(params)

    main(params)
