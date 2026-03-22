"""
train_regressor.py
------------------
Standalone VGG16+FC and ResNet50+FC crowd-count regression trainer.

Reads ShanghaiTech data directly from the raw directory structure:
  <data_root>/train_data/images/IMG_*.jpg
  <data_root>/train_data/ground-truth/GT_IMG_*.mat
  <data_root>/test_data/images/IMG_*.jpg
  <data_root>/test_data/ground-truth/GT_IMG_*.mat

GT .mat files follow the standard ShanghaiTech format:
  scipy.io.loadmat(f)["image_info"][0,0][0,0][0]  ->  (N, 2) head coordinates

VAL log line (grep-friendly):
  VAL epoch=XXX mae=XX.XX mse=XX.XX best_mae=XX.XX

Usage
-----
PYTHON=/home/team_cam_ai/miniconda3/envs/ntt_det/bin/python
BASE=/ssd1/team_cam_ai/ntthai/crowd_counting

$PYTHON -u $BASE/train_regressor.py \
    --dataset ShanghaiB \
    --data-dir $BASE/data/ShanghaiTech/part_B \
    --save-dir $BASE/logs/vgg16_shb_ckpts \
    --model-type vgg16 \
    --epochs 1000 --lr 1e-4 --batch-size 8 --gpu 0
"""

import argparse
import math
import os
import time

import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ShanghaiTechDataset(Dataset):
    def __init__(self, root: str, split: str, transform=None):
        """
        root  : e.g. data/ShanghaiTech/part_B
        split : 'train' or 'test'
        """
        img_dir = os.path.join(root, f"{split}_data", "images")
        gt_dir  = os.path.join(root, f"{split}_data", "ground-truth")

        self.samples = []
        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            stem = os.path.splitext(fname)[0]          # IMG_100
            gt_name = f"GT_{stem}.mat"
            img_path = os.path.join(img_dir, fname)
            gt_path  = os.path.join(gt_dir, gt_name)
            if not os.path.exists(gt_path):
                continue
            self.samples.append((img_path, gt_path))

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, gt_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        mat = scipy.io.loadmat(gt_path)
        pts = mat["image_info"][0, 0][0, 0][0]   # (N, 2)
        count = torch.tensor(float(len(pts)), dtype=torch.float32)
        return img, count


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def _set_regression_head(model: nn.Module, model_name: str) -> nn.Module:
    # Common torchvision classifier interfaces.
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model

    if hasattr(model, "classifier"):
        classifier = getattr(model, "classifier")
        if isinstance(classifier, nn.Linear):
            setattr(model, "classifier", nn.Linear(classifier.in_features, 1))
            return model
        if isinstance(classifier, nn.Sequential):
            layers = list(classifier)
            for i in range(len(layers) - 1, -1, -1):
                if isinstance(layers[i], nn.Linear):
                    layers[i] = nn.Linear(layers[i].in_features, 1)
                    setattr(model, "classifier", nn.Sequential(*layers))
                    return model

    raise ValueError(
        f"Model '{model_name}' does not expose a supported classification head interface (fc/classifier)."
    )


def build_model(model_type: str) -> nn.Module:
    if model_type == "vgg16":
        backbone = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        # Replace classifier with a single linear head
        backbone.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
        )
    elif model_type == "resnet50":
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Linear(2048, 1)
    else:
        available = set(models.list_models(module=models))
        if model_type not in available:
            raise ValueError(f"Unknown model_type: {model_type}")
        backbone = models.get_model(model_type, weights="DEFAULT")
        backbone = _set_regression_head(backbone, model_type)
    return backbone


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for imgs, counts in loader:
        imgs   = imgs.to(device)
        counts = counts.to(device).unsqueeze(1)
        optimizer.zero_grad()
        preds = model(imgs)
        loss  = criterion(preds, counts)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    mae_sum, mse_sum = 0.0, 0.0
    for imgs, counts in loader:
        imgs   = imgs.to(device)
        counts = counts.to(device)
        preds  = model(imgs).squeeze(1)
        diff   = (preds - counts).abs()
        mae_sum += diff.sum().item()
        mse_sum += (diff ** 2).sum().item()
    n = len(loader.dataset)
    mae = mae_sum / n
    mse = math.sqrt(mse_sum / n)
    return mae, mse


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default="ShanghaiB",
                        choices=["ShanghaiA", "ShanghaiB"])
    parser.add_argument("--data-dir",   required=True,
                        help="Path to part_A or part_B folder")
    parser.add_argument("--save-dir",   required=True,
                        help="Where to save checkpoints")
    parser.add_argument("--model-type", default="vgg16",
                        help="vgg16/resnet50 or any torchvision classification model name")
    parser.add_argument("--epochs",     type=int,   default=1000)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int,   default=8)
    parser.add_argument("--patience",   type=int,   default=50)
    parser.add_argument("--gpu",        type=int,   default=0)
    parser.add_argument("--resume",     default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # ---- data transforms ----
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = ShanghaiTechDataset(args.data_dir, "train", train_tf)
    val_ds   = ShanghaiTechDataset(args.data_dir, "test",  val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1,
                              shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train images: {len(train_ds)} | Val images: {len(val_ds)}")

    # ---- model / optimizer / loss ----
    model = build_model(args.model_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    start_epoch = 1
    best_mae    = float("inf")
    no_improve  = 0

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 1) + 1
        best_mae    = ckpt.get("best_mae", float("inf"))
        print(f"Resumed from epoch {start_epoch - 1}, best_mae={best_mae:.4f}")

    # ---- training ----
    t_start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        train_epoch(model, train_loader, criterion, optimizer, device)
        mae, mse = evaluate(model, val_loader, device)

        is_best = mae < best_mae
        if is_best:
            best_mae   = mae
            no_improve = 0
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_mae":  best_mae,
            }, os.path.join(args.save_dir, "model_best.pth"))
        else:
            no_improve += 1

        # Always save latest checkpoint so we can resume
        torch.save({
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_mae":  best_mae,
        }, os.path.join(args.save_dir, "checkpoint.pth"))

        print(f"VAL epoch={epoch} mae={mae:.2f} mse={mse:.2f} best_mae={best_mae:.2f}",
              flush=True)

        if no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - t_start))
    print(f"Training time {elapsed}")


if __name__ == "__main__":
    main()
