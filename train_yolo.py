"""
YOLO training script for head detection across merged datasets.

Datasets (excluding ShanghaiTech):
  - CCTV, JHU-CROWD++, Data_S-HEAD, SCUT-HEAD/PartA, SCUT-HEAD/PartB
  Total: train=7417, val=1728

Usage examples:
  # Train with defaults (yolo11m, 100 epochs, batch=16, GPU 0)
  python train_yolo.py

  # Override model and batch size
  python train_yolo.py --model yolo11l.pt --batch 8 --epochs 150

  # Multi-GPU (e.g., GPU 0 and 1)
  python train_yolo.py --device 0,1 --batch 32

  # Resume a previous run
  python train_yolo.py --resume runs/yolo_head_detection/train/weights/last.pt
"""

import argparse
import os
from pathlib import Path
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
DATA_YAML = ROOT / "data" / "merged_data.yaml"
RUNS_DIR = ROOT / "runs" / "head_detection"


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO on merged head-detection datasets")
    parser.add_argument("--model",   type=str,   default="yolo11m.pt",
                        help="Model checkpoint to start from (default: yolo11m.pt)")
    parser.add_argument("--epochs",  type=int,   default=100,
                        help="Number of training epochs (default: 100)")
    parser.add_argument("--batch",   type=int,   default=16,
                        help="Batch size per GPU (default: 16)")
    parser.add_argument("--imgsz",   type=int,   default=640,
                        help="Input image size (default: 640)")
    parser.add_argument("--device",  type=str,   default="0",
                        help="CUDA device(s), e.g. '0' or '0,1' (default: 0)")
    parser.add_argument("--workers", type=int,   default=8,
                        help="DataLoader workers (default: 8)")
    parser.add_argument("--patience",type=int,   default=20,
                        help="Early-stopping patience in epochs (default: 20)")
    parser.add_argument("--resume",  type=str,   default=None,
                        help="Path to last.pt to resume training from")
    parser.add_argument("--name",    type=str,   default="train",
                        help="Run name under the project directory (default: train)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.resume:
        print(f"[INFO] Resuming from checkpoint: {args.resume}")
        model = YOLO(args.resume)
        model.train(resume=True)
        return

    print(f"[INFO] Loading model: {args.model}")
    model = YOLO(args.model)

    print(f"[INFO] Data config: {DATA_YAML}")
    print(f"[INFO] Epochs={args.epochs}, Batch={args.batch}, Imgsz={args.imgsz}, Device={args.device}")

    results = model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=str(RUNS_DIR),
        name=args.name,
        # Augmentation
        mosaic=1.0,
        mixup=0.1,
        flipud=0.0,
        fliplr=0.5,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        # Logging
        plots=True,
        val=True,
        save=True,
        save_period=-1,     # only save best & last
        exist_ok=False,
    )

    print("\n[DONE] Training complete.")
    print(f"  Best weights : {RUNS_DIR / args.name / 'weights' / 'best.pt'}")
    print(f"  Last weights : {RUNS_DIR / args.name / 'weights' / 'last.pt'}")


if __name__ == "__main__":
    main()
