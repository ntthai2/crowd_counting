"""
Video inference pipeline for crowd counting.
Supports 4 models: regressor (EfficientNet-B0), csrnet, p2pnet, yolo

Usage:
    python video_inference.py --model all --video videos/my_video.mp4 --trained_on sha
    python video_inference.py --model all --video videos/my_video.mp4 --trained_on shb
    python video_inference.py --model all --video videos/my_video.mp4 --trained_on sha \
        --output_csv results/sha.csv --save_video results/sha_out.mp4

--trained_on controls which checkpoint set to load (sha or shb).
YOLO has no sha/shb distinction — uses the same checkpoint always.

Output:
    - Per-frame count CSV
    - Annotated video (optional, --save_video)
    - Summary stats printed to stdout
"""

import os
import sys
import csv
import argparse
import time

import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# ── Checkpoint paths ───────────────────────────────────────────────────────────
DEFAULT_CKPTS = {
    "regressor": {
        "sha": "logs/b0_sha_ckpts/model_best.pth",
        "shb": "logs/b0_shb_ckpts/model_best.pth",
    },
    "csrnet": {
        "sha": "logs/csrnet_sha_ckpts/model_best.pth.tar",
        "shb": "logs/csrnet_shb_ckpts/model_best.pth.tar",
    },
    "p2pnet": {
        "sha": "logs/p2pnet_sha_ckpts/best_mae.pth",
        "shb": "logs/p2pnet_shb_ckpts/best_mae.pth",
    },
    "yolo": {
        "sha": "runs/head_detection/train/weights/best.pt",
        "shb": "runs/head_detection/train/weights/best.pt",
    },
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Model loaders ──────────────────────────────────────────────────────────────

def load_regressor(trained_on="sha", ckpt_path=None, device="cuda"):
    sys.path.insert(0, os.getcwd())
    from train_regressor import build_model
    model = build_model("efficientnet_b0").to(device)
    path = ckpt_path or DEFAULT_CKPTS["regressor"][trained_on]
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("state_dict", ckpt.get("model", ckpt))
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def load_csrnet(trained_on="sha", ckpt_path=None, device="cuda"):
    sys.path.insert(0, os.path.join(os.getcwd(), "CSRNet"))
    sys.path.insert(0, os.getcwd())
    from CSRNet.model import CSRNet
    model = CSRNet(backbone="vgg16").to(device)
    path = ckpt_path or DEFAULT_CKPTS["csrnet"][trained_on]
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state)
    model.eval()
    return model


def load_p2pnet(trained_on="sha", ckpt_path=None, device="cuda"):
    sys.path.insert(0, os.path.join(os.getcwd(), "P2PNet"))
    sys.path.insert(0, os.getcwd())
    import argparse as _ap
    from P2PNet.models import build_model as p2p_build
    fake_args = _ap.Namespace(backbone="vgg16_bn", row=2, line=2, frozen_weights=None)
    model = p2p_build(fake_args).to(device)
    path = ckpt_path or DEFAULT_CKPTS["p2pnet"][trained_on]
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    missing, _ = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [p2pnet] {len(missing)} missing keys (architecture mismatch — counts may be unreliable)")
    model.eval()
    return model


def load_yolo(trained_on="sha", ckpt_path=None):
    from ultralytics import YOLO
    path = ckpt_path or DEFAULT_CKPTS["yolo"][trained_on]
    return YOLO(path)


# ── Inference functions (per frame) ───────────────────────────────────────────

def infer_regressor(model, frame_bgr, device="cuda"):
    preprocess = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    img = preprocess(Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))
    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        count = model(img).item()
    return max(0.0, count)


def infer_csrnet(model, frame_bgr, device="cuda"):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    img = preprocess(Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))
    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        density = model(img)
    return max(0.0, float(density.sum().item()))


def infer_p2pnet(model, frame_bgr, device="cuda", threshold=0.5):
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    w, h = pil.size
    pil = pil.resize((w // 128 * 128, h // 128 * 128), Image.BILINEAR)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    img = preprocess(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
    scores = torch.nn.functional.softmax(outputs["pred_logits"], -1)[0, :, 1]
    return float(int((scores > threshold).sum()))


def infer_yolo(model, frame_bgr, conf=0.25, iou=0.5, imgsz=640, max_det=5000):
    result = model.predict(
        source=frame_bgr, imgsz=imgsz, conf=conf, iou=iou,
        max_det=max_det, classes=[0], verbose=False
    )[0]
    return float(int(result.boxes.xyxy.shape[0]) if result.boxes is not None else 0)


# ── Video overlay ──────────────────────────────────────────────────────────────

def draw_overlay(frame, counts_dict, frame_idx, fps, trained_on):
    overlay = frame.copy()
    pad, line_h = 14, 28
    box_h = pad * 2 + line_h * (len(counts_dict) + 2)
    cv2.rectangle(overlay, (10, 10), (340, 10 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    ts = f"frame {frame_idx}  |  {frame_idx/fps:.1f}s  |  trained: {trained_on.upper()}"
    cv2.putText(frame, ts, (20, 10 + pad + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    for i, (name, cnt) in enumerate(counts_dict.items()):
        cv2.putText(frame, f"{name}: {cnt:.1f}", (20, 10 + pad + line_h * (i + 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    trained_on = args.trained_on
    print(f"Device: {device}  |  trained_on: {trained_on.upper()}")

    model_list = ["regressor", "csrnet", "p2pnet", "yolo"] if args.model == "all" else [args.model]

    models = {}
    infer_fns = {}

    for m in model_list:
        print(f"Loading {m} ({trained_on})...")
        try:
            if m == "regressor":
                models[m] = load_regressor(trained_on, args.ckpt, device)
                infer_fns[m] = lambda f, mdl=models[m]: infer_regressor(mdl, f, device)
            elif m == "csrnet":
                models[m] = load_csrnet(trained_on, args.ckpt, device)
                infer_fns[m] = lambda f, mdl=models[m]: infer_csrnet(mdl, f, device)
            elif m == "p2pnet":
                models[m] = load_p2pnet(trained_on, args.ckpt, device)
                infer_fns[m] = lambda f, mdl=models[m]: infer_p2pnet(mdl, f, device, args.p2p_threshold)
            elif m == "yolo":
                models[m] = load_yolo(trained_on, args.ckpt)
                infer_fns[m] = lambda f, mdl=models[m]: infer_yolo(mdl, f, args.yolo_conf, args.yolo_iou, args.yolo_imgsz)
        except Exception as e:
            print(f"  WARNING: failed to load {m}: {e}")

    if not infer_fns:
        print("No models loaded. Exiting.")
        return

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Cannot open video: {args.video}")
        return

    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_orig  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {args.video}  |  {w_orig}x{h_orig}  {fps:.1f}fps  {total} frames")

    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_video, fourcc, fps, (w_orig, h_orig))

    os.makedirs(os.path.dirname(args.output_csv) if os.path.dirname(args.output_csv) else ".", exist_ok=True)
    csv_file = open(args.output_csv, "w", newline="")
    fieldnames = ["frame", "timestamp_s"] + list(infer_fns.keys())
    writer_csv = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer_csv.writeheader()

    frame_idx = 0
    all_counts = {m: [] for m in infer_fns}
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % args.every_n != 0:
            frame_idx += 1
            continue

        counts = {}
        for m, fn in infer_fns.items():
            try:
                counts[m] = fn(frame)
                all_counts[m].append(counts[m])
            except Exception as e:
                counts[m] = -1.0
                print(f"  frame {frame_idx} {m} error: {e}")

        row = {"frame": frame_idx, "timestamp_s": round(frame_idx / fps, 3)}
        row.update(counts)
        writer_csv.writerow(row)

        if writer:
            annotated = draw_overlay(frame.copy(), counts, frame_idx, fps, trained_on)
            writer.write(annotated)

        if frame_idx % 50 == 0:
            elapsed = time.time() - t0
            print(f"  frame {frame_idx}/{total}  {elapsed:.1f}s  counts={counts}")

        frame_idx += 1

    cap.release()
    csv_file.close()
    if writer:
        writer.release()

    print("\n── Summary ─────────────────────────────────────────────")
    print(f"{'Model':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 56)
    for m, cnts in all_counts.items():
        if cnts:
            a = np.array(cnts)
            print(f"{m:<20} {a.mean():>8.1f} {a.std():>8.1f} {a.min():>8.1f} {a.max():>8.1f}")
    print(f"\nCSV saved to: {args.output_csv}")
    if args.save_video:
        print(f"Annotated video saved to: {args.save_video}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Crowd counting video inference pipeline")
    p.add_argument("--model", default="all",
                   choices=["regressor", "csrnet", "p2pnet", "yolo", "all"],
                   help="Model to run (default: all)")
    p.add_argument("--trained_on", default="sha", choices=["sha", "shb"],
                   help="Which dataset the models were trained on (sha or shb)")
    p.add_argument("--video", required=True,
                   help="Path to input video file")
    p.add_argument("--ckpt", default=None,
                   help="Override checkpoint path for all models")
    p.add_argument("--output_csv", default="results/counts.csv",
                   help="Output CSV path for per-frame counts")
    p.add_argument("--save_video", default=None,
                   help="Save annotated output video to this path")
    p.add_argument("--every_n", type=int, default=1,
                   help="Process every N-th frame (1=all, 5=every 5th)")
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU inference")
    p.add_argument("--p2p_threshold", type=float, default=0.5)
    p.add_argument("--yolo_conf", type=float, default=0.25)
    p.add_argument("--yolo_iou",  type=float, default=0.5)
    p.add_argument("--yolo_imgsz", type=int, default=640)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)