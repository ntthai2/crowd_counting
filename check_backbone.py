#!/usr/bin/env python3
"""
Quick compatibility checker for a torchvision backbone across the 7 counted models:
- regressor
- CSRNet
- Bayesian-Loss
- DM-Count
- MCNN
- P2PNet
- APGCC

This script runs lightweight smoke checks in isolated subprocesses to avoid
module-name collisions between different codebases.

/home/team_cam_ai/miniconda3/envs/ntt_det/bin/python check_backbone_compat.py --backbone resnet18
"""

import argparse
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

MODELS = [
    "regressor",
    "csrnet",
    "bl",
    "dmcount",
    "mcnn",
    "p2pnet",
    "apgcc",
]


def _snippets(backbone: str, input_size: int, device: str):
    b = json.dumps(backbone)
    d = json.dumps(device)
    s = int(input_size)

    return {
        "regressor": textwrap.dedent(
            f"""
            import torch
            from train_regressor import build_model

            device = {d}
            if device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'

            m = build_model({b}).to(device).eval()
            x = torch.randn(1, 3, {s}, {s}, device=device)
            y = m(x)
            print('OK::regressor::shape=' + str(tuple(y.shape)))
            """
        ),
        "csrnet": textwrap.dedent(
            f"""
            import sys
            import torch
            from pathlib import Path

            root = Path('.').resolve()
            sys.path.insert(0, str(root / 'CSRNet'))
            from model import CSRNet

            device = {d}
            if device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'

            m = CSRNet(backbone={b}).to(device).eval()
            x = torch.randn(1, 3, {s}, {s}, device=device)
            y = m(x)
            print('OK::csrnet::shape=' + str(tuple(y.shape)))
            """
        ),
        "bl": textwrap.dedent(
            f"""
            import sys
            import torch
            from pathlib import Path

            root = Path('.').resolve()
            sys.path.insert(0, str(root / 'Bayesian-Loss'))
            from models.vgg import build_model

            device = {d}
            if device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'

            m = build_model({b}).to(device).eval()
            x = torch.randn(1, 3, {s}, {s}, device=device)
            y = m(x)
            print('OK::bl::shape=' + str(tuple(y.shape)))
            """
        ),
        "dmcount": textwrap.dedent(
            f"""
            import sys
            import torch
            from pathlib import Path

            root = Path('.').resolve()
            sys.path.insert(0, str(root / 'DM-Count'))
            from models import build_model

            device = {d}
            if device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'

            m = build_model({b}).to(device).eval()
            x = torch.randn(1, 3, {s}, {s}, device=device)
            y_mu, y_norm = m(x)
            print('OK::dmcount::mu=' + str(tuple(y_mu.shape)) + '::norm=' + str(tuple(y_norm.shape)))
            """
        ),
        "mcnn": textwrap.dedent(
            f"""
            import sys
            import torch
            from pathlib import Path

            root = Path('.').resolve()
            sys.path.insert(0, str(root / 'MCNN'))
            from src.models import MCNN

            device = {d}
            if device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'

            m = MCNN(backbone={b}).to(device).eval()
            x = torch.randn(1, 1, {s}, {s}, device=device)
            y = m(x)
            print('OK::mcnn::shape=' + str(tuple(y.shape)))
            """
        ),
        "p2pnet": textwrap.dedent(
            f"""
            import sys
            import types
            import torch
            from pathlib import Path

            root = Path('.').resolve()
            sys.path.insert(0, str(root / 'P2PNet'))
            from models.backbone import build_backbone

            device = {d}
            if device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'

            args = types.SimpleNamespace(backbone={b})
            m = build_backbone(args).to(device).eval()
            x = torch.randn(1, 3, {s}, {s}, device=device)
            feats = m(x)
            shapes = [tuple(t.shape) for t in feats]
            print('OK::p2pnet::feats=' + str(shapes))
            """
        ),
        "apgcc": textwrap.dedent(
            f"""
            import sys
            import torch
            from pathlib import Path

            root = Path('.').resolve()
            sys.path.insert(0, str(root / 'APGCC' / 'apgcc'))
            from models.Encoder import Base_Torchvision

            device = {d}
            if device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'

            m = Base_Torchvision(name='tv:' + {b}).to(device).eval()
            x = torch.randn(1, 3, {s}, {s}, device=device)
            feats = m(x)
            shapes = [tuple(t.shape) for t in feats]
            outplanes = m.get_outplanes()
            print('OK::apgcc::feats=' + str(shapes) + '::outplanes=' + str(outplanes))
            """
        ),
    }


def run_check(root: Path, model_key: str, code: str, timeout: int):
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    ok = proc.returncode == 0
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()

    if ok:
        detail = stdout.splitlines()[-1] if stdout else "OK"
        return {"model": model_key, "status": "PASS", "detail": detail}

    err_line = stderr.splitlines()[-1] if stderr else "Unknown error"
    return {
        "model": model_key,
        "status": "FAIL",
        "detail": err_line,
        "stdout": stdout,
        "stderr": stderr,
    }


def parse_models(raw: str):
    if raw.strip().lower() == "all":
        return list(MODELS)
    items = [x.strip().lower() for x in raw.split(",") if x.strip()]
    invalid = [x for x in items if x not in MODELS]
    if invalid:
        raise ValueError(f"Unknown model keys: {invalid}. Valid keys: {MODELS}")
    return items


def main():
    parser = argparse.ArgumentParser(description="Check torchvision backbone compatibility across 7 models")
    parser.add_argument("--backbone", required=True, help="torchvision model name, e.g. resnet18")
    parser.add_argument("--models", default="all", help="comma list or 'all'")
    parser.add_argument("--input-size", type=int, default=224, help="square input size for smoke forward")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--timeout", type=int, default=600, help="per-model timeout seconds")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--json-out", default="", help="optional json output file")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    targets = parse_models(args.models)
    snippets = _snippets(args.backbone, args.input_size, args.device)

    print(f"[INFO] root={root}")
    print(f"[INFO] backbone={args.backbone}")
    print(f"[INFO] models={targets}")

    results = []
    for key in targets:
        print(f"[RUN] {key}")
        res = run_check(root, key, snippets[key], timeout=args.timeout)
        results.append(res)
        print(f"[{res['status']}] {key}: {res['detail']}")
        if args.fail_fast and res["status"] == "FAIL":
            break

    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")

    print("\n=== SUMMARY ===")
    print(f"PASS: {passed}")
    print(f"FAIL: {failed}")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "backbone": args.backbone,
            "models": targets,
            "input_size": args.input_size,
            "device": args.device,
            "results": results,
            "pass": passed,
            "fail": failed,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[INFO] wrote json report: {out_path}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
