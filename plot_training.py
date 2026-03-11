#!/usr/bin/env python3
"""
Parse VAL log lines from training logs and plot MAE/MSE over epochs.

Usage:
    python plot_training.py
    python plot_training.py --models mcnn_sha

Expected log format (one line per validation epoch):
    VAL epoch=XXX mae=XX.XX mse=XX.XX best_mae=XX.XX
"""

import argparse
import os
import re
import sys
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Map log file stem -> display name
LOG_FILES = {
    'mcnn_sha':          'MCNN (SHA)',
    'mcnn_shb':          'MCNN (SHB)',
    'csrnet_sha':        'CSRNet (SHA)',
    'csrnet_shb':        'CSRNet (SHB)',
    'bl_sha':            'Bayesian-Loss (SHA)',
    'bl_shb':            'Bayesian-Loss (SHB)',
    'dmcount_sha':       'DM-Count (SHA)',
    'dmcount_shb':       'DM-Count (SHB)',
    'p2pnet_sha':        'P2PNet (SHA)',
    'p2pnet_shb':        'P2PNet (SHB)',
    'vgg16_sha':         'VGG16+FC (SHA)',
    'vgg16_shb':         'VGG16+FC (SHB)',
    'resnet50_sha':      'ResNet50+FC (SHA)',
    'resnet50_shb':      'ResNet50+FC (SHB)',
}

VAL_PATTERN = re.compile(
    r'VAL epoch=(\d+)\s+mae=([\d.]+)\s+mse=([\d.]+)\s+best_mae=([\d.]+)'
)


def parse_log(path):
    """Return (epochs, maes, mses, best_maes) lists from a log file."""
    epochs, maes, mses, bests = [], [], [], []
    try:
        with open(path, 'r', errors='replace') as f:
            for line in f:
                m = VAL_PATTERN.search(line)
                if m:
                    epochs.append(int(m.group(1)))
                    maes.append(float(m.group(2)))
                    mses.append(float(m.group(3)))
                    bests.append(float(m.group(4)))
    except FileNotFoundError:
        pass
    return epochs, maes, mses, bests


def main():

    parser = argparse.ArgumentParser(description='Plot MAE/MSE training curves from log files')
    parser.add_argument('--log-dir', default='logs/',
                        help='directory containing .log files (default: logs/)')
    parser.add_argument('--output', default=None,
                        help='output image path (auto-named if not specified)')
    parser.add_argument('--models', nargs='+', default=None,
                        help='subset of model keys to plot (e.g. csrnet mcnn vgg16_unified)')
    args = parser.parse_args()

    # Determine which models to plot
    keys = list(LOG_FILES.keys())
    if args.models:
        keys = [k for k in args.models if k in LOG_FILES]
        unknown = [k for k in args.models if k not in LOG_FILES]
        if unknown:
            print(f'WARNING: unknown model keys ignored: {unknown}')
            print(f'Valid keys: {list(LOG_FILES.keys())}')

    # Collect data
    data = {}
    for key in keys:
        log_path = os.path.join(args.log_dir, f'{key}.log')
        epochs, maes, mses, bests = parse_log(log_path)
        if epochs:
            data[key] = (epochs, maes, mses, bests)
            print(f'{key}: {len(epochs)} val points, best MAE={min(maes):.2f} at ep={epochs[maes.index(min(maes))]}')
        else:
            print(f'{key}: no VAL lines found in {log_path}')

    if not data:
        print('No data to plot. Check --log-dir and that training has produced VAL log lines.')
        sys.exit(0)

    # Determine output filename
    n_models = len(data)
    if args.output:
        output_path = args.output
    else:
        if args.models is None:
            output_path = 'plots/all.png'
        elif n_models == 1:
            output_path = f"plots/{list(data.keys())[0]}.png"
        elif n_models == len(LOG_FILES):
            output_path = 'plots/all.png'
        else:
            output_path = f"plots/{n_models}.png"

    # Layout: 2 columns (MAE | MSE) per model
    n_cols = 2
    n_rows = n_models
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), squeeze=False)
    fig.suptitle('Training Curves — MAE & MSE per Epoch', fontsize=14, fontweight='bold')

    for row, (key, (epochs, maes, mses, bests)) in enumerate(data.items()):
        label = LOG_FILES.get(key, key)
        best_mae = min(maes)
        best_ep = epochs[maes.index(best_mae)]

        # MAE subplot
        ax_mae = axes[row][0]
        ax_mae.plot(epochs, maes, color='steelblue', linewidth=1.5, label='MAE')
        ax_mae.axhline(best_mae, color='steelblue', linestyle='--', linewidth=1,
                       label=f'best MAE={best_mae:.2f} @ep{best_ep}')
        ax_mae.set_title(f'{label} — MAE', fontsize=10)
        ax_mae.set_xlabel('Epoch')
        ax_mae.set_ylabel('MAE')
        ax_mae.legend(fontsize=8)
        ax_mae.grid(True, alpha=0.3)

        # MSE subplot
        ax_mse = axes[row][1]
        ax_mse.plot(epochs, mses, color='tomato', linewidth=1.5, label='MSE')
        best_mse = mses[maes.index(best_mae)]
        ax_mse.axhline(best_mse, color='tomato', linestyle='--', linewidth=1,
                       label=f'MSE@best_MAE={best_mse:.2f}')
        ax_mse.set_title(f'{label} — MSE', fontsize=10)
        ax_mse.set_xlabel('Epoch')
        ax_mse.set_ylabel('MSE')
        ax_mse.legend(fontsize=8)
        ax_mse.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved plot to {output_path}')


if __name__ == '__main__':
    main()
