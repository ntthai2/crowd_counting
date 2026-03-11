"""Minimal mmcv.runner shim providing BaseModule."""
import torch.nn as nn


class BaseModule(nn.Module):
    """Minimal BaseModule shim — just nn.Module with optional init_cfg."""

    def __init__(self, init_cfg=None):
        super().__init__()
        self.init_cfg = init_cfg
