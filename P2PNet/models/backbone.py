# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Backbone modules."""

import torch
from torch import nn
from torchvision import models as tv_models

import models.vgg_ as models


def _select_multiscale_features(backbone: nn.Module):
    backbone.eval()
    named_modules = list(backbone.named_modules())
    idx_map = {n: i for i, (n, _) in enumerate(named_modules)}

    captures = []
    handles = []

    def make_hook(name):
        def _hook(_m, _i, o):
            if not isinstance(o, torch.Tensor) or o.dim() != 4:
                return
            if int(o.shape[2]) < 2 or int(o.shape[3]) < 2:
                return
            captures.append((idx_map[name], name, int(o.shape[1]), int(o.shape[2]), int(o.shape[3])))

        return _hook

    for name, module in named_modules:
        if not name:
            continue
        handles.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        _ = backbone(torch.zeros(1, 3, 224, 224))

    for h in handles:
        h.remove()

    if not captures:
        raise ValueError("Unable to collect feature maps from backbone")

    by_res = {}
    for item in captures:
        key = (item[3], item[4])
        if key in by_res:
            # choose the deepest (largest channel) feature for each resolution
            if item[2] > by_res[key][2]:
                by_res[key] = item
        else:
            by_res[key] = item

    feats = list(by_res.values())
    feats.sort(key=lambda t: t[3] * t[4], reverse=True)
    if len(feats) < 4:
        raise ValueError("Backbone must expose at least 4 distinct 2D feature scales")

    feats = feats[-4:]
    feats.sort(key=lambda t: t[3] * t[4], reverse=True)
    return [(name, ch) for _, name, ch, _, _ in feats]


class BackboneBase_VGG(nn.Module):
    def __init__(self, backbone: nn.Module, num_channels: int, name: str, return_interm_layers: bool):
        super().__init__()
        features = list(backbone.features.children())
        if return_interm_layers:
            if name == 'vgg16_bn':
                self.body1 = nn.Sequential(*features[:13])
                self.body2 = nn.Sequential(*features[13:23])
                self.body3 = nn.Sequential(*features[23:33])
                self.body4 = nn.Sequential(*features[33:43])
            else:
                self.body1 = nn.Sequential(*features[:9])
                self.body2 = nn.Sequential(*features[9:16])
                self.body3 = nn.Sequential(*features[16:23])
                self.body4 = nn.Sequential(*features[23:30])
        else:
            if name == 'vgg16_bn':
                self.body = nn.Sequential(*features[:44])
            elif name == 'vgg16':
                self.body = nn.Sequential(*features[:30])
        self.num_channels = num_channels
        self.return_interm_layers = return_interm_layers
        self.out_channels = [128, 256, 512, 512]

    def forward(self, tensor_list):
        out = []
        if self.return_interm_layers:
            xs = tensor_list
            for _, layer in enumerate([self.body1, self.body2, self.body3, self.body4]):
                xs = layer(xs)
                out.append(xs)
        else:
            xs = self.body(tensor_list)
            out.append(xs)
        return out


class Backbone_VGG(BackboneBase_VGG):
    def __init__(self, name: str, return_interm_layers: bool):
        if name == 'vgg16_bn':
            backbone = models.vgg16_bn(pretrained=True)
        elif name == 'vgg16':
            backbone = models.vgg16(pretrained=True)
        else:
            raise ValueError(f"Unsupported VGG backbone: {name}")
        num_channels = 256
        super().__init__(backbone, num_channels, name, return_interm_layers)


class Backbone_Torchvision(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        if name not in tv_models.list_models(module=tv_models):
            raise ValueError(f"Unsupported torchvision backbone: {name}")
        self.backbone = tv_models.get_model(name, weights="DEFAULT")
        selected = _select_multiscale_features(self.backbone)
        self.node_names = [n for n, _ in selected]
        self.out_channels = [c for _, c in selected]
        self._mods = dict(self.backbone.named_modules())

    def forward(self, x):
        outputs = {}
        handles = []

        def mk(name):
            def _h(_m, _i, o):
                if isinstance(o, torch.Tensor) and o.dim() == 4:
                    outputs[name] = o

            return _h

        for n in self.node_names:
            handles.append(self._mods[n].register_forward_hook(mk(n)))
        _ = self.backbone(x)
        for h in handles:
            h.remove()
        return [outputs[n] for n in self.node_names]


def build_backbone(args):
    name = args.backbone
    if name in {'vgg16', 'vgg16_bn'}:
        return Backbone_VGG(name, True)
    return Backbone_Torchvision(name)


if __name__ == '__main__':
    Backbone_VGG('vgg16', True)