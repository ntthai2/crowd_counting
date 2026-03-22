# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models

# VGG backbone
class Base_VGG(nn.Module):
    def __init__(self, name: str, last_pool=False , num_channels=256, **kwargs):
        super().__init__()
        print("### VGG16: last_pool=", last_pool)
        # loading backbone features
        from .backbones import vgg as models
        if name == 'vgg16_bn':
            backbone = models.vgg16_bn(pretrained=True)
        elif name == 'vgg16':
            backbone = models.vgg16(pretrained=True)
        
        features = list(backbone.features.children())

        # setting base module.
        if name == 'vgg16_bn':
            self.body1 = nn.Sequential(*features[:13])
            self.body2 = nn.Sequential(*features[13:23])
            self.body3 = nn.Sequential(*features[23:33])
            if last_pool:
                self.body4 = nn.Sequential(*features[33:44])  # 32x down-sample
            else:
                self.body4 = nn.Sequential(*features[33:43])  # 16x down-sample
        else:
            self.body1 = nn.Sequential(*features[:9])
            self.body2 = nn.Sequential(*features[9:16])
            self.body3 = nn.Sequential(*features[16:23])
            if last_pool:
                self.body4 = nn.Sequential(*features[23:31])  # 32x down-sample
            else:
                self.body4 = nn.Sequential(*features[23:30])  # 16x down-sample
        self.num_channels = num_channels
        self.last_pool = last_pool
        
    def get_outplanes(self):
        outplanes = []
        for i in range(4):
            last_dims = 0
            for param_tensor in self.__getattr__('body'+str(i+1)).state_dict():
                if 'weight' in param_tensor:
                    last_dims = list(self.__getattr__('body'+str(i+1)).state_dict()[param_tensor].size())[0]
            outplanes.append(last_dims)
        return outplanes   # get the last layer params of all modules, and trans to the size.

    def forward(self, tensor_list):
        out = []
        xs = tensor_list
        for _, layer in enumerate([self.body1, self.body2, self.body3, self.body4]):
            xs = layer(xs)
            out.append(xs)
        return out

# ResNet backbone
class Base_ResNet(nn.Module):
    def __init__(self, name: str, last_pool=False , num_channels=256, **kwargs):
        super().__init__()
        print("### ResNet: last_pool=", last_pool)
        # loading backbone features
        from .backbones import resnet as models
        if name == 'resnet18':
            self.backbone = models.resnet18_ibn_a(pretrained=True)
        elif name == 'resnet34':
            self.backbone = models.resnet34_ibn_a(pretrained=True)
        elif name == 'resnet50':
            self.backbone = models.resnet50_ibn_a(pretrained=True)
        elif name == 'resnet101':
            self.backbone = models.resnet101_ibn_a(pretrained=True)
        elif name == 'resnet152':
            self.backbone = models.resnet152_ibn_a(pretrained=True)     

        self.num_channels = num_channels
        self.last_pool = last_pool

    def get_outplanes(self):
        outplanes = []
        for Layer in [self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4]:
            last_dims = 0
            for param_tensor in Layer.state_dict():
                if 'weight' in param_tensor:
                    last_dims = list(Layer.state_dict()[param_tensor].size())[0]
            outplanes.append(last_dims)
        return outplanes   # get the last layer params of all modules, and trans to the size.

    def forward(self, tensor_list):
        out = []
        xs = tensor_list
        out = self.backbone(xs)
        return out


class Base_Torchvision(nn.Module):
    def __init__(self, name: str, **kwargs):
        super().__init__()
        if name.startswith('tv:'):
            name = name[3:]
        if name not in tv_models.list_models(module=tv_models):
            raise ValueError(f"Unsupported torchvision encoder: {name}")
        self.backbone = tv_models.get_model(name, weights="DEFAULT")
        self.backbone.eval()
        self.node_names, self._outplanes = self._select_nodes()
        self._modules_map = dict(self.backbone.named_modules())

    def _select_nodes(self):
        named_modules = list(self.backbone.named_modules())
        idx_map = {n: i for i, (n, _) in enumerate(named_modules)}
        captures = []
        handles = []

        def mk(name):
            def _h(_m, _i, o):
                if isinstance(o, torch.Tensor) and o.dim() == 4 and o.shape[2] >= 2 and o.shape[3] >= 2:
                    captures.append((idx_map[name], name, int(o.shape[1]), int(o.shape[2]), int(o.shape[3])))

            return _h

        for name, mod in named_modules:
            if not name:
                continue
            handles.append(mod.register_forward_hook(mk(name)))

        with torch.no_grad():
            _ = self.backbone(torch.zeros(1, 3, 224, 224))

        for h in handles:
            h.remove()

        by_res = {}
        for item in captures:
            by_res[(item[3], item[4])] = item
        feats = list(by_res.values())
        feats.sort(key=lambda t: t[3] * t[4], reverse=True)
        if len(feats) < 4:
            raise ValueError("Torchvision encoder must expose at least 4 feature resolutions")
        feats = feats[-4:]
        feats.sort(key=lambda t: t[3] * t[4], reverse=True)
        node_names = [n for _, n, _, _, _ in feats]
        outplanes = [c for _, _, c, _, _ in feats]
        return node_names, outplanes

    def get_outplanes(self):
        return list(self._outplanes)

    def forward(self, tensor_list):
        outputs = {}
        handles = []

        def mk(name):
            def _h(_m, _i, o):
                if isinstance(o, torch.Tensor) and o.dim() == 4:
                    outputs[name] = o

            return _h

        for name in self.node_names:
            handles.append(self._modules_map[name].register_forward_hook(mk(name)))
        _ = self.backbone(tensor_list)
        for h in handles:
            h.remove()
        return [outputs[name] for name in self.node_names]