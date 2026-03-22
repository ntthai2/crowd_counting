import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from torch.nn import functional as F
from torchvision import models as tv_models

__all__ = ['vgg19']
model_urls = {
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = F.upsample_bilinear(x, scale_factor=2)
        x = self.reg_layer(x)
        return torch.abs(x)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}

def vgg19():
    """VGG 19-layer model (configuration "E")
        model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E']))
    model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    return model


class GenericDensityModel(nn.Module):
    def __init__(self, backbone_name: str):
        super().__init__()
        if backbone_name not in tv_models.list_models(module=tv_models):
            raise ValueError(f"Unsupported torchvision backbone: {backbone_name}")
        self.backbone = tv_models.get_model(backbone_name, weights="DEFAULT")
        self.backbone.eval()
        out_ch = self._infer_last_channels()
        self.reg_layer = nn.Sequential(
            nn.Conv2d(out_ch, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )

    def _infer_last_channels(self) -> int:
        x = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            feat = self._extract_last_4d(x)
        return int(feat.shape[1])

    def _extract_last_4d(self, x: torch.Tensor) -> torch.Tensor:
        last = None
        handles = []

        def _hook(_m, _i, o):
            nonlocal last
            if isinstance(o, torch.Tensor) and o.dim() == 4 and o.shape[2] >= 2 and o.shape[3] >= 2:
                last = o

        for _, module in self.backbone.named_modules():
            handles.append(module.register_forward_hook(_hook))
        _ = self.backbone(x)
        for h in handles:
            h.remove()
        if last is None:
            raise ValueError("Could not extract 4D feature map from torchvision backbone")
        return last

    def forward(self, x):
        feat = self._extract_last_4d(x)
        feat = F.upsample_bilinear(feat, scale_factor=2)
        out = self.reg_layer(feat)
        return torch.abs(out)


def build_model(backbone_name: str = 'vgg19'):
    if backbone_name == 'vgg19':
        return vgg19()
    return GenericDensityModel(backbone_name)

