import torch
import torch.nn as nn
from .network import Conv2d
from torchvision import models as tv_models

class MCNN(nn.Module):
    '''
    Multi-column CNN 
        -Implementation of Single Image Crowd Counting via Multi-column CNN (Zhang et al.)
    '''
    
    def __init__(self, bn=False, backbone='mcnn'):
        super(MCNN, self).__init__()
        self.backbone_name = backbone
        
        self.branch1 = nn.Sequential(Conv2d( 1, 16, 9, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(16, 32, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(32, 16, 7, same_padding=True, bn=bn),
                                     Conv2d(16,  8, 7, same_padding=True, bn=bn))
        
        self.branch2 = nn.Sequential(Conv2d( 1, 20, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(20, 40, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(40, 20, 5, same_padding=True, bn=bn),
                                     Conv2d(20, 10, 5, same_padding=True, bn=bn))
        
        self.branch3 = nn.Sequential(Conv2d( 1, 24, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(24, 48, 3, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(48, 24, 3, same_padding=True, bn=bn),
                                     Conv2d(24, 12, 3, same_padding=True, bn=bn))
        
        self.fuse = nn.Sequential(Conv2d( 30, 1, 1, same_padding=True, bn=bn))

        self.tv_backbone = None
        self.tv_head = None
        if backbone != 'mcnn':
            if backbone not in tv_models.list_models(module=tv_models):
                raise ValueError(f"Unsupported torchvision backbone: {backbone}")
            self.tv_backbone = tv_models.get_model(backbone, weights="DEFAULT")
            out_ch = self._infer_last_channels()
            self.tv_head = nn.Sequential(
                nn.Conv2d(out_ch, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, 1),
            )

    def _extract_last_4d(self, x):
        last = None
        handles = []

        def _hook(_m, _i, o):
            nonlocal last
            if isinstance(o, torch.Tensor) and o.dim() == 4 and o.shape[2] >= 2 and o.shape[3] >= 2:
                last = o

        for _, module in self.tv_backbone.named_modules():
            handles.append(module.register_forward_hook(_hook))
        _ = self.tv_backbone(x)
        for h in handles:
            h.remove()
        if last is None:
            raise ValueError("Could not extract 4D feature map from torchvision backbone")
        return last

    def _infer_last_channels(self):
        with torch.no_grad():
            feat = self._extract_last_4d(torch.zeros(1, 3, 224, 224))
        return int(feat.shape[1])
        
    def forward(self, im_data):
        if self.tv_backbone is not None:
            if im_data.shape[1] == 1:
                im_data = im_data.repeat(1, 3, 1, 1)
            feat = self._extract_last_4d(im_data)
            x = self.tv_head(feat)
            out_h = max(1, im_data.shape[2] // 4)
            out_w = max(1, im_data.shape[3] // 4)
            x = torch.nn.functional.interpolate(x, size=(out_h, out_w), mode='bilinear', align_corners=False)
            return x

        x1 = self.branch1(im_data)
        x2 = self.branch2(im_data)
        x3 = self.branch3(im_data)
        x = torch.cat((x1,x2,x3),1)
        x = self.fuse(x)
        
        return x