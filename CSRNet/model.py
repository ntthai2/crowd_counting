import torch.nn as nn
import torch
from torchvision import models
from utils import save_net,load_net

class CSRNet(nn.Module):
    def __init__(self, load_weights=False, backbone='vgg16'):
        super(CSRNet, self).__init__()
        self.seen = 0
        self.backbone_name = backbone
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512,256,128,64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat,in_channels = 512,dilation = True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.alt_backbone = None
        self.alt_proj = None

        if backbone == 'vgg16' and not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            frontend_items = list(self.frontend.state_dict().items())
            mod_items = list(mod.state_dict().items())
            for i in range(len(frontend_items)):
                frontend_items[i][1].data[:] = mod_items[i][1].data[:]

        if backbone != 'vgg16':
            self._initialize_weights()
            self.alt_backbone, out_ch = _build_torchvision_feature_extractor(backbone)
            self.alt_proj = nn.Conv2d(out_ch, 512, kernel_size=1)

    def forward(self,x):
        if self.alt_backbone is None:
            x = self.frontend(x)
        else:
            x = self.alt_backbone(x)
            x = self.alt_proj(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
                
def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)                


def _build_torchvision_feature_extractor(backbone_name: str):
    name = backbone_name.lower()
    if name not in models.list_models(module=models):
        raise ValueError(f"Unsupported torchvision backbone: {backbone_name}")

    model = models.get_model(name, weights="DEFAULT")
    model.eval()

    class Last4D(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, inp):
            last = None
            handles = []

            def _hook(_m, _i, o):
                nonlocal last
                if isinstance(o, torch.Tensor) and o.dim() == 4 and o.shape[2] >= 2 and o.shape[3] >= 2:
                    last = o

            for _, m in self.base.named_modules():
                handles.append(m.register_forward_hook(_hook))
            _ = self.base(inp)
            for h in handles:
                h.remove()
            if last is None:
                raise ValueError(f"Could not extract 4D feature map from backbone {backbone_name}")
            return last

    dummy = torch.zeros(1, 3, 224, 224)
    extractor = Last4D(model)
    with torch.no_grad():
        feat = extractor(dummy)
    return extractor, int(feat.shape[1])