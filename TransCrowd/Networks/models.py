# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models import register_model
from timm.layers import trunc_normal_
import torchvision.models as tv_models



class VisionTransformer_token(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))

        trunc_normal_(self.pos_embed, std=.02)

        self.output1 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 1)
        )
        self.output1.apply(self._init_weights)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        x = self.output1(x)

        return x




class VisionTransformer_gap(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))

        trunc_normal_(self.pos_embed, std=.02)

        self.output1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(6912 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
        self.output1.apply(self._init_weights)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        x = x[:, 1:]

        return x

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        x = F.adaptive_avg_pool1d(x, (48))
        x = x.view(x.shape[0], -1)
        # print(x.shape)
        x = self.output1(x)
        return x



@register_model
def base_patch16_384_token(pretrained=False, **kwargs):
    model = VisionTransformer_token(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        '''download from https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth'''
        checkpoint = torch.load(
            './Networks/deit_base_patch16_384-8de9b5d1.pth')
        model.load_state_dict(checkpoint["model"], strict=False)
        print("load transformer pretrained")
    return model


@register_model
def base_patch16_384_gap(pretrained=False, **kwargs):
    model = VisionTransformer_gap(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        '''download from https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth'''
        checkpoint = torch.load(
            './Networks/deit_base_patch16_384-8de9b5d1.pth')
        model.load_state_dict(checkpoint["model"], strict=False)
        print("load transformer pretrained")
    return model


# ---------------------------------------------------------------------------
# VGG16 + FC regression head
# ---------------------------------------------------------------------------

class VGG16CountNet(nn.Module):
    """VGG16 backbone with a lightweight FC regression head for crowd counting."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        vgg = tv_models.vgg16(pretrained=pretrained)
        # Keep only the convolutional feature extractor
        self.features = vgg.features          # output: (B, 512, H/32, W/32)
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        # Regression head: 512*7*7 = 25088 → 512 → 1
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
        )
        self._init_regressor()

    def _init_regressor(self):
        for m in self.regressor.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.regressor(x)
        return x


def vgg16_count(pretrained: bool = True) -> VGG16CountNet:
    return VGG16CountNet(pretrained=pretrained)


# ---------------------------------------------------------------------------
# ResNet-50 + FC regression head
# ---------------------------------------------------------------------------

class ResNet50CountNet(nn.Module):
    """ResNet-50 backbone with a lightweight FC regression head for crowd counting."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        resnet = tv_models.resnet50(pretrained=pretrained)
        # Remove the original classification FC layer; keep up to global avg pool
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # (B, 2048, 1, 1)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
        )
        self._init_regressor()

    def _init_regressor(self):
        for m in self.regressor.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.backbone(x)
        x = self.regressor(x)
        return x


def resnet50_count(pretrained: bool = True) -> ResNet50CountNet:
    return ResNet50CountNet(pretrained=pretrained)

