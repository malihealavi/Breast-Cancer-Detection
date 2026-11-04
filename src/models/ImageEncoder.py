import torch
import timm
from torch import nn

class ImageEncoderPerView(nn.Module):
    def __init__(self, model_name='efficientnetv2_rw_m', pretrained=True):
        super().__init__()
        if model_name == 'efficientnetv2_rw_m':
            self.backbone = timm.create_model(
                model_name,
                pretrained=True,
                in_chans=1,
                num_classes=0,
                pretrained_cfg_overlay=dict(file='/kaggle/input/datasets/malihealavi/efficientnetv2-rw-m-weights/efficientnetv2_rw_m.pth')
            )
        elif model_name == 'convnext_small':
            self.backbone = timm.create_model(
                'convnext_small',
                pretrained=pretrained,
                in_chans=1,
                num_classes=0,
                pretrained_cfg_overlay=dict(file='/kaggle/input/datasets/malihealavi/convnext-small/convnext_small_weights.pth')
            )
        else:
            raise ValueError(f"Unknown model_name: {model_name}")
        self.num_features = self.backbone.num_features

    def forward(self, images):
        # images: (B, MAX_VIEWS, 1, H, W)
        B, N, C, H, W = images.shape
        x = images.view(B * N, C, H, W)
        feats = self.backbone(x)
        return feats.view(B, N, self.num_features)