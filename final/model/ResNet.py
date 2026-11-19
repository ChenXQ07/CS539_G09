"""ResNet-50 encoder + U-Net style decoder for semantic segmentation.

A plain CNN baseline: ImageNet-pretrained ResNet-50 (timm) as the encoder, and a
decoder that upsamples step by step, concatenating the encoder skip feature at
each scale (stride 32 -> 16 -> 8 -> 4 -> 2 -> 1).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    """Upsample x to the skip's size, concatenate the skip, then two 3x3 convs."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels + skip_channels, out_channels)
        self.conv2 = ConvBNReLU(out_channels, out_channels)

    def forward(self, x, skip=None):
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        else:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.conv2(self.conv1(x))


class ResNetUNet(nn.Module):
    def __init__(self,
                 num_classes=6,
                 backbone_name='resnet50.a1_in1k',
                 pretrained=True,
                 decoder_channels=(256, 128, 64, 32, 16),
                 dropout=0.1):
        super().__init__()
        # features at strides 2, 4, 8, 16, 32
        self.backbone = timm.create_model(backbone_name, features_only=True,
                                          out_indices=(0, 1, 2, 3, 4), pretrained=pretrained)
        enc = self.backbone.feature_info.channels()  # e.g. [64, 256, 512, 1024, 2048]

        skips = [enc[3], enc[2], enc[1], enc[0], 0]
        in_chs = [enc[4]] + list(decoder_channels[:-1])
        self.blocks = nn.ModuleList([
            DecoderBlock(i, s, o) for i, s, o in zip(in_chs, skips, decoder_channels)
        ])
        self.head = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1),
        )
        self._init_decoder()

    def _init_decoder(self):
        for m in list(self.blocks.modules()) + list(self.head.modules()):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # classifier: small init so the initial logits (and loss) start near uniform
        nn.init.normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        h, w = x.shape[-2:]
        f1, f2, f3, f4, f5 = self.backbone(x)
        skips = [f4, f3, f2, f1, None]
        y = f5
        for block, skip in zip(self.blocks, skips):
            y = block(y, skip)
        y = self.head(y)
        if y.shape[-2:] != (h, w):
            y = F.interpolate(y, size=(h, w), mode='bilinear', align_corners=False)
        return y
