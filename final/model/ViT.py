"""Vision Transformer (ViT) encoder + simple-feature-pyramid / FPN decoder.

The encoder is a plain ImageNet-pretrained ViT-B/16 (timm). Tokens from four
evenly spaced blocks are reshaped to 2-D maps (stride 16) and turned into a
multi-scale pyramid (strides 4/8/16/32) the way ViTDet does: deconvolutions
for the shallow levels, max-pooling for the deepest. An FPN top-down pathway
fuses them and a light head predicts per-pixel classes at stride 4, which is
then upsampled to the input size.
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


class ViTSeg(nn.Module):
    def __init__(self,
                 num_classes=6,
                 backbone_name='vit_base_patch16_224.augreg2_in21k_ft_in1k',
                 pretrained=True,
                 img_size=256,
                 out_indices=(2, 5, 8, 11),
                 decode_channels=256,
                 dropout=0.1,
                 drop_path_rate=0.1):
        super().__init__()
        # img_size != 224 -> timm interpolates the pretrained position embedding;
        # dynamic_img_size lets the model also accept other input sizes.
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, img_size=img_size,
                                          num_classes=0, dynamic_img_size=True,
                                          drop_path_rate=drop_path_rate)
        self.out_indices = out_indices
        dim = self.backbone.embed_dim

        # simple feature pyramid from the stride-16 token maps
        self.to_s4 = nn.Sequential(
            nn.ConvTranspose2d(dim, dim // 2, 2, stride=2),
            nn.BatchNorm2d(dim // 2), nn.GELU(),
            nn.ConvTranspose2d(dim // 2, dim // 4, 2, stride=2),
        )
        self.to_s8 = nn.ConvTranspose2d(dim, dim // 2, 2, stride=2)
        self.to_s16 = nn.Identity()
        self.to_s32 = nn.MaxPool2d(2)
        pyr_chs = [dim // 4, dim // 2, dim, dim]

        # FPN
        self.lateral = nn.ModuleList([ConvBNReLU(c, decode_channels, 1) for c in pyr_chs])
        self.fpn_out = nn.ModuleList([ConvBNReLU(decode_channels, decode_channels, 3) for _ in pyr_chs])
        self.fuse = ConvBNReLU(decode_channels * 4, decode_channels, 3)
        self.head = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(decode_channels, num_classes, kernel_size=1),
        )
        self._init_decoder()

    def _init_decoder(self):
        for name, m in self.named_modules():
            if name.startswith('backbone'):
                continue
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
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
        feats = self.backbone.forward_intermediates(x, indices=self.out_indices,
                                                    output_fmt='NCHW', intermediates_only=True)
        pyr = [self.to_s4(feats[0]), self.to_s8(feats[1]), self.to_s16(feats[2]), self.to_s32(feats[3])]

        lat = [l(p) for l, p in zip(self.lateral, pyr)]
        for i in range(len(lat) - 1, 0, -1):  # top-down
            lat[i - 1] = lat[i - 1] + F.interpolate(lat[i], size=lat[i - 1].shape[-2:],
                                                    mode='bilinear', align_corners=False)
        outs = [conv(l) for conv, l in zip(self.fpn_out, lat)]
        size = outs[0].shape[-2:]
        outs = [outs[0]] + [F.interpolate(o, size=size, mode='bilinear', align_corners=False) for o in outs[1:]]

        y = self.head(self.fuse(torch.cat(outs, dim=1)))
        return F.interpolate(y, size=(h, w), mode='bilinear', align_corners=False)
