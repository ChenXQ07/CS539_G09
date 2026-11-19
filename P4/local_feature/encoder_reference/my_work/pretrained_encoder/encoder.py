"""Torchvision ResNet-18 with its classification head removed."""

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from .._contract import validate_input, validate_weights


class ResNet18Encoder(nn.Module):
    out_channels = (64, 128, 256, 512)
    out_strides = (4, 8, 16, 32)

    def __init__(self, weights="imagenet"):
        super().__init__()
        validate_weights(weights)
        self.weight_source = weights
        try:
            reference = resnet18(
                weights=ResNet18_Weights.IMAGENET1K_V1 if weights else None
            )
        except Exception as exc:
            if weights is None:
                raise
            raise RuntimeError(
                "Unable to load ImageNet ResNet-18 weights. Check network/cache access; "
                "use weights=None explicitly for offline random initialization."
            ) from exc
        for name in ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4"):
            setattr(self, name, getattr(reference, name))

    def forward(self, images):
        validate_input(images)
        x = self.maxpool(self.relu(self.bn1(self.conv1(images))))
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return f1, f2, f3, f4
