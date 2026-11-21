"""Explicit ResNet-18 architecture; torchvision is used only to fetch weights."""

from torch import nn

if __package__:
    from ._contract import validate_input, validate_weights
else:
    from _contract import validate_input, validate_weights


class BasicBlock(nn.Module):
    """Two convolutions plus an identity or projected residual shortcut."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet18Encoder(nn.Module):
    out_channels = (64, 128, 256, 512)
    out_strides = (4, 8, 16, 32)

    def __init__(self, weights="imagenet"):
        super().__init__()
        validate_weights(weights)
        self.weight_source = weights
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._stage(64, 64, 1)
        self.layer2 = self._stage(64, 128, 2)
        self.layer3 = self._stage(128, 256, 2)
        self.layer4 = self._stage(256, 512, 2)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        if weights == "imagenet":
            from torchvision.models import ResNet18_Weights

            try:
                state = ResNet18_Weights.IMAGENET1K_V1.get_state_dict(
                    progress=True, check_hash=True
                )
            except Exception as exc:
                raise RuntimeError(
                    "Unable to load ImageNet ResNet-18 weights. Check network/cache access; "
                    "use weights=None explicitly for offline random initialization."
                ) from exc
            self.load_classification_weights(state)

    @staticmethod
    def _stage(in_channels, out_channels, stride):
        return nn.Sequential(
            BasicBlock(in_channels, out_channels, stride),
            BasicBlock(out_channels, out_channels),
        )

    def load_classification_weights(self, state):
        """Import reference weights, excluding only the two classifier entries."""
        encoder_state = {k: v for k, v in state.items() if k not in ("fc.weight", "fc.bias")}
        return self.load_state_dict(encoder_state, strict=True)

    def forward(self, images):
        validate_input(images)
        x = self.maxpool(self.relu(self.bn1(self.conv1(images))))
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return f1, f2, f3, f4
