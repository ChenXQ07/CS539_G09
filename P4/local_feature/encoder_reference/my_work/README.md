# UNetFormer encoder contribution

Two interchangeable ResNet-18 encoders return four spatial feature maps for a
teammate's UNetFormer decoder. All work for this contribution lives here.

## Quick start

Run from the project root (the directory containing `my_work`). Python 3.12 is
the tested interpreter family. Create an environment and install requirements:

```powershell
python -m venv my_work/.venv
my_work/.venv/Scripts/python.exe -m pip install -r my_work/requirements.txt
my_work/.venv/Scripts/python.exe -m pytest my_work/tests -q
my_work/.venv/Scripts/python.exe -m my_work.examples.inspect_encoder_outputs
```

The example and tests use random weights and synthetic data by default, with no
downloads. The example copies the reference weights into the custom encoder so
the comparison is meaningful. To check actual pretrained weights:

```powershell
$env:TORCH_HOME = "$PWD/my_work/.cache/torch"
my_work/.venv/Scripts/python.exe -m my_work.examples.inspect_encoder_outputs --weights imagenet
```

This downloads the official torchvision checkpoint if it is not cached. Failures
are reported explicitly; pretrained initialization never silently becomes random.
See [verification.md](docs/verification.md) for tested versions and results.

## Usage

```python
import torch
from my_work.pretrained_encoder import ResNet18Encoder
# Interchangeable alternative:
# from my_work.custom_encoder import ResNet18Encoder

encoder = ResNet18Encoder(weights="imagenet")
images = torch.randn(2, 3, 512, 512)  # synthetic example, not image preprocessing
f1, f2, f3, f4 = encoder(images)
```

Both constructors default to `weights="imagenet"`. Pass `weights=None` for random
initialization. Parameters are trainable, and models start in training mode.
The model performs no normalization or band reordering. Input must be a floating
tensor `[B,3,H,W]` with nonempty batch and H/W positive multiples of 32. The data
handoff uses float32. For training, BatchNorm needs more than one value per
channel: a batch of one with a 32x32 patch is not valid at the deepest stage.
Use the default 512x512 patches, or evaluation mode for such tiny checks.

| Feature | Channels | Input-relative size | Size at 512x512 |
|---|---:|---|---|
| f1 | 64 | H/4 x W/4 | 128x128 |
| f2 | 128 | H/8 x W/8 | 64x64 |
| f3 | 256 | H/16 x W/16 | 32x32 |
| f4 | 512 | H/32 x W/32 | 16x16 |

## What is implemented

- `pretrained_encoder`: torchvision ResNet-18 stem and four stages, with the
  classification pool and classifier removed.
- `custom_encoder`: explicit PyTorch stem, eight BasicBlocks, and stage outputs.
  It does not instantiate a torchvision model. In ImageNet mode it uses
  torchvision's weight enum only to retrieve the checkpoint.
- `_contract.py`: shared input/weight argument validation, without model layers.
- Tests: output shapes, input errors, gradients, BatchNorm mode behavior,
  checkpoint round trip, strict weight import, and numerical equivalence.
- Example: compares both implementations on synthetic inputs or a supplied batch.

The stem is 7x7 Conv(3->64, stride 2), BatchNorm, ReLU, and 3x3 max pooling
(stride 2). Each of the four stages contains two BasicBlocks, with widths
64/128/256/512. Stage 1 keeps the stem resolution; the first block in each later
stage downsamples by two. A BasicBlock applies Conv-BN-ReLU-Conv-BN, adds its
shortcut, then applies ReLU. Stage transitions project the shortcut using a
stride-2 1x1 convolution and BatchNorm. The custom initialization uses Kaiming
normal convolution weights and BatchNorm scale 1 / bias 0.

## Relationship to the paper

UNetFormer uses a CNN encoder and a transformer decoder. Section 3.1 and Figure 3
describe the pretrained ResNet-18 encoder. The paper's Vaihingen setup uses
NIR/red/green TOP imagery, excluding DSM/nDSM. Transformer attention, decoder-side
channel projection and weighted fusion, feature refinement, and class prediction
are outside this contribution.

This implementation uses `ResNet18_Weights.IMAGENET1K_V1`. The official GeoSeg
UNetFormer source defaults to timm's `swsl_resnet18`. Therefore, this delivers the
same ResNet-18 feature architecture/interface, not identical official pretraining
or a claim to reproduce the paper's reported accuracy. The Swin-based
FT-UNetFormer variant is also outside this scope.

References:
- Wang et al., *UNetFormer: A UNet-like Transformer for Efficient Semantic
  Segmentation of Remote Sensing Urban Scene Imagery*, local `unetformer paper.pdf`
  in the project root; https://arxiv.org/abs/2109.08937
- Official model: https://github.com/WangLibo1995/GeoSeg/blob/main/geoseg/models/UNetFormer.py
- ResNet reference: https://docs.pytorch.org/vision/stable/_modules/torchvision/models/resnet.html

## Handoff and limits

Read [data_contract.md](docs/data_contract.md) with the preprocessing partner and
[decoder_handoff.md](docs/decoder_handoff.md) with the decoder/training owner.
The real-batch check remains pending until a confirmed preprocessed batch arrives.
No raw imagery is assumed to meet that contract. Unit tests establish encoder
correctness, not segmentation accuracy. Training, class labels, losses, metrics,
and trained segmentation checkpoints belong to the other project components.
