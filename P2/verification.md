# Verification results

The original package records verification on October 30, 2026, on Windows with
CPU execution. Those historical results are preserved below. The current P2
layout has not been verified by a new test run for this documentation update.

## Install dependencies

From the repository root, using your Python environment:

```text
python -m pip install -r P2/requirements.txt
```

## Run the custom encoder

Run this Python example with the repository root as your working directory:

```python
import torch
from P2.custom_encoder import ResNet18Encoder

model = ResNet18Encoder(weights=None).eval()
images = torch.randn(1, 3, 256, 256)
with torch.no_grad():
    features = model(images)
assert all(torch.isfinite(feature).all() for feature in features)
print([tuple(feature.shape) for feature in features])
```

Expected shapes:

```text
[(1, 64, 64, 64), (1, 128, 32, 32), (1, 256, 16, 16), (1, 512, 8, 8)]
```

This uses random initialization and does not download weights. Use
`weights="imagenet"` to request ImageNet initialization, which requires cached
weights or network access. Import `P2.custom_encoder`. Executing `custom_encoder.py` directly only defines
the model and prints no output.

## Automated checks

Run the current P2 tests from the repository root:

```text
python -m pytest P2 -q
```

Historical Windows result: **34 passed in 25.91 seconds**. This is the original
package's recorded result, not a new test result. These checks cover both encoders'
shapes and metadata, invalid inputs and weight arguments, gradient propagation,
BatchNorm training/evaluation behavior, save/load equivalence, explicit download
failure, strict classification-weight import, and shared-weight numerical
equivalence. Shapes were checked at 256x256, 512x512, and 256x512, including
batches of two. The suite does not download weights.

## Local feature connection check

The repository must also contain `P4/local_feature/p4_local`. Run:

```text
python P2/encoder_connection.py
```

This script imports the P2 custom encoder and P4 local feature branch, checks
backpropagation and a parameter update, and writes `P2/encoder_connection.json`.
It uses synthetic data and `weights=None`. This is a connection smoke test,
not full decoder integration or a segmentation accuracy evaluation.

## Recorded dependency lock (requirements-lock.txt)

The complete original `requirements-lock.txt` contents are embedded below;
no separate lock file is required to read this record. The recorded Python
version was **3.12.14** on Windows.

```text
colorama==0.4.6
filelock==4.2.5
fsspec==2026.9.0
iniconfig==2.3.1
jinja2==3.1.6
markupsafe==3.0.4
mpmath==1.3.0
networkx==3.7
numpy==2.5.4
packaging==26.3
pillow==13.0.0
pluggy==1.7.0
pygments==2.21.0
pytest==9.2.0
setuptools==84.2.1
sympy==1.14.0
torch==2.15.0
torchvision==0.29.2
typing-extensions==4.16.0
```

To inspect the versions installed in your Python environment:

```text
python --version
python -m pip freeze
```

## Historical pretrained weight check (Windows)

The official `resnet18-f37072fd.pth` checkpoint was downloaded through torchvision
with its hash check and cached under `my_work/.cache/torch`. Both encoder classes
independently loaded ImageNet weights, then ran in evaluation mode on the same
synthetic float32 CPU tensor `[1,3,512,512]`.

| Feature | Shape | Maximum absolute difference |
|---|---|---:|
| f1 | [1,64,128,128] | 0 |
| f2 | [1,128,64,64] | 0 |
| f3 | [1,256,32,32] | 0 |
| f4 | [1,512,16,16] | 0 |

All features were finite and satisfied `rtol=1e-4, atol=1e-5`.

## Pending checks

- Partner-provided batch: pending confirmed preprocessing and band metadata.
- GPU execution: not tested; original Windows CPU correctness checks completed.
- Actual teammate decoder integration: pending their implementation.
- Segmentation accuracy: outside encoder-only verification; requires the trained
  complete model and held-out labeled data.

Existing raw and derived imagery was not substituted for the partner's batch,
because its channel/scaling contract has not been confirmed.
