# P4 Local Feature Branch

Owner: @ChenXQ07. Component delivery for the local-feature branch in the submitted G09 proposal. This is not a trained segmentation system.

## Method

LocalFeatureBranch(channels=64) maps [B,64,h,w] to the same shape. It adds parallel 3x3 Conv-BN and 1x1 Conv-BN outputs, with no additional activation, softmax or class head. Convolution biases are disabled. This matches local2(x) + local1(x) in the supplied RSSS GlobalLocalAttention. State-dict suffixes local1/local2 and Sequential indices match the source.

Input is the prepared decoder feature after the shared block normalization, NOT raw imagery or all four encoder outputs. Do not normalize twice. Instantiate three independent branches, one for each GLTB. At 256x256 input their normal resolutions are 8x8, 16x16 and 32x32. BatchNorm requires more than one value per channel in training mode. No parameters are frozen.

## Run from this folder with Python 3.12

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe examples/encoder_connection.py
```

Tests need no data or pretrained downloads. The encoder copy is unchanged from Alex's ZIP. The connection example explicitly uses weights=None to follow the submitted from-scratch plan. Its 1x1 projections and normalizations are smoke-test adapters, NOT the production shared decoder.

## Contents

- p4_local/: local branch implementation.
- tests/: source equivalence, gradients, optimizer update, shapes, BatchNorm and serialization checks.
- examples/encoder_connection.py: three-branch synthetic encoder connection.
- docs/YIN_HANDOFF.md: fusion boundary and pending team integration.
- reference/UNetFormer_RSSS.py: unchanged supplied source for provenance; only its ConvBN definition is executed by tests.
- encoder_reference/my_work/: unchanged Alex contribution.
- verification/: actual local results and source hashes.

## Limits

Full global/local fusion, shared decoder, real-data training, GPU checks, held-out mIoU, and final report/slides remain team integration work. Synthetic gradient checks do not establish segmentation accuracy. Alex's default ImageNet initialization and proposed 512x512/ImageNet-normalization contract differ from the submitted 256x256 from-scratch plan; agree on preprocessing before real-data integration.
