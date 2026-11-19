# Verification results

Verified on October 30, 2026, on Windows with CPU execution.

| Dependency | Tested version |
|---|---|
| Python | 3.12.14 |
| torch | 2.15.0 |
| torchvision | 0.29.2 |
| pytest | 9.2.0 |

Top-level versions are pinned in `requirements.txt`; `requirements-lock.txt`
records all installed packages in the tested environment. The local `.venv`
was created with the bundled Python and populated with `uv pip`, because the
bundled interpreter's ensurepip bootstrap was unavailable. This environment can
run the commands directly without activation. To reinstall into it, use:

```powershell
uv pip install --python my_work/.venv/Scripts/python.exe -r my_work/requirements-lock.txt --cache-dir my_work/.cache/uv
```

On a normal Python installation, the README's `python -m venv` and pip setup
commands are suitable. The environment is local and not intended to be copied
to a teammate's machine; recreate it from requirements there.

## Automated checks

Command from project root:

```powershell
my_work/.venv/Scripts/python.exe -m pytest my_work/tests -q
```

Result: **34 passed in 25.91 seconds**. These cover both encoders' shapes and
metadata, invalid inputs and weight arguments, gradient propagation, BatchNorm
training/evaluation behavior, save/load equivalence, explicit download failure,
strict classification-weight import, and shared-weight numerical equivalence.
Shapes were checked at 256x256, 512x512, and 256x512, including batches of two.
The suite does not download weights.

## Actual pretrained weight check

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
- GPU execution: not tested; CPU correctness checks completed.
- Actual teammate decoder integration: pending their implementation.
- Segmentation accuracy: outside encoder-only verification; requires the trained
  complete model and held-out labeled data.

Existing raw and derived imagery was not substituted for the partner's batch,
because its channel/scaling contract has not been confirmed.
