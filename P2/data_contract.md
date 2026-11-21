# Preprocessing handoff

Status: proposed initial contract; partner confirmation and a real batch pending.

| Field | Required contract |
|---|---|
| Bands | Exactly NIR, red, green, in that order |
| Layout | `[batch, 3, height, width]` |
| Type | float32 |
| Dimensions | 512x512 initially; positive multiples of 32 supported |
| Scaling | Document original units/range and map imagery to [0,1] |
| Normalization | Once, outside the encoder, after scaling |
| Invalid pixels | Resolve NaN/infinity/nodata before encoding; record policy |
| Identification | Preserve original tile ID, patch location, and split |
| Splits | Split original tiles before extracting patches |

For the initial ImageNet initialization, use mean `(0.485, 0.456, 0.406)` and
standard deviation `(0.229, 0.224, 0.225)` on the three channels after scaling.
For example, given already scaled float32 images:

```python
mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
images = (images - mean) / std
```

Divide by 255 only if the source really is 8-bit imagery with that encoding.
Do not infer scaling from one patch's maximum. Do not apply torchvision's full
classification transform: resizing/center cropping would change segmentation
geometry. Use only the agreed normalization after spatial preprocessing.

ImageNet's RGB pretraining and statistics are a baseline transfer choice for
NIR/red/green, not evidence that these bands are RGB. Keep the exact order and
normalization documented for both implementations. If the partner already uses
training-set statistics, reconcile the choice before integration; never normalize
twice. Any statistics estimated from this dataset must use training tiles only.

The partner should supply one normalized batch saved as a tensor:

```python
torch.save(images.cpu(), "partner_batch.pt")
```

Alongside it, supply channel meanings, source scaling, normalization parameters,
patch dimensions, tile IDs, crop coordinates, split membership, and nodata policy.
Store the handoff file as `P2/partner_batch.pt`. From the repository root,
check the batch with:

```python
import torch
from P2.custom_encoder import ResNet18Encoder

images = torch.load("P2/partner_batch.pt", map_location="cpu", weights_only=True)
assert torch.isfinite(images).all()
encoder = ResNet18Encoder(weights="imagenet").eval()
with torch.no_grad():
    features = encoder(images)
assert all(torch.isfinite(feature).all() for feature in features)
print([tuple(feature.shape) for feature in features])
```

This checks finite features and the input shape contract. It cannot prove band
order or normalization correctness from the numbers alone. Preprocessing remains
responsible for resolving nonfinite values before training.

Masks are not inputs to the encoder. The training owner separately needs masks
aligned with image augmentations, six-class integer IDs, and an ignore-label
policy. Do not assume that available `4c` or `sam_augmented` files meet this
contract. Height input and SAM-derived inputs are not enabled in this version.
