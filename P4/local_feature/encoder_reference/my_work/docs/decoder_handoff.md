# Decoder handoff

Both implementations export `ResNet18Encoder(weights="imagenet")` and return
`(f1, f2, f3, f4)` in shallow-to-deep order. Metadata:
`out_channels=(64,128,256,512)` and `out_strides=(4,8,16,32)`.

```python
from my_work.custom_encoder import ResNet18Encoder

encoder = ResNet18Encoder(weights="imagenet")
# decoder is supplied by the teammate, not this package.
# For the official GeoSeg Decoder interface:
height, width = images.shape[-2:]
features = encoder(images)
predictions = decoder(*features, height, width)
```

Construct the decoder using `encoder_channels=encoder.out_channels`. Its 1x1
projections, weighted fusion, global-local transformer blocks, refinement head,
and segmentation heads remain decoder responsibilities. Pass original image
dimensions so its predictions can be resized correctly. The official decoder
returns main and auxiliary predictions during training and main predictions
during evaluation; this encoder always returns four features in both modes.

The parent segmentation model must register encoder and decoder as submodules
and move both, plus images, to the same device. Do not detach features or put the
encoder inside `no_grad()` during joint training. Include encoder parameters in
the optimizer. Calling `.train()`/`.eval()` on the parent propagates to both.
Freezing parameters alone does not freeze BatchNorm running statistics; the
training owner must explicitly set evaluation mode if a fully frozen encoder is
intended. No parameters are frozen by default.

## Save and restore

Save the initialization provenance alongside weights. For a standalone encoder:

```python
torch.save({
    "variant": "custom",  # or "pretrained"
    "weight_source": encoder.weight_source,
    "channels": ["NIR", "red", "green"],
    "normalization": {"mean": [0.485, 0.456, 0.406],
                      "std": [0.229, 0.224, 0.225]},
    "state_dict": encoder.state_dict(),
}, "my_work/encoder.pt")
```

```python
import torch
from my_work.custom_encoder import ResNet18Encoder as Custom
from my_work.pretrained_encoder import ResNet18Encoder as Pretrained

checkpoint = torch.load("my_work/encoder.pt", map_location="cpu", weights_only=True)
cls = {"custom": Custom, "pretrained": Pretrained}[checkpoint["variant"]]
encoder = cls(weights=None)  # loading saved weights must not need a download
encoder.load_state_dict(checkpoint["state_dict"], strict=True)
encoder.weight_source = checkpoint["weight_source"]
encoder.eval()
```

Both variants intentionally share state-dict keys, so an encoder-only checkpoint
can also be transferred between them with strict loading. Do not feed a full
UNetFormer checkpoint directly to this method: it has different module prefixes
and decoder entries. Custom `load_classification_weights()` accepts reference
ResNet-18 weights, strips only `fc.weight`/`fc.bias`, and rejects other mismatches.

Checkpoint provenance should also record the training run, data split, and any
later normalization changes. `weight_source` describes initialization, not
whether the model has since been fine-tuned. Encoder-only tests do not produce a
trained model or justify accuracy claims.
