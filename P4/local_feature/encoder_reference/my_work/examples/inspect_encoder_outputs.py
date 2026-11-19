"""Inspect synthetic or partner-provided, already normalized tensor batches."""

import argparse

import torch

from my_work.custom_encoder import ResNet18Encoder as Custom
from my_work.pretrained_encoder import ResNet18Encoder as Pretrained


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", choices=("none", "imagenet"), default="none")
    parser.add_argument("--batch", help="Path to a .pt file containing a normalized float32 tensor")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(539)
    images = (torch.load(args.batch, map_location="cpu", weights_only=True)
              if args.batch else torch.randn(1, 3, 512, 512))
    if not isinstance(images, torch.Tensor) or images.dtype != torch.float32:
        raise ValueError("Expected a float32 tensor batch")
    if not torch.isfinite(images).all():
        raise ValueError("Input contains NaN or infinity; fix preprocessing first")
    images = images.to(args.device)
    weights = None if args.weights == "none" else "imagenet"
    reference = Pretrained(weights=weights).to(args.device).eval()
    custom = Custom(weights=weights).to(args.device).eval()
    if weights is None:
        custom.load_state_dict(reference.state_dict(), strict=True)
    print("Input:", tuple(images.shape), images.dtype, images.device)
    with torch.no_grad():
        expected = reference(images)
        actual = custom(images)
        for i, (a, b) in enumerate(zip(actual, expected), 1):
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
            print(f"f{i}: {tuple(a.shape)}; max difference: {(a-b).abs().max().item():.3g}")
    print("PASS: both encoders agree. This does not measure segmentation accuracy.")


if __name__ == "__main__":
    main()
