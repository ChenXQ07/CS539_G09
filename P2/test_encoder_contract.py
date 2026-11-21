import io

import pytest
import torch

from P2.custom_encoder import ResNet18Encoder as Custom
from P2.pretrained_encoder import ResNet18Encoder as Pretrained


@pytest.fixture(params=[Custom, Pretrained])
def encoder_class(request):
    return request.param


@pytest.mark.parametrize("size", [(256, 256), (512, 512), (256, 512)])
def test_shapes(encoder_class, size):
    model = encoder_class(weights=None).eval()
    x = torch.randn(2, 3, *size)
    with torch.no_grad():
        outputs = model(x)
    assert isinstance(outputs, tuple) and len(outputs) == 4
    assert model.out_channels == (64, 128, 256, 512)
    assert model.out_strides == (4, 8, 16, 32)
    assert not hasattr(model, "fc") and not hasattr(model, "avgpool")
    for f, channels, stride in zip(outputs, model.out_channels, model.out_strides):
        assert f.shape == (2, channels, size[0] // stride, size[1] // stride)
        assert f.device == x.device and f.dtype == x.dtype
        assert torch.isfinite(f).all()


@pytest.mark.parametrize("x,error", [
    (None, TypeError),
    (torch.zeros(3, 64, 64), ValueError),
    (torch.zeros(1, 4, 64, 64), ValueError),
    (torch.zeros(1, 3, 64, 64, dtype=torch.int64), TypeError),
    (torch.zeros(1, 3, 65, 64), ValueError),
    (torch.zeros(1, 3, 0, 64), ValueError),
    (torch.zeros(0, 3, 64, 64), ValueError),
])
def test_invalid_inputs(encoder_class, x, error):
    with pytest.raises(error):
        encoder_class(weights=None)(x)


def test_invalid_weights(encoder_class):
    with pytest.raises(ValueError, match="weights"):
        encoder_class(weights="unknown")


def test_gradients(encoder_class):
    model = encoder_class(weights=None).train()
    assert all(p.requires_grad for p in model.parameters())
    outputs = model(torch.randn(2, 3, 64, 64))
    sum(f.square().mean() for f in outputs).backward()
    for component in (model.conv1, model.layer1, model.layer2, model.layer3, model.layer4):
        for parameter in component.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
        assert sum(p.grad.abs().sum() for p in component.parameters()) > 0


def test_batchnorm_modes(encoder_class):
    model = encoder_class(weights=None)
    norms = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    before = [m.running_mean.clone() for m in norms]
    with torch.no_grad():
        model.train()(torch.randn(2, 3, 64, 64) + 2)
    assert all(not torch.equal(a, m.running_mean) for a, m in zip(before, norms))
    before = [(m.running_mean.clone(), m.running_var.clone()) for m in norms]
    with torch.no_grad():
        model.eval()(torch.randn(2, 3, 64, 64))
    for (mean, var), m in zip(before, norms):
        assert torch.equal(mean, m.running_mean)
        assert torch.equal(var, m.running_var)


def test_checkpoint_roundtrip(encoder_class):
    model = encoder_class(weights=None).eval()
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = encoder_class(weights=None).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        for actual, expected in zip(restored(x), model(x)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_weight_download_failure_is_explicit(encoder_class, monkeypatch):
    from torchvision.models import ResNet18_Weights

    def fail(*args, **kwargs):
        raise OSError("simulated unavailable download")

    monkeypatch.setattr(ResNet18_Weights, "get_state_dict", fail)
    with pytest.raises(RuntimeError, match="Unable to load ImageNet"):
        encoder_class(weights="imagenet")
