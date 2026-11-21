import pytest
import torch
from torchvision.models import resnet18

from P2.custom_encoder import ResNet18Encoder as Custom
from P2.pretrained_encoder import ResNet18Encoder as Pretrained


@pytest.mark.parametrize("size", [(256, 256), (512, 512), (256, 512)])
def test_shared_random_weights(size):
    torch.manual_seed(539)
    reference = Pretrained(weights=None).eval()
    custom = Custom(weights=None).eval()
    custom.load_state_dict(reference.state_dict(), strict=True)
    x = torch.randn(1, 3, *size)
    with torch.no_grad():
        for actual, expected in zip(custom(x), reference(x)):
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_classification_import_is_strict():
    state = resnet18(weights=None).state_dict()
    custom = Custom(weights=None)
    custom.load_classification_weights(state)
    bad = dict(state, unexpected=torch.tensor(0))
    with pytest.raises(RuntimeError, match="Unexpected key"):
        custom.load_classification_weights(bad)
    missing = dict(state)
    del missing["conv1.weight"]
    with pytest.raises(RuntimeError, match="Missing key"):
        custom.load_classification_weights(missing)
