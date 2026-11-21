import torch

from utils.style_aug import gin


def test_gin_keeps_shape_and_standardisation():
    torch.manual_seed(0)
    x = torch.randn(6, 3, 32, 32)
    y = gin(x, p=1.0)
    assert y.shape == x.shape and torch.isfinite(y).all()
    assert torch.allclose(y.mean((2, 3)), torch.zeros(6, 3), atol=1e-4)
    assert torch.allclose(y.std((2, 3)), torch.ones(6, 3), atol=1e-2)
    assert not torch.allclose(y, x)
    assert torch.equal(gin(x, p=0.0), x)
