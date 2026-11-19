import os

import numpy as np
import pytest
import torch

from models.frequency_extractor import FrequencyExtractor, amplitude_spectrum
from models.sam_builder import build_sam_parts

CKPT = os.path.join(os.path.dirname(__file__), "..", "pretrained", "sam_vit_b_01ec64.pth")


def test_amplitude_matches_numpy_and_is_float32():
    x = torch.randn(2, 3, 64, 64)
    a = amplitude_spectrum(x)
    ref = np.log1p(np.abs(np.fft.fft2(x.numpy(), axes=(-2, -1))))
    ref = (ref - ref.mean((-2, -1), keepdims=True)) / (ref.std((-2, -1), keepdims=True, ddof=1) + 1e-6)
    assert a.dtype == torch.float32
    np.testing.assert_allclose(a.numpy(), ref, atol=1e-3)
    assert torch.allclose(a.mean((-2, -1)), torch.zeros(2, 3), atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda")
def test_fft_float32_under_bf16_autocast_and_finite():
    x = torch.randn(2, 3, 128, 128, device="cuda").bfloat16()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        a = amplitude_spectrum(x)
    assert a.dtype == torch.float32 and torch.isfinite(a).all()
    z = amplitude_spectrum(torch.zeros(1, 3, 32, 32))  # degenerate input must not NaN
    assert torch.isfinite(z).all()


@pytest.mark.parametrize("size", [512, 384])
def test_dual_embedding_shapes_and_grad(size):
    enc, _, _ = build_sam_parts(size, num_classes=2, checkpoint=CKPT)
    assert enc.pos_embed.shape == (1, size // 16, size // 16, 768)
    fe = FrequencyExtractor(enc.patch_embed, enc.pos_embed)
    x = torch.randn(1, 3, size, size, requires_grad=True)
    e_img, e_freq = fe(x)
    assert e_img.shape == e_freq.shape == (1, size // 16, size // 16, 768)
    assert not torch.allclose(e_img, e_freq)
    (e_img.sum() + e_freq.sum()).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
