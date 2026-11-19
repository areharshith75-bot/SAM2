import os

import pytest
import torch

from models.fsam import FSAM
from models.prompt_generator import AutomatedPromptGenerator

CKPT = os.path.join(os.path.dirname(__file__), "..", "pretrained", "sam_vit_b_01ec64.pth")


def test_prompt_generator_math():
    g = AutomatedPromptGenerator(256, 64)
    e = torch.randn(3, 256, 8, 8)
    p = e.mean((2, 3)) + e.amax((2, 3))
    p_hat, alpha = g.refine(p)
    assert torch.allclose(alpha.sum(-1), torch.ones(3)) and p_hat.shape == (3, 256)
    cos = torch.nn.functional.cosine_similarity(p[:, None], g.memory[None], dim=-1)
    assert torch.allclose(alpha, cos.softmax(-1), atol=1e-6)
    assert g(e).shape == (3, 256, 8, 8)
    assert g(e[:1]).shape == (1, 256, 8, 8)  # batch of 1 works


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda")
@pytest.mark.parametrize("size,ncls", [(512, 2), (384, 1)])
def test_fsam_forward_backward_gpu(size, ncls):
    m = FSAM(size, ncls, checkpoint=CKPT).cuda().train()
    bd = m.trainable_parameter_breakdown()
    assert bd["other"] == 0
    assert bd["lora"] == 221_184
    assert bd["freq_adapters"] == 12 * (2 * 768 + 768 * 192 + 192 + 192 * 768 + 768)
    assert bd["prompt_generator"] == 256 * 256 + (513 * 256 + 256)
    print(size, bd)
    x = torch.randn(2, 3, size, size, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = m(x)
    assert out.shape == (2, ncls, size, size)
    out.float().mean().backward()
    for n, p in m.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), n
