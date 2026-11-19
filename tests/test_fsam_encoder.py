import copy
import os

import torch

from models.fsam_encoder import FSAMEncoder, FrequencyAdapter
from models.sam_builder import build_sam_parts

CKPT = os.path.join(os.path.dirname(__file__), "..", "pretrained", "sam_vit_b_01ec64.pth")


def test_adapter_architecture():
    a = FrequencyAdapter(768, 4)
    assert isinstance(a.norm, torch.nn.LayerNorm)
    assert (a.down.in_features, a.down.out_features, a.up.out_features) == (768, 192, 768)
    assert torch.all(a(torch.randn(2, 5, 768)) == 0)  # zero-init up projection


def test_encoder_parity_at_init_shapes_and_grads():
    sam_enc, _, _ = build_sam_parts(384, num_classes=1, checkpoint=CKPT)
    ref = copy.deepcopy(sam_enc).eval()
    enc = FSAMEncoder(sam_enc).eval()
    assert len(enc.freq_adapters) == 12
    x = torch.randn(1, 3, 384, 384)
    with torch.no_grad():
        out = enc(x)
        # LoRA B=0 and adapter up=0 -> identical to pretrained SAM image encoder at init
        assert torch.allclose(out, ref(x), atol=1e-5)
    assert out.shape == (1, 256, 24, 24)

    # perturb adapters so the frequency stream actually contributes, then check grads reach it
    enc.train()
    for a in enc.freq_adapters:
        torch.nn.init.normal_(a.up.weight, std=1e-3)
    enc(x).square().mean().backward()
    for n, p in enc.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), n
    frozen = [n for n, p in enc.named_parameters() if not p.requires_grad]
    assert all("lora_" not in n and "freq_adapters" not in n for n in frozen)
    assert any(n.startswith("neck") for n in frozen) and any("patch_embed" in n for n in frozen)
