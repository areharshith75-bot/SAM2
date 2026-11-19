import copy
import os

import torch

from models.lora_sam import LoRAQKV, inject_lora
from models.sam_builder import build_sam_parts

CKPT = os.path.join(os.path.dirname(__file__), "..", "pretrained", "sam_vit_b_01ec64.pth")


def test_lora_zero_delta_parity_and_trainable_set():
    enc, _, _ = build_sam_parts(384, num_classes=1, checkpoint=CKPT)
    ref = copy.deepcopy(enc).eval()
    enc = inject_lora(enc, r=4).eval()
    x = torch.randn(1, 3, 384, 384)
    with torch.no_grad():
        assert torch.equal(ref(x), enc(x))  # Delta W = 0 at init => exact parity

    trainable = {n for n, p in enc.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n for n in trainable)
    assert len(trainable) == 12 * 3 * 2  # 12 blocks x {Q,K,V} x {A,B}
    n = sum(p.numel() for n_, p in enc.named_parameters() if p.requires_grad)
    assert n == 12 * 3 * (768 * 4 + 4 * 768)  # 221,184


def test_lora_slices_are_independent():
    lin = torch.nn.Linear(8, 24)
    m = LoRAQKV(lin, r=2)
    torch.nn.init.ones_(m.lora_B[1].weight)  # only K gets an update
    x = torch.randn(3, 8)
    d = m(x) - lin(x)
    assert torch.all(d[:, :8] == 0) and torch.all(d[:, 16:] == 0) and d[:, 8:16].abs().sum() > 0
