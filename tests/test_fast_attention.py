import copy
import os

import pytest
import torch

from models.fast_attention import enable_fast_attention
from models.sam_builder import build_sam_parts

CKPT = os.path.join(os.path.dirname(__file__), "..", "pretrained", "sam_vit_b_01ec64.pth")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda")
@pytest.mark.parametrize("size", [512, 384])
def test_fast_attention_matches_sam(size):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    enc, _, _ = build_sam_parts(size, 1, CKPT)
    enc = enc.cuda().eval()
    fast = enable_fast_attention(copy.deepcopy(enc))
    x = torch.randn(1, 3, size, size, device="cuda")
    with torch.no_grad():
        assert torch.allclose(enc(x), fast(x), atol=1e-4)
