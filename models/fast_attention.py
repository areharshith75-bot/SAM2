"""Numerically-equivalent, faster forward for SAM's ViT attention.

SAM computes softmax(q k^T * scale + rel_pos_bias) v explicitly. Here the decomposed relative
position term is built as an additive bias and passed to torch's fused
`scaled_dot_product_attention` (memory-efficient kernel). Same function, less memory traffic.
"""
import torch
import torch.nn.functional as F
from segment_anything.modeling.image_encoder import Attention, get_rel_pos


def _rel_bias(q, rel_pos_h, rel_pos_w, hw):
    H, W = hw
    Rh = get_rel_pos(H, H, rel_pos_h)
    Rw = get_rel_pos(W, W, rel_pos_w)
    BH, _, dim = q.shape
    r_q = q.reshape(BH, H, W, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh.to(q.dtype))
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw.to(q.dtype))
    bias = rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    return bias.reshape(BH, H * W, H * W)


def fast_forward(self: Attention, x: torch.Tensor) -> torch.Tensor:
    B, H, W, _ = x.shape
    qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
    bias = _rel_bias(q, self.rel_pos_h, self.rel_pos_w, (H, W)) if self.use_rel_pos else None
    out = F.scaled_dot_product_attention(q[None], k[None], v[None],
                                         attn_mask=None if bias is None else bias[None], scale=self.scale)[0]
    out = out.view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
    return self.proj(out)


def enable_fast_attention(image_encoder):
    for blk in image_encoder.blocks:
        blk.attn.forward = fast_forward.__get__(blk.attn, Attention)
    return image_encoder
