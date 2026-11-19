"""FSAM Sec. 2.1 / Eq. 2: LoRA on the Q, K, V projections of every frozen SAM ViT block.

SAM fuses Q/K/V into one `nn.Linear(dim, 3*dim)`; `LoRAQKV` keeps that frozen weight and adds
independent low-rank updates A_Q B_Q, A_K B_K, A_V B_V to the three output slices.
"""
import math

import torch
import torch.nn as nn


class LoRAQKV(nn.Module):
    def __init__(self, qkv: nn.Linear, r: int = 4, alpha: float = 4.0):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        assert qkv.out_features == 3 * self.dim
        self.scale = alpha / r
        # A: (dim -> r) Gaussian-ish init, B: (r -> dim) zero init => Delta W = 0 at start
        self.lora_A = nn.ModuleList([nn.Linear(self.dim, r, bias=False) for _ in range(3)])
        self.lora_B = nn.ModuleList([nn.Linear(r, self.dim, bias=False) for _ in range(3)])
        for a, b in zip(self.lora_A, self.lora_B):
            nn.init.kaiming_uniform_(a.weight, a=math.sqrt(5))
            nn.init.zeros_(b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.qkv(x)
        delta = torch.cat([b(a(x)) for a, b in zip(self.lora_A, self.lora_B)], dim=-1)
        return out + self.scale * delta


def inject_lora(image_encoder: nn.Module, r: int = 4, alpha: float = 4.0) -> nn.Module:
    """Freeze 100% of the SAM image encoder, then wrap every block's attn.qkv with LoRA."""
    for p in image_encoder.parameters():
        p.requires_grad = False
    for blk in image_encoder.blocks:
        blk.attn.qkv = LoRAQKV(blk.attn.qkv, r=r, alpha=alpha)
    return image_encoder


def lora_parameters(module: nn.Module):
    return [p for n, p in module.named_parameters() if "lora_" in n]
