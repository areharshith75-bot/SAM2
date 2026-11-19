"""FSAM encoder (Fig. 1): frozen SAM ViT-B + LoRA, with N=12 parallel Frequency Adapters.

For block l:
    H_img  = ViTBlock_l(E_img)              (frozen block, LoRA on Q/K/V)
    H_freq = FreqAdapter_l(E_freq)          (LN -> down -> GELU -> up)
    E_img  <- H_img + H_freq
    E_freq <- E_freq + H_freq
then SAM's frozen neck maps E_img to e_i in R^{B x 256 x H/16 x W/16}.
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .frequency_extractor import FrequencyExtractor
from .lora_sam import inject_lora


class FrequencyAdapter(nn.Module):
    def __init__(self, dim: int = 768, reduction_ratio: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, dim // reduction_ratio)
        self.act = nn.GELU()
        self.up = nn.Linear(dim // reduction_ratio, dim)
        nn.init.zeros_(self.up.weight)  # start exactly from pretrained SAM features
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(self.act(self.down(self.norm(x))))


class FSAMEncoder(nn.Module):
    def __init__(self, sam_image_encoder: nn.Module, lora_rank: int = 4, reduction_ratio: int = 4,
                 grad_checkpoint: bool = False, fft_shift: bool = False):
        super().__init__()
        enc = inject_lora(sam_image_encoder, r=lora_rank, alpha=lora_rank)
        self.img_size = enc.img_size
        self.extractor = FrequencyExtractor(enc.patch_embed, enc.pos_embed, shift=fft_shift)
        self.blocks = enc.blocks
        self.neck = enc.neck
        dim = enc.pos_embed.shape[-1]
        self.freq_adapters = nn.ModuleList(
            [FrequencyAdapter(dim, reduction_ratio) for _ in range(len(self.blocks))])
        self.grad_checkpoint = grad_checkpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e_img, e_freq = self.extractor(x)
        for blk, adapter in zip(self.blocks, self.freq_adapters):
            if self.grad_checkpoint and self.training:
                h_img = checkpoint(blk, e_img, use_reentrant=False)
            else:
                h_img = blk(e_img)
            h_freq = adapter(e_freq)
            e_img = h_img + h_freq
            e_freq = e_freq + h_freq
        return self.neck(e_img.permute(0, 3, 1, 2))
