"""Full FSAM model: FSAM encoder -> automated prompt generator -> trainable SAM mask decoder."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fast_attention import enable_fast_attention
from .fsam_encoder import FSAMEncoder
from .prompt_generator import AutomatedPromptGenerator
from .sam_builder import build_sam_parts


class FSAM(nn.Module):
    """Outputs `num_classes` mask logits at input resolution (one sigmoid channel per structure,
    as in DAPSAM: RIGA+ -> [OD, OC]; prostate -> [prostate])."""

    def __init__(self, img_size: int, num_classes: int, checkpoint: str | None = None,
                 lora_rank: int = 4, num_prototypes: int = 256, similarity: str = "cosine",
                 grad_checkpoint: bool = False, fft_shift: bool = False, fast_attention: bool = True,
                 freq_mode: str = "global"):
        super().__init__()
        image_encoder, prompt_encoder, mask_decoder = build_sam_parts(img_size, num_classes, checkpoint)
        if fast_attention:  # fused SDPA, numerically equivalent (tests/test_fast_attention.py)
            enable_fast_attention(image_encoder)
        self.img_size = img_size
        self.num_classes = num_classes
        self.encoder = FSAMEncoder(image_encoder, lora_rank=lora_rank, grad_checkpoint=grad_checkpoint,
                                   fft_shift=fft_shift, freq_mode=freq_mode)
        self.prompt_generator = AutomatedPromptGenerator(256, num_prototypes, similarity)
        self.mask_decoder = mask_decoder  # trainable (IoU head is unused by the loss -> frozen)
        for p in self.mask_decoder.iou_prediction_head.parameters():
            p.requires_grad = False
        emb = img_size // 16
        # SAM's pretrained dense positional encoding for the decoder (kept frozen, as a buffer)
        with torch.no_grad():
            pe = prompt_encoder.pe_layer((emb, emb)).unsqueeze(0)
        self.register_buffer("image_pe", pe, persistent=False)

    def decode(self, emb: torch.Tensor, dense: torch.Tensor):
        md = self.mask_decoder
        b, c, h, w = emb.shape
        tokens = torch.cat([md.iou_token.weight, md.mask_tokens.weight], 0).unsqueeze(0).expand(b, -1, -1)
        src = emb + dense
        pos = self.image_pe.expand(b, -1, -1, -1).to(src.dtype)
        hs, src = md.transformer(src, pos, tokens)
        mask_tokens_out = hs[:, 1:1 + md.num_mask_tokens]
        up = md.output_upscaling(src.transpose(1, 2).reshape(b, c, h, w))
        hyper = torch.stack([mlp(mask_tokens_out[:, i]) for i, mlp in enumerate(md.output_hypernetworks_mlps)], 1)
        bb, cc, hh, ww = up.shape
        masks = (hyper @ up.view(bb, cc, hh * ww)).view(bb, -1, hh, ww)
        return masks, md.iou_prediction_head(hs[:, 0])

    def forward(self, x: torch.Tensor, x_freq: torch.Tensor | None = None):
        emb = self.encoder(x, x_freq)
        dense = self.prompt_generator(emb)
        low_res, _ = self.decode(emb, dense)
        return F.interpolate(low_res, (x.shape[-2], x.shape[-1]), mode="bilinear", align_corners=False)

    def trainable_parameter_breakdown(self):
        groups = {"lora": 0, "freq_adapters": 0, "prompt_generator": 0, "mask_decoder": 0, "other": 0}
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            key = ("lora" if "lora_" in n else "freq_adapters" if "freq_adapters" in n else
                   "prompt_generator" if n.startswith("prompt_generator") else
                   "mask_decoder" if n.startswith("mask_decoder") else "other")
            groups[key] += p.numel()
        groups["total_trainable"] = sum(groups.values())
        groups["total"] = sum(p.numel() for p in self.parameters())
        return groups
