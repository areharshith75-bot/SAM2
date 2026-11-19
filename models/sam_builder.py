"""Build SAM ViT-B at a non-native input size (512 / 384) from the official 1024px checkpoint.

Mirrors DAPSAM/SAMed `load_from`: the absolute `pos_embed` and the relative position tables of the
global-attention blocks are bilinearly resized to the new token grid; mask-decoder heads whose
shape depends on the number of output masks (`mask_tokens`, `output_hypernetworks_mlps`,
`iou_prediction_head`) are re-initialised because FSAM predicts `num_classes` masks.
"""
from functools import partial

import torch
import torch.nn.functional as F
from segment_anything.modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, TwoWayTransformer

PATCH = 16
VIT_B = dict(embed_dim=768, depth=12, num_heads=12, global_attn_indexes=(2, 5, 8, 11))
PROMPT_DIM = 256
DECODER_SKIP = ("mask_tokens", "output_hypernetworks_mlps", "iou_prediction_head")


def build_sam_parts(img_size: int, num_classes: int, checkpoint: str | None = None):
    image_encoder = ImageEncoderViT(
        depth=VIT_B["depth"], embed_dim=VIT_B["embed_dim"], img_size=img_size, mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), num_heads=VIT_B["num_heads"],
        patch_size=PATCH, qkv_bias=True, use_rel_pos=True,
        global_attn_indexes=VIT_B["global_attn_indexes"], window_size=14, out_chans=PROMPT_DIM,
    )
    emb = img_size // PATCH
    prompt_encoder = PromptEncoder(  # only its pretrained dense positional encoding is used
        embed_dim=PROMPT_DIM, image_embedding_size=(emb, emb),
        input_image_size=(img_size, img_size), mask_in_chans=16,
    )
    mask_decoder = MaskDecoder(
        num_multimask_outputs=num_classes - 1,  # => num_mask_tokens == num_classes
        transformer=TwoWayTransformer(depth=2, embedding_dim=PROMPT_DIM, mlp_dim=2048, num_heads=8),
        transformer_dim=PROMPT_DIM, iou_head_depth=3, iou_head_hidden_dim=256,
    )
    if checkpoint is not None:
        sd = torch.load(checkpoint, map_location="cpu", weights_only=True)
        _load(image_encoder, sd, "image_encoder.", emb)
        _load(prompt_encoder, sd, "prompt_encoder.", emb)
        _load(mask_decoder, sd, "mask_decoder.", emb, skip=DECODER_SKIP)
    return image_encoder, prompt_encoder, mask_decoder


def _load(module, sd, prefix, token_size, skip=()):
    own = module.state_dict()
    new = {}
    for k, v in sd.items():
        if not k.startswith(prefix):
            continue
        name = k[len(prefix):]
        if name not in own or any(s in name for s in skip):
            continue
        if name == "pos_embed" and v.shape[1] != token_size:
            v = F.interpolate(v.permute(0, 3, 1, 2), (token_size, token_size), mode="bilinear",
                              align_corners=False).permute(0, 2, 3, 1)
        elif "rel_pos" in name and v.shape != own[name].shape:
            # only global-attention blocks change size (2*token-1); windowed blocks stay 2*14-1
            v = F.interpolate(v[None, None], own[name].shape, mode="bilinear", align_corners=False)[0, 0]
        assert v.shape == own[name].shape, (k, v.shape, own[name].shape)
        new[name] = v
    missing = [k for k in own if k not in new and not any(s in k for s in skip)]
    assert not missing, f"unloaded {prefix} keys: {missing}"
    module.load_state_dict(new, strict=False)
