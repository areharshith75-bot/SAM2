"""FSAM Sec. 2.2: memory-based automated prompt generator (from DAPSAM, Wei et al. 2024).

p_i      = GAP(e_i) + GMP(e_i)                                    (B, C)
s_i      = cos(p_i, M_j)  for the N_p learnable prototypes M      (Eq. 3)
p_hat_i  = sum_j softmax(s_i)_j M_j                               (Eq. 4)
A_i      = cos(p_hat_i, e_i[:, :, y, x])                          (B, 1, h, w)
prompt   = Conv1x1([e_i, p_hat_i (broadcast), A_i])               (B, C, h, w)

`similarity="dot"` reproduces the DAPSAM reference code, which uses an un-normalised dot product
in the memory read-out; the default follows the FSAM paper (cosine, Eq. 3).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AutomatedPromptGenerator(nn.Module):
    def __init__(self, embed_dim: int = 256, num_prototypes: int = 256, similarity: str = "cosine"):
        super().__init__()
        assert similarity in ("cosine", "dot")
        self.similarity = similarity
        self.memory = nn.Parameter(torch.empty(num_prototypes, embed_dim))
        stdv = 1.0 / math.sqrt(embed_dim)
        nn.init.uniform_(self.memory, -stdv, stdv)  # as DAPSAM MemoryUnit
        self.fuse_conv = nn.Conv2d(2 * embed_dim + 1, embed_dim, kernel_size=1)

    def refine(self, p: torch.Tensor):
        if self.similarity == "cosine":
            s = F.normalize(p, dim=-1, eps=1e-8) @ F.normalize(self.memory, dim=-1, eps=1e-8).t()
        else:
            s = p @ self.memory.t()
        alpha = s.softmax(dim=-1)  # (B, N_p)
        return alpha @ self.memory, alpha

    def forward(self, e: torch.Tensor):
        b, c, h, w = e.shape
        p = F.adaptive_avg_pool2d(e, 1).flatten(1) + F.adaptive_max_pool2d(e, 1).flatten(1)
        p_hat, _ = self.refine(p)
        p_map = p_hat[:, :, None, None].expand(b, c, h, w)
        act = F.cosine_similarity(p_map, e, dim=1, eps=1e-7).unsqueeze(1)
        return self.fuse_conv(torch.cat([e, p_map, act], dim=1))
