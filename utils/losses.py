"""FSAM Eq. 5: L = (1 - lambda) L_CE + lambda L_Dice, lambda = 0.8.

Matches the DAPSAM/SAMed reference: every output channel is an independent sigmoid map
(RIGA+: OD, OC; prostate: gland), CE is binary cross-entropy, Dice is the batch-level soft Dice
`1 - (2*sum(p*y)+1)/(sum p + sum y + 1)`, and the per-channel losses are summed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(prob: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    inter = (prob * target).sum()
    return 1 - (2 * inter + smooth) / (prob.sum() + target.sum() + smooth)


class HybridSegmentationLoss(nn.Module):
    def __init__(self, lambda_dice: float = 0.8):
        super().__init__()
        self.lam = lambda_dice

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        """logits, target: (B, C, H, W); target in {0,1}. Computed in float32."""
        logits, target = logits.float(), target.float()
        total = ce_sum = dice_sum = 0.0
        for c in range(logits.shape[1]):
            ce = F.binary_cross_entropy_with_logits(logits[:, c], target[:, c])
            dice = soft_dice_loss(torch.sigmoid(logits[:, c]), target[:, c])
            total = total + (1 - self.lam) * ce + self.lam * dice
            ce_sum, dice_sum = ce_sum + ce.detach(), dice_sum + dice.detach()
        return total, ce_sum, dice_sum
