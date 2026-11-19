"""DSC metrics following the DAPSAM evaluation code.

RIGA+: per-image hard Dice (sigmoid > 0.5) for OD = (label > 0) and OC = (label == 2), averaged
       over the images of a domain (`get_hard_dice`, smooth 1e-6).
Prostate: slices of a case are stacked into a volume and a single 3D Dice is computed per case
       (`calculate_metrics` -> `dice_metric`, smooth 1e-6); the domain score is the case mean.
"""
import numpy as np
import torch

SMOOTH = 1e-6


def hard_dice(prob: torch.Tensor, target: torch.Tensor, thr: float = 0.5) -> float:
    p = (prob > thr).double()
    t = (target > thr).double()
    return float((2 * (p * t).sum() + SMOOTH) / (p.sum() + t.sum() + SMOOTH))


def riga_targets(label: torch.Tensor) -> torch.Tensor:
    """label (B, H, W) with 0=BG, 1=disc rim, 2=cup  ->  (B, 2, H, W) [OD, OC]."""
    return torch.stack([(label > 0), (label == 2)], dim=1).float()


class CaseDice:
    """Accumulates slice predictions per case and returns volume-level Dice per case."""

    def __init__(self):
        self.inter, self.psum, self.tsum = {}, {}, {}

    def update(self, case_ids, prob: torch.Tensor, target: torch.Tensor, thr: float = 0.5):
        p = (prob > thr).double().flatten(1)
        t = (target > thr).double().flatten(1)
        for i, cid in enumerate(case_ids):
            self.inter[cid] = self.inter.get(cid, 0.0) + float((p[i] * t[i]).sum())
            self.psum[cid] = self.psum.get(cid, 0.0) + float(p[i].sum())
            self.tsum[cid] = self.tsum.get(cid, 0.0) + float(t[i].sum())

    def per_case(self):
        return {c: (2 * self.inter[c] + SMOOTH) / (self.psum[c] + self.tsum[c] + SMOOTH) for c in self.inter}

    def mean(self):
        return float(np.mean(list(self.per_case().values())))
