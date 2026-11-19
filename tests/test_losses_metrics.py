import torch

from utils.losses import HybridSegmentationLoss, soft_dice_loss
from utils.metrics import CaseDice, hard_dice, riga_targets


def test_hybrid_loss_matches_dapsam_formula():
    torch.manual_seed(0)
    logits, y = torch.randn(2, 2, 16, 16), (torch.rand(2, 2, 16, 16) > 0.5).float()
    total, _, _ = HybridSegmentationLoss(0.8)(logits, y)
    ref = 0
    for c in range(2):
        p = torch.sigmoid(logits[:, c])
        bce = torch.nn.BCELoss()(p, y[:, c])
        dice = 1 - (2 * (p * y[:, c]).sum() + 1) / (p.sum() + y[:, c].sum() + 1)
        ref = ref + 0.2 * bce + 0.8 * dice
    assert torch.allclose(total, ref, atol=1e-5)
    assert soft_dice_loss(y, y) < 0.01


def test_metrics():
    lab = torch.tensor([[[0, 1, 2, 2]]])
    t = riga_targets(lab)
    assert t[0, 0].tolist() == [[0, 1, 1, 1]] and t[0, 1].tolist() == [[0, 0, 1, 1]]
    assert abs(hard_dice(torch.tensor([0.9, 0.9, 0.1]), torch.tensor([1., 0., 0.])) - 2 / 3) < 1e-5
    acc = CaseDice()  # volume Dice aggregates over slices, not mean of slice Dice
    acc.update(["c1", "c1"], torch.tensor([[1., 0.], [1., 1.]]), torch.tensor([[1., 0.], [0., 0.]]))
    assert abs(acc.per_case()["c1"] - 2 * 1 / (3 + 1)) < 1e-5
