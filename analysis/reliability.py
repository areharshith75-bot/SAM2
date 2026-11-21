"""Extension 3 - Label-free failure warning (reliability score).

On an unseen hospital there are no labels, so a deployed model cannot know when it has failed. This module
gives every prediction a reliability score computed from the model alone:

  U = disagreement between the prediction and the prediction on the horizontally flipped image,
      measured as 1 - Dice(mask, flipped-back mask) over the predicted structure
      (RIGA+: per image, optic cup; prostate: per case, all labelled slices of the case).

We then check, on the labelled target domains, whether U ranks failures correctly:
  * Spearman correlation between U and the true Dice (should be strongly negative)
  * AUROC for detecting failures (true Dice < threshold; RIGA+ OC < 80, prostate < 70)
  * Dice of the most-flagged 20 % vs the remaining 80 %

  python -m analysis.reliability outputs/riga_BinRushed outputs/prostate_A ...
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import spearmanr

from analysis.common import load_targets
from eval import load_model

THRESH = {"riga": 80.0, "prostate": 70.0}


def auroc(score, positive):
    """P(score of a random positive > score of a random negative), ties counted half."""
    s, y = np.asarray(score), np.asarray(positive, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


@torch.no_grad()
def collect(model, cfg, batches):
    """Return per-unit (image or case) arrays: true Dice and uncertainty U."""
    riga = cfg["dataset"] == "riga"
    ch = 1 if riga else 0  # optic cup for RIGA+
    acc = defaultdict(lambda: np.zeros(5))  # inter_gt, sum_pred, sum_gt, inter_flip, sum_flip
    n = 0
    for dom, x, tgt, cid in batches:
        x = x.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = torch.sigmoid(model(x).float())
            pf = torch.sigmoid(model(x.flip(-1)).float()).flip(-1)
        m, mf = (p[:, ch] > 0.5).float().cpu(), (pf[:, ch] > 0.5).float().cpu()
        g = (tgt[:, ch] > 0).float() if riga else (tgt[:, 0] > 0).float()
        keys = [f"{dom}/{n + i}" for i in range(len(x))] if riga else cid
        n += len(x)
        for i, k in enumerate(keys):
            acc[k] += np.array([(m[i] * g[i]).sum(), m[i].sum(), g[i].sum(), (m[i] * mf[i]).sum(), mf[i].sum()])
    dice, unc = [], []
    for a in acc.values():
        dice.append(100 * (2 * a[0] + 1e-6) / (a[1] + a[2] + 1e-6))
        unc.append(1 - (2 * a[3] + 1e-6) / (a[1] + a[4] + 1e-6))
    return np.array(dice), np.array(unc)


def analyse(run):
    model, cfg, ck, _ = load_model(run)
    dice, unc = collect(model, cfg, load_targets(cfg))
    thr = THRESH[cfg["dataset"]]
    fail = dice < thr
    k = max(1, int(round(0.2 * len(unc))))
    order = np.argsort(-unc)
    out = {"run": os.path.basename(run.rstrip("/")), "unit": "image (optic cup)" if cfg["dataset"] == "riga" else "case",
           "n": int(len(dice)), "failure_threshold": thr, "n_failures": int(fail.sum()),
           "spearman_U_vs_dice": float(spearmanr(unc, dice).statistic), "auroc_failure": auroc(unc, fail),
           "dice_flagged_20pct": float(dice[order[:k]].mean()), "dice_rest_80pct": float(dice[order[k:]].mean()),
           "dice_all": float(dice.mean())}
    os.makedirs(os.path.join(run, "analysis"), exist_ok=True)
    json.dump(out | {"per_unit": {"dice": dice.tolist(), "U": unc.tolist()}},
              open(os.path.join(run, "analysis", "reliability.json"), "w"), indent=2)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--md", default="results/extension_reliability.md")
    a = p.parse_args()
    rows = ["| Run | Unit | n | Failures | Spearman(U, Dice) | Failure AUROC | Dice of 20 % most-flagged | Dice of the rest |",
            "|---|---|---|---|---|---|---|---|"]
    for r in a.runs:
        o = analyse(r)
        rows.append(f"| {o['run']} | {o['unit']} | {o['n']} | {o['n_failures']} (<{o['failure_threshold']:.0f}) | "
                    f"{o['spearman_U_vs_dice']:+.2f} | {o['auroc_failure']:.2f} | {o['dice_flagged_20pct']:.1f} | "
                    f"{o['dice_rest_80pct']:.1f} |")
        print(rows[-1], flush=True)
    os.makedirs(os.path.dirname(a.md), exist_ok=True)
    open(a.md, "w").write("# Extension 3 - Label-free failure warning\n\nU = 1 - Dice(prediction, prediction on the "
                          "flipped image). Computed without labels; labels are used only to check it.\n\n"
                          + "\n".join(rows) + "\n")
