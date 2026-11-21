"""Shared helpers for the extension analyses: score a trained run on its unseen target domains while
optionally transforming the image-stream input and/or the frequency-stream input independently."""
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.datasets import (PROSTATE_SITES, ProstateDataset, RIGADataset, prostate_collate_test,
                           riga_collate_test)
from eval import RIGA_TARGETS
from utils.metrics import CaseDice, hard_dice, riga_targets


def restandardise(x):
    m, s = x.mean((2, 3), keepdim=True), x.std((2, 3), keepdim=True)
    return (x - m) / (s + 1e-8)


def load_targets(cfg):
    """List of (domain, x, target, case_ids) batches, kept in memory so many variants can be scored fast."""
    out = []
    if cfg["dataset"] == "riga":
        for t in RIGA_TARGETS:
            for x, y, _ in DataLoader(RIGADataset(cfg["data_root"], t), 8, collate_fn=riga_collate_test, num_workers=4):
                out.append((t, x, riga_targets(y), None))
    else:
        for site in PROSTATE_SITES.values():
            if site == cfg["source_site"]:
                continue
            for x, y, cid in DataLoader(ProstateDataset(cfg["data_root"], site), 16,
                                        collate_fn=prostate_collate_test, num_workers=4):
                out.append((site, x, y, cid))
    return out


@torch.no_grad()
def score(model, cfg, batches, img_tf=None, freq_tf=None, seed=0):
    """img_tf / freq_tf: callables on the (z-scored, GPU) batch; None = unchanged.
    Returns RIGA+ {'OD','OC'} (mean over the 3 targets of per-image Dice) or prostate {'Dice'} (all target
    cases pooled, DAPSAM protocol)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    riga = cfg["dataset"] == "riga"
    per_dom, acc = {}, CaseDice()
    for dom, x, tgt, cid in batches:
        x = x.cuda()
        xi = restandardise(img_tf(x, g)) if img_tf else x
        xf = restandardise(freq_tf(x, g)) if freq_tf else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = torch.sigmoid(model(xi, xf).float()).cpu()
        if riga:
            d = per_dom.setdefault(dom, ([], []))
            for i in range(len(p)):
                d[0].append(hard_dice(p[i, 0], tgt[i, 0])); d[1].append(hard_dice(p[i, 1], tgt[i, 1]))
        else:
            acc.update(cid, p, tgt)
    if riga:
        return {k: 100 * float(np.mean([np.mean(v[j]) for v in per_dom.values()])) for j, k in enumerate(["OD", "OC"])}
    return {"Dice": 100 * float(np.mean(list(acc.per_case().values())))}
