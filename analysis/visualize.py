"""Qualitative figure for presentations: image | ground truth | prediction | flip-disagreement (failure warning).

Picks examples from the unseen target domains: the most confident and the most-flagged ones.

  python -m analysis.visualize outputs/prostate_E --n 3
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageDraw

from analysis.common import load_targets
from eval import load_model

COL = {"gt": (40, 200, 90), "pred": (230, 120, 30), "unc": (220, 40, 40)}


def to_rgb(x):
    x = x[:3].float().cpu().numpy() if x.shape[0] >= 3 else x[:1].repeat(3, 0).cpu().numpy()
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return (x.transpose(1, 2, 0) * 255).astype(np.uint8)


def overlay(rgb, mask, color, alpha=0.45):
    out = rgb.astype(np.float32).copy()
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * np.array(color, np.float32)
    return out.astype(np.uint8)


@torch.no_grad()
def main(run, n, size=256):
    model, cfg, _, _ = load_model(run)
    riga = cfg["dataset"] == "riga"
    ch = 1 if riga else 0
    items = []
    for dom, x, tgt, cid in load_targets(cfg):
        xg = x.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = torch.sigmoid(model(xg).float())
            pf = torch.sigmoid(model(xg.flip(-1)).float()).flip(-1)
        for i in range(len(x)):
            g = (tgt[i, ch] > 0).numpy() if riga else (tgt[i, 0] > 0).numpy()
            if g.sum() < 50:
                continue
            m, mf = (p[i, ch] > 0.5).cpu().numpy(), (pf[i, ch] > 0.5).cpu().numpy()
            dice = 2 * (m & g).sum() / (m.sum() + g.sum() + 1e-6)
            u = 1 - 2 * (m & mf).sum() / (m.sum() + mf.sum() + 1e-6)
            items.append((u, dice, dom, x[i], g, m, m ^ mf))
    items.sort(key=lambda t: t[0])
    picks = items[:n] + items[-n:]
    W = size * 4
    canvas = Image.new("RGB", (W, (size + 22) * len(picks) + 26), "white")
    d = ImageDraw.Draw(canvas)
    for j, t in enumerate(["image", "ground truth", "prediction", "flip disagreement"]):
        d.text((j * size + 6, 6), t, fill="black")
    for r, (u, dice, dom, x, g, m, dis) in enumerate(picks):
        rgb = to_rgb(x)
        tiles = [rgb, overlay(rgb, g, COL["gt"]), overlay(rgb, m, COL["pred"]), overlay(rgb, dis, COL["unc"], 0.8)]
        y = 26 + r * (size + 22)
        for j, t in enumerate(tiles):
            canvas.paste(Image.fromarray(t).resize((size, size)), (j * size, y))
        tag = "most confident" if r < n else "most flagged"
        d.text((6, y + size + 4), f"{dom} | {tag} | warning U={u:.3f} | true Dice={100 * dice:.1f}", fill="black")
    os.makedirs("results/figures", exist_ok=True)
    out = f"results/figures/{os.path.basename(run.rstrip('/'))}_examples.png"
    canvas.save(out)
    print(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--n", type=int, default=3)
    a = p.parse_args()
    for r in a.runs:
        main(r, a.n)
