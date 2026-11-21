"""Extension 2 - Corruption-robustness benchmark.

Applies realistic image corruptions to the unseen target images at test time and reports Dice when the
corruption reaches (a) both streams (realistic), (b) only the image stream, (c) only the frequency stream.
(b) vs (c) shows which stream carries the sensitivity.

Corruptions (on the z-scored input, then re-standardised):
  noise    : additive Gaussian, sigma = 0.15 / 0.3 / 0.6
  blur     : Gaussian blur, sigma = 1 / 2 / 4 px
  shading  : smooth multiplicative field (uneven illumination / MRI bias field), strength 0.25 / 0.5 / 0.8
  contrast : gamma-like contrast compression |x|^g, g = 0.8 / 0.6 / 0.4

  python -m analysis.corruption_benchmark outputs/riga_BinRushed ...
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from analysis.common import load_targets, score
from eval import load_model


def gblur(x, sigma):
    k = int(4 * sigma) | 1
    t = torch.arange(k, device=x.device) - k // 2
    w = torch.exp(-t ** 2 / (2 * sigma ** 2)); w = (w / w.sum()).to(x.dtype)
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (k // 2,) * 4, mode="reflect"), w.view(1, 1, 1, k).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(x, w.view(1, 1, k, 1).repeat(c, 1, 1, 1), groups=c)


def shading(x, strength, g):
    b, _, h, w = x.shape
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=x.device), torch.linspace(-1, 1, w, device=x.device),
                            indexing="ij")
    th = torch.rand(b, 1, 1, 1, device=x.device, generator=g) * 2 * math.pi  # random direction per image
    field = 1 + strength * (torch.cos(th) * xx + torch.sin(th) * yy + 0.5 * (xx ** 2 - yy ** 2))
    return (x - x.amin((2, 3), keepdim=True)) * field.clamp(min=0.05)


CORRUPTIONS = {
    "noise": [lambda x, g, s=s: x + s * torch.randn(x.shape, device=x.device, generator=g) for s in (0.15, 0.3, 0.6)],
    "blur": [lambda x, g, s=s: gblur(x, s) for s in (1.0, 2.0, 4.0)],
    "shading": [lambda x, g, s=s: shading(x, s, g) for s in (0.25, 0.5, 0.8)],
    "contrast": [lambda x, g, s=s: torch.sign(x) * x.abs() ** s for s in (0.8, 0.6, 0.4)],
}
STREAMS = {"both": ("img", "freq"), "image-only": ("img",), "frequency-only": ("freq",)}


def main_metric(r):
    return r["OC"] if "OC" in r else r["Dice"]


def analyse(run):
    model, cfg, ck, _ = load_model(run)
    batches = load_targets(cfg)
    clean = score(model, cfg, batches)
    res = {"clean": clean}
    for name, levels in CORRUPTIONS.items():
        for lvl, f in enumerate(levels, 1):
            for st, where in STREAMS.items():
                ident = lambda x, g: x  # noqa: E731 - frequency stream sees the clean image
                res[f"{name}-{lvl}/{st}"] = score(model, cfg, batches,
                                                  img_tf=f if "img" in where else None,
                                                  freq_tf=(f if "freq" in where else ident) if st != "both" else None)
    out = {"run": os.path.basename(run.rstrip("/")), "freq_mode": cfg.get("freq_mode", "global"),
           "checkpoint": ck, "metric": "OC" if "OC" in clean else "Dice", "results": res}
    # mean drop vs clean over all corruptions & severities, per stream (main structure: OC / prostate)
    out["mean_drop"] = {st: float(np.mean([main_metric(clean) - main_metric(v) for k, v in res.items()
                                           if k.endswith("/" + st)])) for st in STREAMS}
    os.makedirs(os.path.join(run, "analysis"), exist_ok=True)
    json.dump(out, open(os.path.join(run, "analysis", "corruption_benchmark.json"), "w"), indent=2)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--md", default="results/extension_corruption_benchmark.md")
    a = p.parse_args()
    L = ["# Extension 2 - Corruption-robustness benchmark", "",
         "Main structure Dice % (RIGA+: optic cup; prostate: pooled case Dice) on unseen targets, averaged over 3 "
         "severities. 'Drop' = clean minus corrupted, averaged over all 12 corruption settings.", ""]
    for r in a.runs:
        o = analyse(r)
        res, m = o["results"], o["metric"]
        L += [f"## {o['run']} (frequency input: {o['freq_mode']}, metric: {m})", "",
              f"Clean: {main_metric(res['clean']):.2f}", "",
              "| Corruption | Both streams | Image stream only | Frequency stream only |", "|---|---|---|---|"]
        for name in CORRUPTIONS:
            L.append(f"| {name} | " + " | ".join(
                f"{np.mean([main_metric(res[f'{name}-{l}/{st}']) for l in (1, 2, 3)]):.2f}" for st in STREAMS) + " |")
        L += ["| **mean drop** | " + " | ".join(f"{o['mean_drop'][st]:.2f}" for st in STREAMS) + " |", ""]
        print("\n".join(L[-9:]), flush=True)
    os.makedirs(os.path.dirname(a.md), exist_ok=True)
    open(a.md, "w").write("\n".join(L) + "\n")
