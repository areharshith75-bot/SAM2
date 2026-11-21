"""Extension 1 - Frequency Reliance Analyzer.

How much does a trained FSAM actually use the *content* of its frequency stream? The image stream is kept
intact and only the frequency-stream input is replaced:
  own      : the image's own spectrum (normal inference)
  shuffled : the spectrum of a different test image of the same batch
  zeros    : an all-zero input (the stream then carries only SAM's positional embedding)
  noise    : Gaussian noise
Frequency Reliance Score (FRS) = Dice(own) - Dice(shuffled), averaged over structures. FRS ~ 0 means the
frequency adapters act as a fixed, input-independent offset.

  python -m analysis.frequency_reliance outputs/riga_BinRushed outputs/prostate_A ...
"""
import argparse
import json
import os

import numpy as np
import torch

from analysis.common import load_targets, score
from eval import load_model

PROBES = {
    "own": None,
    "shuffled": lambda x, g: x.roll(1, 0) if len(x) > 1 else x.flip(-1).flip(-2),
    "zeros": lambda x, g: torch.zeros_like(x),
    "noise": lambda x, g: torch.randn(x.shape, device=x.device, generator=g),
}


def analyse(run):
    model, cfg, ck, _ = load_model(run)
    batches = load_targets(cfg)
    res = {k: score(model, cfg, batches, freq_tf=f) for k, f in PROBES.items()}
    keys = list(res["own"])
    frs = float(np.mean([res["own"][k] - res["shuffled"][k] for k in keys]))
    out = {"run": os.path.basename(run.rstrip("/")), "freq_mode": cfg.get("freq_mode", "global"),
           "checkpoint": ck, "probes": res, "FRS": frs}
    os.makedirs(os.path.join(run, "analysis"), exist_ok=True)
    json.dump(out, open(os.path.join(run, "analysis", "frequency_reliance.json"), "w"), indent=2)
    return out


def fmt(r):
    return " / ".join(f"{v:.2f}" for v in r.values())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--md", default="results/extension_frequency_reliance.md")
    a = p.parse_args()
    rows = ["| Run | Frequency input | Own spectrum | Another image's spectrum | Zeros | Noise | FRS |",
            "|---|---|---|---|---|---|---|"]
    for r in a.runs:
        o = analyse(r)
        pr = o["probes"]
        rows.append(f"| {o['run']} | {o['freq_mode']} | {fmt(pr['own'])} | {fmt(pr['shuffled'])} | "
                    f"{fmt(pr['zeros'])} | {fmt(pr['noise'])} | {o['FRS']:+.2f} |")
        print(rows[-1], flush=True)
    os.makedirs(os.path.dirname(a.md), exist_ok=True)
    open(a.md, "w").write("# Extension 1 - Frequency Reliance Analyzer\n\nDice % on unseen target domains "
                          "(RIGA+: OD / OC; prostate: pooled case Dice). Only the frequency-stream input is "
                          "replaced; the image stream is untouched. FRS = own - another image's spectrum.\n\n"
                          + "\n".join(rows) + "\n")
