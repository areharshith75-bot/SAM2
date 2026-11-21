"""Cross-domain evaluation of trained FSAM checkpoints + report generation.

  python eval.py --run outputs/riga_BinRushed            # evaluate one run on its target domains
  python eval.py --generate_report                       # aggregate all runs into results/*.md

Model selection (no target data is ever used for selection):
  RIGA+   : the checkpoint at the stop epoch (`last_model.pth`), as in DAPSAM (no in-domain val split).
  Prostate: `best_model.pth`, chosen on the 10% in-domain validation cases of the source site.
"""
import argparse
import datetime
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.datasets import (PROSTATE_SITES, ProstateDataset, RIGADataset, prostate_collate_test,
                           riga_collate_test)
from models.fsam import FSAM
from utils.metrics import CaseDice, hard_dice, riga_targets

ROOT = os.path.dirname(os.path.abspath(__file__))
RIGA_TARGETS = ["MESSIDOR_Base1", "MESSIDOR_Base2", "MESSIDOR_Base3"]
PAPER_T1 = {  # FSAM rows of Table 1 (OD, OC) per target + average
    "BinRushed": {"MESSIDOR_Base1": (96.13, 88.12), "MESSIDOR_Base2": (96.29, 86.20),
                  "MESSIDOR_Base3": (96.32, 88.28), "Average": (96.25, 87.53)},
    "Magrabia": {"MESSIDOR_Base1": (96.28, 87.61), "MESSIDOR_Base2": (96.41, 90.07),
                 "MESSIDOR_Base3": (96.48, 88.20), "Average": (96.39, 88.63)},
}
PAPER_T2 = {"A": 86.16, "B": 82.52, "C": 74.98, "D": 86.44, "E": 83.14, "F": 83.19, "Average": 82.74}


def load_model(run, which="selected"):
    cfg = json.load(open(os.path.join(run, "config.json")))
    name = "last_model.pth" if (cfg["dataset"] == "riga" or which == "last") else "best_model.pth"
    ck = torch.load(os.path.join(run, name), map_location="cpu", weights_only=False)
    model = FSAM(cfg["img_size"], cfg["num_classes"], cfg["ckpt"], lora_rank=cfg["lora_rank"],
                 num_prototypes=cfg["num_prototypes"], similarity=cfg["similarity"],
                 freq_mode=cfg.get("freq_mode", "global"))
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    assert not unexpected, unexpected
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable <= set(ck["model"]), trainable - set(ck["model"])
    return model.cuda().eval(), cfg, name, ck.get("epoch")


TTA = False  # set by --tta: average the prediction with a horizontally flipped copy (extension)


@torch.no_grad()
def predict(model, x):
    x = x.cuda(non_blocking=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        p = torch.sigmoid(model(x).float())
        if TTA:
            p = 0.5 * (p + torch.sigmoid(model(x.flip(-1)).float()).flip(-1))
    return p.cpu()


def eval_riga(model, cfg):
    res = {}
    for t in RIGA_TARGETS:
        ds = RIGADataset(cfg["data_root"], t)
        od, oc = [], []
        for x, y, _ in DataLoader(ds, 8, collate_fn=riga_collate_test, num_workers=4):
            prob, tgt = predict(model, x), riga_targets(y)
            for i in range(len(prob)):
                od.append(hard_dice(prob[i, 0], tgt[i, 0]))
                oc.append(hard_dice(prob[i, 1], tgt[i, 1]))
        res[t] = {"OD": 100 * float(np.mean(od)), "OC": 100 * float(np.mean(oc)), "n_images": len(ds)}
    res["Average"] = {k: float(np.mean([res[t][k] for t in RIGA_TARGETS])) for k in ("OD", "OC")}
    return res


def eval_prostate(model, cfg):
    res, pooled = {}, []
    for letter, site in PROSTATE_SITES.items():
        if site == cfg["source_site"]:
            continue
        ds = ProstateDataset(cfg["data_root"], site)
        acc = CaseDice()
        for x, y, cid in DataLoader(ds, 16, collate_fn=prostate_collate_test, num_workers=4):
            acc.update(cid, predict(model, x), y)
        per_case = acc.per_case()
        pooled += list(per_case.values())
        res[site] = {"Dice": 100 * float(np.mean(list(per_case.values()))), "n_cases": len(per_case)}
    targets = [k for k in res]
    res["mean_over_cases"] = 100 * float(np.mean(pooled))      # DAPSAM protocol: all target cases pooled
    res["mean_over_domains"] = float(np.mean([res[t]["Dice"] for t in targets]))
    return res


def evaluate_run(run, which="selected"):
    model, cfg, ck_name, epoch = load_model(run, which)
    res = eval_riga(model, cfg) if cfg["dataset"] == "riga" else eval_prostate(model, cfg)
    out = {"run": os.path.basename(run), "checkpoint": ck_name, "epoch": epoch, "source": cfg["source"],
           "dataset": cfg["dataset"], "results": res, "evaluated_at": datetime.datetime.now().isoformat()}
    out["tta"] = TTA
    fname = "results.json" if which == "selected" else f"results_{which}.json"
    if TTA:
        fname = fname.replace(".json", "_tta.json")
    with open(os.path.join(run, fname), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return out


def _load_results(dataset, fname="results.json"):
    out = {}
    od = os.path.join(ROOT, "outputs")
    for r in sorted(os.listdir(od)) if os.path.isdir(od) else []:
        p = os.path.join(od, r, fname)
        cfgp = os.path.join(od, r, "config.json")
        if os.path.exists(cfgp) and json.load(open(cfgp)).get("freq_mode", "global") != "global":
            continue  # extension runs are reported by analysis/, not in the paper-reproduction tables
        if os.path.exists(p):
            d = json.load(open(p))
            if d["dataset"] == dataset:
                out[d["source"]] = d
    return out


def generate_report():
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    riga = _load_results("riga")
    L = ["# Table 1 reproduction — RIGA+ (DSC %, OD / OC)", "",
         "Reproduced = this repository, real training + evaluation; Paper = FSAM row of Table 1.", "",
         "| Source | Row | " + " | ".join(f"{t} OD | {t} OC" for t in RIGA_TARGETS + ["Average"]) + " |",
         "|---|---|" + "---|---|" * 4]
    for src in ["BinRushed", "Magrabia"]:
        cols = RIGA_TARGETS + ["Average"]
        if src in riga:
            r = riga[src]["results"]
            L.append(f"| {src} | Reproduced | " + " | ".join(f"{r[c]['OD']:.2f} | {r[c]['OC']:.2f}" for c in cols) + " |")
        else:
            L.append(f"| {src} | Reproduced | " + " | ".join("— | —" for _ in cols) + " |")
        L.append(f"| {src} | Paper | " + " | ".join(f"{PAPER_T1[src][c][0]:.2f} | {PAPER_T1[src][c][1]:.2f}" for c in cols) + " |")
        if src in riga:
            r = riga[src]["results"]
            L.append(f"| {src} | Δ | " + " | ".join(
                f"{r[c]['OD'] - PAPER_T1[src][c][0]:+.2f} | {r[c]['OC'] - PAPER_T1[src][c][1]:+.2f}" for c in cols) + " |")
    L += ["", "Checkpoints: " + ", ".join(f"{s}: {d['checkpoint']} (epoch {d['epoch']})" for s, d in riga.items())]
    open(os.path.join(ROOT, "results", "table1_riga_reproduction.md"), "w").write("\n".join(L) + "\n")

    pro = _load_results("prostate")
    pro_last = _load_results("prostate", "results_last.json")
    letters = list(PROSTATE_SITES)
    L = ["# Table 2 reproduction — multi-site prostate (DSC %)", "",
         "Each column: trained on that site (90% cases, best epoch on the 10% in-domain validation cases),",
         "tested on the 5 unseen sites. Main row = mean over all target cases (DAPSAM protocol);",
         "the per-domain-mean row averages the 5 site means instead.", "",
         "| Row | " + " | ".join(f"{l} ({PROSTATE_SITES[l]})" for l in letters) + " | Average |",
         "|---|" + "---|" * 7]

    def row(name, key, src=pro):
        vals = [src[l]["results"][key] if l in src else None for l in letters]
        got = [v for v in vals if v is not None]
        avg = f"{np.mean(got):.2f}" if len(got) == 6 else ("partial" if got else "—")
        return f"| {name} | " + " | ".join(f"{v:.2f}" if v is not None else "—" for v in vals) + f" | {avg} |"

    L.append(row("Reproduced", "mean_over_cases"))
    L.append(row("Reproduced (per-domain mean)", "mean_over_domains"))
    if pro_last:
        L.append(row("Supplementary: final epoch-160 checkpoint (not used for selection)", "mean_over_cases", pro_last))
    L.append("| Paper | " + " | ".join(f"{PAPER_T2[l]:.2f}" for l in letters) + f" | {PAPER_T2['Average']:.2f} |")
    L += ["", "## Per-target breakdown (DSC %)", ""]
    for l in letters:
        if l in pro:
            r = pro[l]["results"]
            L.append(f"- **{l} ({PROSTATE_SITES[l]})**, {pro[l]['checkpoint']} epoch {pro[l]['epoch']}: " +
                     ", ".join(f"{s} {v['Dice']:.2f} (n={v['n_cases']})" for s, v in r.items() if isinstance(v, dict)))
    open(os.path.join(ROOT, "results", "table2_prostate_reproduction.md"), "w").write("\n".join(L) + "\n")
    print(open(os.path.join(ROOT, "results", "table1_riga_reproduction.md")).read())
    print(open(os.path.join(ROOT, "results", "table2_prostate_reproduction.md")).read())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", nargs="*", default=[])
    p.add_argument("--checkpoint", default="selected", choices=["selected", "last"],
                   help="selected = protocol checkpoint (RIGA+: last, prostate: best in-domain val)")
    p.add_argument("--tta", action="store_true", help="flip test-time averaging (extension; writes *_tta.json)")
    p.add_argument("--generate_report", action="store_true")
    a = p.parse_args()
    TTA = a.tta
    for r in a.run:
        evaluate_run(r, a.checkpoint)
    if a.generate_report:
        generate_report()
