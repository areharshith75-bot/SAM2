"""Train FSAM for single-source domain generalisation (RIGA+ or multi-site prostate).

Schedule / optimisation follow the paper + the DAPSAM/SAMed reference trainer:
AdamW(lr=5e-4, wd=0.1, betas=(0.9,0.999)), linear per-iteration warm-up for `warmup_period`
iterations (25 RIGA+, 250 prostate), then poly decay lr*(1 - it/max_it)^0.9 with
max_it = max_epochs * iters_per_epoch; training stops at `stop_epoch` (160) of `max_epochs` (200).
"""
import argparse
import json
import logging
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.datasets import (PROSTATE_SITES, ProstateDataset, RIGADataset, prostate_collate_test,
                           prostate_collate_train, prostate_split, riga_collate_train)
from models.fsam import FSAM
from utils.losses import HybridSegmentationLoss
from utils.metrics import CaseDice, riga_targets

ROOT = os.path.dirname(os.path.abspath(__file__))


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["riga", "prostate"], required=True)
    p.add_argument("--source", required=True, help="RIGA domain (BinRushed/Magrabia) or prostate site letter A-F")
    p.add_argument("--data_root", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--ckpt", default=os.path.join(ROOT, "pretrained", "sam_vit_b_01ec64.pth"))
    p.add_argument("--img_size", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--base_lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_period", type=int, default=None)
    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--stop_epoch", type=int, default=160)
    p.add_argument("--dice_param", type=float, default=0.8)
    p.add_argument("--lora_rank", type=int, default=4)
    p.add_argument("--num_prototypes", type=int, default=256)
    p.add_argument("--similarity", default="cosine", choices=["cosine", "dot"])
    p.add_argument("--eval_every", type=int, default=5, help="prostate in-domain validation interval (epochs)")
    p.add_argument("--num_workers", type=int, default=12)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--grad_checkpoint", action="store_true")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--max_iters_debug", type=int, default=0, help="stop after N iterations (smoke test only)")
    a = p.parse_args(argv)
    riga = a.dataset == "riga"
    a.img_size = a.img_size or (512 if riga else 384)
    a.warmup_period = a.warmup_period or (25 if riga else 250)
    a.num_classes = 2 if riga else 1
    a.data_root = a.data_root or os.path.join(ROOT, "data_processed", "RIGA_plus" if riga else "Prostate")
    if not riga:
        a.source_site = PROSTATE_SITES[a.source]
    a.out = a.out or os.path.join(ROOT, "outputs", f"{a.dataset}_{a.source}")
    return a


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def trainable_state(model):
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in names}


@torch.no_grad()
def validate_prostate(model, loader, amp):
    model.eval()
    acc = CaseDice()
    for x, y, cid in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(x.cuda(non_blocking=True))
        acc.update(cid, torch.sigmoid(logits.float()).cpu(), y)
    model.train()
    return acc.mean()


def main(argv=None):
    a = get_args(argv)
    os.makedirs(a.out, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%m-%d %H:%M:%S",
                        handlers=[logging.FileHandler(os.path.join(a.out, "train.log")), logging.StreamHandler(sys.stdout)])
    log = logging.getLogger()
    seed_all(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    amp = not a.no_amp

    # ---------------- data (whole datasets live in RAM as numpy arrays)
    g = torch.Generator().manual_seed(a.seed)
    common = dict(num_workers=a.num_workers, pin_memory=True, persistent_workers=a.num_workers > 0)
    val_loader = None
    if a.dataset == "riga":
        ds = RIGADataset(a.data_root, a.source)
        loader = DataLoader(ds, a.batch_size, shuffle=True, drop_last=False, collate_fn=riga_collate_train,
                            generator=g, **common)
        a.train_items = len(ds)
    else:
        tr_cases, va_cases = prostate_split(a.data_root, a.source_site, 0.1, a.seed)
        ds = ProstateDataset(a.data_root, a.source_site, tr_cases, batch_size=a.batch_size)
        loader = DataLoader(ds, a.batch_size, shuffle=True, drop_last=False, collate_fn=prostate_collate_train,
                            generator=g, **common)
        va = ProstateDataset(a.data_root, a.source_site, va_cases)
        val_loader = DataLoader(va, 16, shuffle=False, collate_fn=prostate_collate_test, num_workers=2)
        a.train_cases, a.val_cases = tr_cases.tolist(), va_cases.tolist()
        a.train_items = int(np.isin(np.load(os.path.join(a.data_root, f"{a.source_site}.npz"))["case"], tr_cases).sum())
        log.info(f"source {a.source_site}: train cases {len(tr_cases)} ({a.train_items} slices), "
                 f"val cases {len(va_cases)} ({len(va)} slices); pool replicated to {len(ds)}")

    # ---------------- model / optimiser
    model = FSAM(a.img_size, a.num_classes, a.ckpt, lora_rank=a.lora_rank, num_prototypes=a.num_prototypes,
                 similarity=a.similarity, grad_checkpoint=a.grad_checkpoint).cuda().train()
    bd = model.trainable_parameter_breakdown()
    log.info(f"params: {bd}")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.base_lr / a.warmup_period, betas=(0.9, 0.999), weight_decay=a.weight_decay)
    crit = HybridSegmentationLoss(a.dice_param)

    iters_per_epoch = len(loader)
    max_iterations = a.max_epochs * iters_per_epoch
    a.iters_per_epoch, a.max_iterations = iters_per_epoch, max_iterations
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump(vars(a) | {"param_breakdown": bd}, f, indent=2)
    log.info(f"{iters_per_epoch} it/epoch, max_iterations {max_iterations}, stop at epoch {a.stop_epoch}")

    # ---------------- resume
    start_epoch, it, best, history = 0, 0, -1.0, []
    resume = os.path.join(a.out, "resume.pth")
    if os.path.exists(resume):
        st = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"], strict=False)
        opt.load_state_dict(st["opt"])
        start_epoch, it, best, history = st["epoch"] + 1, st["iter"], st["best"], st["history"]
        torch.set_rng_state(st["rng"])
        log.info(f"resumed from epoch {st['epoch']} (iter {it}, best {best:.4f})")

    writer = SummaryWriter(os.path.join(a.out, "tb"))
    lr = a.base_lr / a.warmup_period
    for epoch in range(start_epoch, a.max_epochs):
        t0, run_loss = time.time(), []
        for x, y in loader:
            x = x.cuda(non_blocking=True)
            y = riga_targets(y).cuda(non_blocking=True) if a.dataset == "riga" else y.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = model(x)
            loss, l_ce, l_dice = crit(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            # DAPSAM/SAMed schedule (lr is updated after the step, per iteration)
            if it < a.warmup_period:
                lr = a.base_lr * ((it + 1) / a.warmup_period)
            else:
                lr = a.base_lr * (1.0 - (it - a.warmup_period) / max_iterations) ** 0.9
            for gp in opt.param_groups:
                gp["lr"] = lr
            it += 1
            run_loss.append(loss.item())
            if it % 50 == 0:
                writer.add_scalar("train/loss", loss.item(), it)
                writer.add_scalar("train/ce", float(l_ce), it)
                writer.add_scalar("train/dice", float(l_dice), it)
                writer.add_scalar("train/lr", lr, it)
            if a.max_iters_debug and it >= a.max_iters_debug:
                break
        dt = time.time() - t0
        rec = {"epoch": epoch, "iter": it, "loss": float(np.mean(run_loss)), "lr": lr, "sec": round(dt, 1)}
        last_epoch = epoch >= a.stop_epoch - 1 or epoch >= a.max_epochs - 1 or bool(a.max_iters_debug)

        if val_loader is not None and ((epoch + 1) % a.eval_every == 0 or last_epoch):
            vd = validate_prostate(model, val_loader, amp)
            rec["val_dice"] = vd
            writer.add_scalar("val/dice", vd, epoch)
            if vd > best:
                best = vd
                torch.save({"model": trainable_state(model), "epoch": epoch, "val_dice": vd},
                           os.path.join(a.out, "best_model.pth"))
                rec["best"] = True
        history.append(rec)
        mem = torch.cuda.max_memory_allocated() / 2**30
        log.info(f"epoch {epoch} loss {rec['loss']:.4f} lr {lr:.2e} {dt:.0f}s ({dt / max(1, len(run_loss)):.3f}s/it) "
                 f"peak {mem:.1f}GiB" + (f" val_dice {rec['val_dice']:.4f} (best {best:.4f})" if "val_dice" in rec else ""))

        if last_epoch:
            torch.save({"model": trainable_state(model), "epoch": epoch}, os.path.join(a.out, "last_model.pth"))
            log.info(f"stopped at epoch {epoch}; saved last_model.pth")
            break
        torch.save({"model": trainable_state(model), "opt": opt.state_dict(), "epoch": epoch, "iter": it,
                    "best": best, "history": history, "rng": torch.get_rng_state()}, resume)

    with open(os.path.join(a.out, "history.json"), "w") as f:
        json.dump(history, f, indent=1)
    if os.path.exists(resume) and not a.max_iters_debug:
        os.remove(resume)
    writer.close()


if __name__ == "__main__":
    main()
