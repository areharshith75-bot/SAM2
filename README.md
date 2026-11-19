# SAM2 — FSAM Reproduction

Reproduction of **"Frequency Adapter with SAM for Generalized Medical Image Segmentation"**
(FSAM, arXiv:2605.09925v1): SAM ViT-B + LoRA (r=4) + N=12 frequency adapters + DAPSAM's
memory-based automated prompt generator, evaluated for single-source domain generalization
on RIGA+ (fundus OD/OC) and the 6-site prostate MRI benchmark.

Status: work in progress — see commit history.

## Layout
```
models/    FSAM architecture (frequency extractor, LoRA, adapters, prompt generator)
utils/     losses, metrics
data/      dataset download / preprocessing / verification
scripts/   experiment launchers
tests/     unit + end-to-end tests
pretrained/ SAM checkpoint (sam_vit_b_01ec64.pth, not tracked)
```
