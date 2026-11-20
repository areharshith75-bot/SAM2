# SAM2 — FSAM Reproduction

Independent, end-to-end reproduction of **"Frequency Adapter with SAM for Generalized Medical Image
Segmentation"** (FSAM, Bui et al., arXiv:2605.09925v1) for **single-source domain generalisation** on
RIGA+ (optic disc / cup) and the 6-site prostate MRI benchmark. FSAM builds on DAPSAM
(Wei et al., MICCAI 2024); its code was used as the reference for data handling, loss and evaluation.

All numbers in `results/` come from real training and evaluation runs in this repository on the
real datasets (no synthetic data, no copied numbers). See [Results](#results).

## Architecture

```
               ┌──────────────── FSAM encoder (SAM ViT-B, frozen) ────────────────┐
image I ──┬──► patch_embed + pos_embed ─► E_img ─► [ViT block l + LoRA(Q,K,V)] ─► H_img ─(+)─► E_img(l+1)
          │                                                                       ▲
          └─► log|FFT2(I)| ─► patch_embed + pos_embed ─► E_freq ─► [Freq Adapter l] ─► H_freq
                (fp32, standardised)                       E_freq(l+1) = E_freq + H_freq        × 12 blocks
                                                                                     │
          E_img(12) ─► SAM neck ─► e_i (256 × H/16 × W/16) ──┬──────────────────────►│ SAM mask decoder ─► masks
                                                             └► Automated prompt generator ─► dense prompt
                                                                (GAP+GMP prototype, cosine read-out of a
                                                                 256-slot memory bank, activation map, 1×1 conv)
```

| Component | File | Trainable params |
|---|---|---|
| FFT amplitude + shared patch embedding | `models/frequency_extractor.py` | 0 (frozen SAM) |
| LoRA r=4 on Q, K, V of all 12 blocks | `models/lora_sam.py` | 221,184 |
| 12 Frequency Adapters (LN → 768→192 → GELU → 192→768) | `models/fsam_encoder.py` | 3,568,896 |
| Automated prompt generator | `models/prompt_generator.py` | 197,120 |
| SAM mask decoder (num_classes mask tokens) | `models/fsam.py` | 3.65 M (512) / 3.51 M (384) |

## Setup

```bash
python3 -m venv .venv            # or: python3 -m venv --without-pip .venv + bootstrap pip
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install -r requirements.txt
curl -L -o pretrained/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
data/download_datasets.sh        # RIGA+ (Zenodo) + SAML prostate (Google Drive) -> data_processed/, then verifies
.venv/bin/python -m pytest -q    # unit / parity / GPU end-to-end tests
```

## Running

```bash
scripts/run_all_experiments.sh                     # all 8 runs + evaluation + report (resumable)

# or individually
.venv/bin/python train.py --dataset riga --source BinRushed
.venv/bin/python train.py --dataset prostate --source A        # A..F = RUNMC BMC I2CVB UCL BIDMC HK
.venv/bin/python eval.py --run outputs/riga_BinRushed outputs/prostate_A
.venv/bin/python eval.py --generate_report                     # -> results/table{1,2}_*.md
```

## Protocol

| | RIGA+ | Prostate |
|---|---|---|
| Input | 512×512 RGB | 384×384 (slice ×3 channels) |
| Source training data | all labelled images of BinRushed (195) / Magrabia (95) | 90% of the source site's cases |
| Model selection | checkpoint at stop epoch 160 | best in-domain Dice on the 10% validation cases |
| Targets | official test lists of MESSIDOR Base1/2/3 (35/30/27 images) | all cases of the 5 other sites |
| Metric | per-image Dice for OD and OC, averaged per domain | per-case 3D Dice over labelled slices |
| Warm-up | 25 iterations | 250 iterations |

Shared settings: SAM ViT-B, LoRA rank 4, AdamW (lr 5e-4, wd 0.1), linear warm-up then poly(0.9) decay,
batch 8, max 200 epochs with early stop at 160, loss (1−λ)·BCE + λ·Dice with λ = 0.8, DAPSAM
augmentations (batchgenerators), per-image z-score normalisation, bf16 autocast (FFT in fp32).
No target-domain data is used for training or model selection.

### Choices where the paper is silent or differs from the request

* **Output head**: one sigmoid channel per structure (RIGA+: OD, OC; prostate: gland), exactly as the
  DAPSAM code. This replaces a softmax over [BG, OD, OC].
* **Prompt generator**: cosine similarity in the memory read-out (paper Eq. 3). The DAPSAM code uses a dot
  product (`--similarity dot`). The memory has 256 prototypes (DAPSAM default).
* **Decoder positional encoding**: SAM's pretrained dense PE. In DAPSAM it is a fresh random Gaussian matrix.
* **Frequency input**: `log1p(|FFT2|)` amplitude, not fft-shifted, standardised per image and channel.
* **Attention kernel**: SAM attention runs through fused `scaled_dot_product_attention`, with the
  relative-position term passed as an additive bias. It matches SAM's original attention to 1e-6 in fp32
  (`tests/test_fast_attention.py`) and is about 18% faster.
* **Prostate augmentation border**: DAPSAM fills the mask border with −1 and feeds it to BCE as a target.
  Here, out-of-image mask pixels are treated as background.

## Results

See `results/table1_riga_reproduction.md` and `results/table2_prostate_reproduction.md`. They are
generated by `eval.py --generate_report` from `outputs/*/results.json`.

### Table 1 — RIGA+

Reproduced = this repository, real training + evaluation; Paper = FSAM row of Table 1.

| Source | Row | MESSIDOR_Base1 OD | MESSIDOR_Base1 OC | MESSIDOR_Base2 OD | MESSIDOR_Base2 OC | MESSIDOR_Base3 OD | MESSIDOR_Base3 OC | Average OD | Average OC |
|---|---|---|---|---|---|---|---|---|---|
| BinRushed | Reproduced | 96.33 | 88.14 | 96.68 | 84.69 | 96.34 | 86.38 | 96.45 | 86.40 |
| BinRushed | Paper | 96.13 | 88.12 | 96.29 | 86.20 | 96.32 | 88.28 | 96.25 | 87.53 |
| BinRushed | Δ | +0.20 | +0.02 | +0.39 | -1.51 | +0.02 | -1.90 | +0.20 | -1.13 |
| Magrabia | Reproduced | 96.30 | 86.58 | 96.67 | 87.50 | 96.45 | 87.31 | 96.47 | 87.13 |
| Magrabia | Paper | 96.28 | 87.61 | 96.41 | 90.07 | 96.48 | 88.20 | 96.39 | 88.63 |
| Magrabia | Δ | +0.02 | -1.03 | +0.26 | -2.57 | -0.03 | -0.89 | +0.08 | -1.50 |

Checkpoints: BinRushed: last_model.pth (epoch 159), Magrabia: last_model.pth (epoch 159)

### Table 2 — Prostate

Each column: trained on that site (90% cases, best epoch on the 10% in-domain validation cases),
tested on the 5 unseen sites. Main row = mean over all target cases (DAPSAM protocol);
the per-domain-mean row averages the 5 site means instead.

| Row | A (RUNMC) | B (BMC) | C (I2CVB) | D (UCL) | E (BIDMC) | F (HK) | Average |
|---|---|---|---|---|---|---|---|
| Reproduced | 82.78 | 82.17 | 76.08 | 81.70 | 77.31 | 79.49 | 79.92 |
| Reproduced (per-domain mean) | 81.68 | 82.67 | 72.94 | 79.83 | 77.19 | 78.40 | 78.78 |
| Supplementary: final epoch-160 checkpoint (not used for selection) | 83.49 | 81.65 | 73.54 | 81.85 | 79.30 | 81.14 | 80.16 |
| Paper | 86.16 | 82.52 | 74.98 | 86.44 | 83.14 | 83.19 | 82.74 |

## Per-target breakdown (DSC %)

- **A (RUNMC)**, best_model.pth epoch 114: BMC 86.11 (n=30), I2CVB 82.97 (n=19), UCL 87.88 (n=13), BIDMC 63.23 (n=12), HK 88.19 (n=12)
- **B (BMC)**, best_model.pth epoch 39: RUNMC 83.45 (n=30), I2CVB 74.13 (n=19), UCL 85.45 (n=13), BIDMC 84.24 (n=12), HK 86.08 (n=12)
- **C (I2CVB)**, best_model.pth epoch 9: RUNMC 79.56 (n=30), BMC 82.83 (n=30), UCL 80.71 (n=13), BIDMC 53.94 (n=12), HK 67.66 (n=12)
- **D (UCL)**, best_model.pth epoch 124: RUNMC 88.69 (n=30), BMC 82.81 (n=30), I2CVB 76.89 (n=19), BIDMC 62.68 (n=12), HK 88.08 (n=12)
- **E (BIDMC)**, best_model.pth epoch 99: RUNMC 77.12 (n=30), BMC 85.64 (n=30), I2CVB 57.40 (n=19), UCL 76.99 (n=13), HK 88.82 (n=12)
- **F (HK)**, best_model.pth epoch 124: RUNMC 86.82 (n=30), BMC 80.56 (n=30), I2CVB 67.05 (n=19), UCL 80.28 (n=13), BIDMC 77.27 (n=12)

### Reading the results

* **RIGA+**: optic disc Dice matches or slightly beats the paper on all targets (+0.0 to +0.4).
  Optic cup is 1.1 / 1.5 points lower on average for BinRushed / Magrabia; the largest gap is MESSIDOR Base2.
* **Prostate**: 79.92 vs. 82.74 average (−2.8). B and C match or exceed the paper. A, D, E and F
  are 2.5–5.8 points lower, mostly because of the BIDMC and I2CVB targets, which are the hardest
  shifts in this benchmark.
* **Prostate model selection is noisy.** The paper's 10% in-domain split leaves only 1–3 validation
  cases per site, and validation Dice is flat at about 0.89–0.91 through most of training. Site C was
  selected at epoch 9. The final epoch-160 checkpoints give 80.16 on average (supplementary row), which
  shows how much the result depends on selection.
* All results are single runs (seed 1234), each with one checkpoint per experiment. The paper reports
  no variance, and test sets of 12–35 cases/images can move by about a point between seeds.
* Wall-clock on one RTX 2000 Ada (16 GB): RIGA+ 39 / 20 min, prostate 3.1–3.4 h per site
  (about 20 h for all 8 runs). Peak VRAM was 5.3 GB (512px) and 2.9 GB (384px).

Per-run metrics, configs and training histories: `results/raw/`.

