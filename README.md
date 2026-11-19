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

<!-- RESULTS -->
