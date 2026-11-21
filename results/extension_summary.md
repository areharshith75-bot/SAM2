# Extensions over the FSAM baseline - summary

All numbers are real runs in this repository (single seed 1234, same protocol as the reproduction).
Baseline = our FSAM reproduction (Tables 1-2).

## 1. Local (patch-wise) frequency adapter - `--freq_mode local`

| RIGA+ OD / OC (avg of 3 targets) | Baseline FSAM | Local frequency |
|---|---|---|
| BinRushed | 96.45 / 86.40 | 96.37 / 86.34 |
| Magrabia | 96.47 / 87.13 | 96.02 / 87.80 |

Frequency-stream reliance (OC Dice when the frequency input is replaced; image stream untouched):

| | Baseline own / zeros / noise | Local own / zeros / noise |
|---|---|---|
| BinRushed | 86.40 / 86.53 / 85.91 | 86.34 / 79.30 / 77.48 |
| Magrabia | 87.13 / 87.04 / 87.05 | 87.80 / 84.91 / 82.74 |

Accuracy is on par; the frequency branch, which the baseline ignores on all 8 models
(`extension_frequency_reliance.md`), becomes functional.

## 2. GIN random-style augmentation - `--style_aug gin`

| | Baseline | GIN | Delta |
|---|---|---|---|
| RIGA+ BinRushed OD / OC | 96.45 / 86.40 | 96.23 / 84.66 | -0.22 / -1.74 |
| RIGA+ Magrabia OD / OC | 96.47 / 87.13 | 96.42 / 88.39 | -0.05 / +1.26 |
| Prostate E (BIDMC), selected checkpoint | 77.31 | 78.66 | +1.35 |
| Prostate E (BIDMC), final checkpoint | 79.30 | 81.08 | +1.78 |
| Prostate C (I2CVB), selected checkpoint | 76.08 | 75.36 | -0.72 |
| Prostate C (I2CVB), final checkpoint | 73.54 | 71.59 | -1.95 |

Mixed: clear gains when training on BIDMC (both checkpoints) and on Magrabia's optic cup, losses on
BinRushed and I2CVB. Not a uniform improvement with one seed.

## 3. Label-free failure warning - `analysis/reliability.py`

Prostate failures (case Dice < 70) detected with AUROC 0.86-0.99 on all 6 source sites; referring the 20 %
most-flagged cases raises the mean Dice of the remaining cases from 79.9 to ~84.0. See `extension_reliability.md`.

## 4. Corruption benchmark - `analysis/corruption_benchmark.py`

Sensitivity to noise / blur / shading / contrast comes almost entirely through the image stream
(frequency-only corruption: <= 0.3 drop). See `extension_corruption_benchmark*.md`.

## 5. Flip test-time averaging - `eval.py --tta`

+0.5 prostate average (79.92 -> 80.40), +0.2 / +0.4 RIGA+ OC; consistent but small.
