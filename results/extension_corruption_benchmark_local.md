# Extension 2 - Corruption-robustness benchmark

Main structure Dice % (RIGA+: optic cup; prostate: pooled case Dice) on unseen targets, averaged over 3 severities. 'Drop' = clean minus corrupted, averaged over all 12 corruption settings.

## riga_BinRushed_localfreq (frequency input: local, metric: OC)

Clean: 86.34

| Corruption | Both streams | Image stream only | Frequency stream only |
|---|---|---|---|
| noise | 84.97 | 85.12 | 85.94 |
| blur | 86.20 | 85.99 | 86.69 |
| shading | 84.12 | 84.78 | 86.33 |
| contrast | 81.78 | 83.79 | 85.14 |
| **mean drop** | 2.08 | 1.43 | 0.32 |

## riga_Magrabia_localfreq (frequency input: local, metric: OC)

Clean: 87.80

| Corruption | Both streams | Image stream only | Frequency stream only |
|---|---|---|---|
| noise | 87.55 | 87.60 | 87.38 |
| blur | 86.81 | 86.95 | 87.36 |
| shading | 86.66 | 86.79 | 88.00 |
| contrast | 86.49 | 87.09 | 87.75 |
| **mean drop** | 0.93 | 0.69 | 0.18 |

