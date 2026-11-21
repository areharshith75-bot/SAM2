# Extension 2 - Corruption-robustness benchmark

Main structure Dice % (RIGA+: optic cup; prostate: pooled case Dice) on unseen targets, averaged over 3 severities. 'Drop' = clean minus corrupted, averaged over all 12 corruption settings.

## riga_BinRushed (frequency input: global, metric: OC)

Clean: 86.40

| Corruption | Both streams | Image stream only | Frequency stream only |
|---|---|---|---|
| noise | 82.98 | 82.62 | 86.10 |
| blur | 87.80 | 87.67 | 86.65 |
| shading | 83.74 | 83.75 | 86.41 |
| contrast | 82.62 | 82.70 | 86.37 |
| **mean drop** | 2.12 | 2.22 | 0.02 |

## riga_Magrabia (frequency input: global, metric: OC)

Clean: 87.13

| Corruption | Both streams | Image stream only | Frequency stream only |
|---|---|---|---|
| noise | 87.32 | 87.54 | 87.13 |
| blur | 86.21 | 86.41 | 86.90 |
| shading | 85.86 | 85.87 | 87.18 |
| contrast | 86.49 | 86.36 | 87.21 |
| **mean drop** | 0.66 | 0.58 | 0.03 |

## prostate_E (frequency input: global, metric: Dice)

Clean: 77.31

| Corruption | Both streams | Image stream only | Frequency stream only |
|---|---|---|---|
| noise | 76.82 | 77.42 | 76.42 |
| blur | 74.71 | 74.32 | 77.53 |
| shading | 76.84 | 76.83 | 77.31 |
| contrast | 69.94 | 70.07 | 77.18 |
| **mean drop** | 2.73 | 2.65 | 0.20 |

