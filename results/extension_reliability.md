# Extension 3 - Label-free failure warning

U = 1 - Dice(prediction, prediction on the flipped image). Computed without labels; labels are used only to check it.

| Run | Unit | n | Failures | Spearman(U, Dice) | Failure AUROC | Dice of 20 % most-flagged | Dice of the rest |
|---|---|---|---|---|---|---|---|
| riga_BinRushed | image (optic cup) | 92 | 16 (<80) | -0.45 | 0.88 | 79.6 | 88.2 |
| riga_Magrabia | image (optic cup) | 92 | 9 (<80) | -0.35 | 0.78 | 81.0 | 88.6 |
| prostate_A | case | 86 | 10 (<70) | -0.74 | 0.99 | 64.8 | 87.2 |
| prostate_B | case | 86 | 8 (<70) | -0.67 | 0.86 | 73.6 | 84.3 |
| prostate_C | case | 97 | 20 (<70) | -0.66 | 0.88 | 64.8 | 78.8 |
| prostate_D | case | 103 | 12 (<70) | -0.85 | 0.96 | 58.9 | 87.5 |
| prostate_E | case | 104 | 21 (<70) | -0.87 | 0.96 | 56.0 | 82.7 |
| prostate_F | case | 104 | 16 (<70) | -0.78 | 0.91 | 63.9 | 83.4 |
