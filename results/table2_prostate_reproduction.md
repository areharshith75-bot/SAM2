# Table 2 reproduction — multi-site prostate (DSC %)

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
