"""Print and sanity-check the preprocessed datasets; exit non-zero if anything is missing/empty."""
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_processed")
RIGA = ["BinRushed", "Magrabia", "MESSIDOR_Base1", "MESSIDOR_Base2", "MESSIDOR_Base3"]
PROSTATE = {"A": "RUNMC", "B": "BMC", "C": "I2CVB", "D": "UCL", "E": "BIDMC", "F": "HK"}
EXPECTED_CASES = {"RUNMC": 30, "BMC": 30, "I2CVB": 19, "UCL": 13, "BIDMC": 12, "HK": 12}  # 116 total


def main():
    errors = []
    print("== RIGA+ (512x512; label 0=BG, 1=disc rim, 2=cup) ==")
    for d in RIGA:
        p = os.path.join(ROOT, "RIGA_plus", f"{d}.npz")
        if not os.path.exists(p):
            errors.append(f"missing {p}")
            continue
        z = np.load(p)
        im, m = z["images"], z["masks"]
        frac = [float((m == k).mean()) for k in (0, 1, 2)]
        print(f"{d:15s} images {len(im):4d} shape {im.shape[1:]} {im.dtype} | label frac BG {frac[0]:.3f} "
              f"rim {frac[1]:.4f} cup {frac[2]:.4f} | images w/o cup {(m.reshape(len(m), -1) == 2).any(1).__invert__().sum()}")
        if len(im) == 0 or im.shape[1:] != (512, 512, 3) or m.shape[1:] != (512, 512):
            errors.append(f"bad RIGA domain {d}")
        if not set(np.unique(m)) <= {0, 1, 2}:
            errors.append(f"unexpected labels in {d}: {np.unique(m)}")

    print("\n== Multi-site prostate (384x384; labelled slices only) ==")
    total = 0
    for letter, site in PROSTATE.items():
        p = os.path.join(ROOT, "Prostate", f"{site}.npz")
        if not os.path.exists(p):
            errors.append(f"missing {p}")
            continue
        z = np.load(p)
        n_cases = len(z["case_names"])
        total += n_cases
        fg = float(z["masks"].mean())
        print(f"{letter} {site:6s} cases {n_cases:3d} slices {len(z['images']):4d} shape {z['images'].shape[1:]} "
              f"| prostate frac {fg:.4f} | labels {np.unique(z['masks']).tolist()}")
        if n_cases != EXPECTED_CASES[site] or len(z["images"]) == 0 or z["images"].shape[1:] != (384, 384):
            errors.append(f"bad prostate site {site}")
        if len(np.unique(z["case"])) != n_cases:
            errors.append(f"{site}: a case has no labelled slices")
    print(f"total prostate cases: {total}")
    if total != 116:
        errors.append(f"expected 116 prostate cases, got {total}")

    if errors:
        print("\nFAILED:\n  " + "\n  ".join(errors))
        sys.exit(1)
    print("\nAll datasets verified.")


if __name__ == "__main__":
    main()
