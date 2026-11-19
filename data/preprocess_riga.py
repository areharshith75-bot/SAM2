"""Preprocess RIGA+ (Hu et al., Zenodo record 6325549, RIGAPlus.zip) into 512x512 arrays.

Splits follow the official CSVs shipped with RIGAPlus.zip:
  * source domains (BinRushed, Magrabia): all labelled images = <D>_train.csv + <D>_test.csv
    (195 / 95 images), as used by CCSDG/DAPSAM for single-source training;
  * target domains (MESSIDOR_Base1/2/3): the official labelled test split <D>_test.csv.
Masks: rater-1 annotation (`-1.tif`, DAPSAM `convert_labeled_list(r=1)`); value 0 = BG,
255 = disc rim, 128 = cup -> stored as 0/1/2 (DAPSAM: mask[label>0]=1; mask[label==128]=2).
Images are resized with bicubic, masks with nearest neighbour (as DAPSAM's PIL resize).
"""
import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image

SPLITS = {
    "BinRushed": ["BinRushed_train.csv", "BinRushed_test.csv"],
    "Magrabia": ["Magrabia_train.csv", "Magrabia_test.csv"],
    "MESSIDOR_Base1": ["MESSIDOR_Base1_test.csv"],
    "MESSIDOR_Base2": ["MESSIDOR_Base2_test.csv"],
    "MESSIDOR_Base3": ["MESSIDOR_Base3_test.csv"],
}


def load_pair(raw, img_rel, mask_rel, size, rater=1):
    img = Image.open(os.path.join(raw, img_rel)).convert("RGB").resize((size, size), Image.BICUBIC)
    lab = Image.open(os.path.join(raw, mask_rel.replace(".tif", f"-{rater}.tif"))).resize((size, size), Image.NEAREST)
    lab = np.array(lab)
    assert set(np.unique(lab)) <= {0, 128, 255}, np.unique(lab)
    m = np.zeros(lab.shape, np.uint8)
    m[lab > 0] = 1
    m[lab == 128] = 2
    return np.array(img, np.uint8), m


def check_cup_inside_disc(m):
    """Sanity check of the label convention: the cup (2) must lie within the disc's bounding box."""
    cy, cx = np.nonzero(m == 2)
    dy, dx = np.nonzero(m > 0)
    if len(cy) == 0:
        return True
    return dy.min() <= cy.min() and cy.max() <= dy.max() and dx.min() <= cx.min() and cx.max() <= dx.max()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data_raw/RIGAPlus")
    p.add_argument("--out", default="data_processed/RIGA_plus")
    p.add_argument("--size", type=int, default=512)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for dom, csvs in SPLITS.items():
        df = pd.concat([pd.read_csv(os.path.join(a.raw, c)) for c in csvs], ignore_index=True)
        assert not df["image"].duplicated().any()
        imgs, msks = zip(*(load_pair(a.raw, r.image, r.mask, a.size) for r in df.itertuples()))
        msks = np.stack(msks)
        bad = sum(not check_cup_inside_disc(m) for m in msks)
        assert bad == 0, f"{dom}: {bad} masks with cup outside disc"
        np.savez(os.path.join(a.out, f"{dom}.npz"), images=np.stack(imgs), masks=msks,
                 names=df["image"].to_numpy().astype(str), csvs=np.array(csvs))
        print(f"{dom}: {len(imgs)} images from {csvs}")
