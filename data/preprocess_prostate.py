"""Preprocess the SAML multi-site prostate MRI data (Liu et al., MICCAI 2020) into RAM-friendly arrays.

Source: https://liuquande.github.io/SAML/  ->  Processed_data_nii.zip (116 T2 cases, 6 sites, 384x384
axial slices, already resampled/standardised by the dataset authors).

Mirrors DAPSAM's PROSTATE_dataset exactly:
  * only slices whose mask contains prostate are used (train and test),
  * per slice: 0.2 / 99.8 percentile clipping of the non-zero intensities,
  * mask labels > 1 are merged into 1 (whole gland).
Output: data_processed/Prostate/<SITE>.npz with images float32 (S,384,384), masks uint8,
        case int (S,), case_names (n_cases,), slice_index (S,).
"""
import argparse
import glob
import os
import re

import numpy as np
import SimpleITK as sitk

SITES = ["RUNMC", "BMC", "I2CVB", "UCL", "BIDMC", "HK"]


def clip_slice(x):
    x = x.copy()
    m = x > 0
    if m.any():
        lo, hi = np.percentile(x[m], 0.2), np.percentile(x[m], 99.8)
        x[m & (x < lo)] = lo
        x[m & (x > hi)] = hi
    return x


def process_site(raw, site, size=384):
    files = sorted(f for f in glob.glob(os.path.join(raw, site, "*.nii.gz")) if "segmentation" not in f.lower())
    imgs, msks, case, sl, names = [], [], [], [], []
    for ci, f in enumerate(files):
        seg = [g for g in glob.glob(f.replace(".nii.gz", "_*.nii.gz")) if "segmentation" in g.lower()]
        assert len(seg) == 1, (f, seg)
        im = sitk.GetArrayFromImage(sitk.ReadImage(f)).astype(np.float32)
        lb = sitk.GetArrayFromImage(sitk.ReadImage(seg[0]))
        assert im.shape == lb.shape and im.shape[1:] == (size, size), (f, im.shape, lb.shape)
        names.append(re.sub(r"\.nii\.gz$", "", os.path.basename(f)))
        for s in range(im.shape[0]):
            if lb[s].max() > 0:
                imgs.append(clip_slice(im[s]))
                msks.append((lb[s] > 0).astype(np.uint8))
                case.append(ci)
                sl.append(s)
    return dict(images=np.stack(imgs), masks=np.stack(msks), case=np.array(case), slice_index=np.array(sl),
                case_names=np.array(names))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data_raw/prostate")
    p.add_argument("--out", default="data_processed/Prostate")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for site in SITES:
        d = process_site(a.raw, site)
        np.savez(os.path.join(a.out, f"{site}.npz"), **d)
        print(f"{site}: {len(d['case_names'])} cases, {len(d['images'])} labelled slices")
