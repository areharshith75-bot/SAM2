"""In-RAM datasets for the preprocessed RIGA+ / prostate arrays, with the DAPSAM augmentations.

Preprocessed layout (written by data/preprocess_*.py):
  data_processed/RIGA_plus/<Domain>.npz : images uint8 (N,512,512,3), masks uint8 (N,512,512) {0,1,2}
  data_processed/Prostate/<Site>.npz    : images float32 (S,384,384) (percentile-clipped),
                                          masks uint8 (S,384,384) {0,1}, case int (S,), case_names
"""
import math
import os

import numpy as np
import torch
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.color_transforms import (BrightnessMultiplicativeTransform, BrightnessTransform,
                                                         ContrastAugmentationTransform, GammaTransform)
from batchgenerators.transforms.noise_transforms import GaussianBlurTransform, GaussianNoiseTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform, SpatialTransform, SpatialTransform_2
from torch.utils.data import Dataset

RIGA_DOMAINS = ["BinRushed", "Magrabia", "MESSIDOR_Base1", "MESSIDOR_Base2", "MESSIDOR_Base3"]
PROSTATE_SITES = {"A": "RUNMC", "B": "BMC", "C": "I2CVB", "D": "UCL", "E": "BIDMC", "F": "HK"}


def zscore(x: np.ndarray) -> np.ndarray:
    """Per-image, per-channel standardisation (DAPSAM normalize_image), x: (B, C, H, W)."""
    m = x.mean(axis=(2, 3), keepdims=True)
    s = x.std(axis=(2, 3), keepdims=True)
    return ((x - m) / (s + 1e-8)).astype(np.float32)


# ----------------------------------------------------------------------------- RIGA+
def riga_train_transform(patch=(512, 512)):
    return Compose([
        SpatialTransform_2(
            patch, [i // 2 for i in patch], do_elastic_deform=True, deformation_scale=(0, 0.25),
            do_rotation=True, angle_x=(-15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
            angle_y=(-15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi), do_scale=True, scale=(0.75, 1.25),
            border_mode_data="constant", border_cval_data=0, border_mode_seg="constant", border_cval_seg=0,
            order_seg=1, order_data=3, random_crop=True,
            p_el_per_sample=0.1, p_rot_per_sample=0.1, p_scale_per_sample=0.1),
        MirrorTransform(axes=(0, 1, 2)),
        BrightnessMultiplicativeTransform((0.7, 1.5), per_channel=True, p_per_sample=0.15),
        GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.15),
    ])


class RIGADataset(Dataset):
    def __init__(self, root, domain):
        d = np.load(os.path.join(root, f"{domain}.npz"))
        self.images = d["images"]  # (N,H,W,3) uint8
        self.masks = d["masks"]    # (N,H,W) uint8
        self.names = d["names"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = self.images[i].transpose(2, 0, 1).astype(np.float32)
        return img, self.masks[i][None].astype(np.float32), i


def riga_collate_train(batch):
    img, seg, _ = zip(*batch)
    d = riga_train_transform(img[0].shape[-2:])(data=np.stack(img), seg=np.stack(seg))
    return torch.from_numpy(zscore(d["data"])), torch.from_numpy(np.round(d["seg"][:, 0]).astype(np.int64))


def riga_collate_test(batch):
    img, seg, idx = zip(*batch)
    return torch.from_numpy(zscore(np.stack(img))), torch.from_numpy(np.stack(seg)[:, 0].astype(np.int64)), list(idx)


# ----------------------------------------------------------------------------- Prostate
def prostate_train_transform(patch=(384, 384)):
    return Compose([
        SpatialTransform(
            patch, patch_center_dist_from_border=[i // 2 for i in patch], do_elastic_deform=True,
            alpha=(0., 900.), sigma=(9., 13.), do_rotation=True,
            angle_x=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            angle_y=(-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi), do_scale=True, scale=(0.85, 1.25),
            border_mode_data="constant", border_cval_data=0, order_data=3, border_mode_seg="constant",
            border_cval_seg=-1, order_seg=1, random_crop=True,
            p_el_per_sample=0.2, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
            independent_scale_for_each_axis=False, data_key="data", label_key="mask"),
        GaussianNoiseTransform(p_per_sample=0.1, data_key="data"),
        GaussianBlurTransform(blur_sigma=(0.5, 1.), different_sigma_per_channel=True, p_per_channel=0.5,
                              p_per_sample=0.2, data_key="data"),
        BrightnessMultiplicativeTransform((0.75, 1.25), p_per_sample=0.15, data_key="data"),
        BrightnessTransform(0.0, 0.1, True, p_per_sample=0.15, p_per_channel=0.5, data_key="data"),
        ContrastAugmentationTransform(p_per_sample=0.15, data_key="data"),
        SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True, p_per_channel=0.5,
                                       order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                       ignore_axes=None, data_key="data"),
        GammaTransform(gamma_range=(0.7, 1.5), invert_image=False, per_channel=True, retain_stats=True,
                       p_per_sample=0.15, data_key="data"),
        MirrorTransform(axes=(0, 1), data_key="data", label_key="mask"),
    ])


class ProstateDataset(Dataset):
    """Labelled slices of the selected cases. `min_iters`/`batch_size` replicate the slice pool so an
    epoch has >= 250 iterations, exactly as DAPSAM's PROSTATE_dataset does for training."""

    def __init__(self, root, site, cases=None, batch_size=None, min_iters=250):
        d = np.load(os.path.join(root, f"{site}.npz"))
        keep = np.ones(len(d["case"]), bool) if cases is None else np.isin(d["case"], cases)
        self.images = d["images"][keep]
        self.masks = d["masks"][keep]
        self.case = d["case"][keep]
        self.site = site
        self.index = np.arange(len(self.images))
        if batch_size is not None:
            scale = math.ceil(min_iters / (len(self.index) // batch_size))
            self.index = np.tile(self.index, scale)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        j = self.index[i]
        return self.images[j][None].copy(), self.masks[j][None].astype(np.float32), f"{self.site}/{self.case[j]}"


def prostate_collate_train(batch):
    img, msk, _ = zip(*batch)
    d = prostate_train_transform(img[0].shape[-2:])(data=np.stack(img), mask=np.stack(msk))
    x = np.repeat(d["data"], 3, axis=1)
    y = (d["mask"] > 0).astype(np.float32)  # border fill -1 -> 0, as `y` is used as float target
    return torch.from_numpy(zscore(x)), torch.from_numpy(y)


def prostate_collate_test(batch):
    img, msk, cid = zip(*batch)
    x = np.repeat(np.stack(img), 3, axis=1)
    return torch.from_numpy(zscore(x)), torch.from_numpy(np.stack(msk)), list(cid)


def prostate_split(root, site, val_frac=0.1, seed=1234):
    """Case-level 90% / 10% split of a source site (fixed seed)."""
    cases = np.unique(np.load(os.path.join(root, f"{site}.npz"))["case"])
    rng = np.random.RandomState(seed)
    perm = rng.permutation(cases)
    n_val = max(1, int(round(val_frac * len(cases))))
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])
