"""Real-data access and preprocessing for MAMA-MIA.

IMPORTANT: nothing in this file has been run against the real dataset. It was written from
the dataset paper (folder layout: images/<case>/<case>_000X.nii.gz, segmentations/expert/
<case>.nii.gz, one clinical/imaging Excel table). Verify folder names and column names on
the first download and adjust COLUMN_MAP; the functions fail loudly if something is missing.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

SYNAPSE_ID = "syn60868042"

# our name -> candidate column names in the MAMA-MIA table (check on first download)
COLUMN_MAP = {
    "case_id": ["patient_id"],
    "collection": ["dataset"],
    "pcr": ["pcr"],
    "age": ["age"],
    "hr": ["hr"],
    "her2": ["her2"],
    "menopausal": ["menopause", "menopausal_status"],
    "multifocal": ["multifocal_cancer"],
    "ethnicity": ["ethnicity"],
    "vendor": ["manufacturer"],
    "field_strength": ["field_strength"],
    "acq_times": ["acquisition_times"],
    "rfs_months": ["days_to_recurrence"],
    "rfs_event": ["recurrence"],
}


def download(dest: str, token: str | None = None, entity: str = SYNAPSE_ID):
    """Download from Synapse. Needs a free account and a personal access token."""
    import synapseclient
    import synapseutils

    syn = synapseclient.Synapse()
    syn.login(authToken=token or os.environ["SYNAPSE_AUTH_TOKEN"])
    os.makedirs(dest, exist_ok=True)
    return synapseutils.syncFromSynapse(syn, entity, path=dest)


def harmonise_table(raw: pd.DataFrame) -> pd.DataFrame:
    lower = {c.lower(): c for c in raw.columns}
    out, missing = {}, []
    for ours, cands in COLUMN_MAP.items():
        hit = next((lower[c] for c in cands if c in lower), None)
        if hit is None:
            missing.append(ours)
        else:
            out[ours] = raw[hit]
    required = {"case_id", "collection", "pcr", "age", "hr", "her2"}
    if required & set(missing):
        raise KeyError(f"columns not found for {sorted(required & set(missing))}; edit COLUMN_MAP. "
                       f"Available: {list(raw.columns)[:40]}")
    df = pd.DataFrame(out)
    df["collection"] = df["collection"].astype(str).str.upper()
    return df


def read_image(path: str, pixel_type=None):
    """SimpleITK read that tolerates mangled extensions such as '.niigz' (seen in a Kaggle mirror).

    The file is linked under a temporary '.nii.gz' or '.nii' name chosen from its magic bytes.
    """
    import tempfile

    import SimpleITK as sitk

    def _read(p):
        return sitk.ReadImage(p, pixel_type) if pixel_type is not None else sitk.ReadImage(p)

    if path.endswith((".nii.gz", ".nii", ".mha", ".nrrd")):
        return _read(path)
    with open(path, "rb") as f:
        gz = f.read(2) == b"\x1f\x8b"
    with tempfile.TemporaryDirectory() as d:
        link = os.path.join(d, "img.nii.gz" if gz else "img.nii")
        os.symlink(os.path.abspath(path), link)
        return _read(link)


def preprocess_case(phase_paths: list[str], mask_path: str, size: int = 64, margin: float = 0.25,
                    return_info: bool = False):
    """Resample to 1 mm, z-score over all phases, crop a cube around the mask.

    Returns (image[3, size, size, size] float16, mask[size, size, size] uint8).
    Channels: pre-contrast, first post-contrast, subtraction.
    """
    import SimpleITK as sitk

    def resample(img, interp):
        sp = img.GetSpacing()
        new = [int(round(s * p)) for s, p in zip(img.GetSize(), sp)]
        return sitk.Resample(img, new, sitk.Transform(), interp, img.GetOrigin(), (1.0, 1.0, 1.0),
                             img.GetDirection(), 0, img.GetPixelID())

    if len(phase_paths) < 2:
        raise ValueError("need at least pre-contrast and one post-contrast phase")
    phases = [sitk.GetArrayFromImage(resample(read_image(p, sitk.sitkFloat32), sitk.sitkLinear)) for p in phase_paths[:2]]
    mask = sitk.GetArrayFromImage(resample(read_image(mask_path, sitk.sitkUInt8), sitk.sitkNearestNeighbor)) > 0
    if not mask.any():
        raise ValueError(f"empty mask: {mask_path}")
    stack = np.stack(phases)
    stack = (stack - stack.mean()) / (stack.std() + 1e-6)
    idx = np.argwhere(mask)
    lo, hi = idx.min(0), idx.max(0) + 1
    centre = (lo + hi) // 2
    half = int(np.ceil((hi - lo).max() * (1 + margin) / 2))
    half = max(half, 16)
    sl, pad = [], []
    for c, n in zip(centre, mask.shape):
        a, b = c - half, c + half
        sl.append(slice(max(a, 0), min(b, n)))
        pad.append((max(-a, 0), max(b - n, 0)))
    crop = np.pad(stack[(slice(None), *sl)], [(0, 0), *pad])
    mcrop = np.pad(mask[tuple(sl)], pad)

    import torch
    import torch.nn.functional as F

    t = F.interpolate(torch.from_numpy(crop)[None], size=(size,) * 3, mode="trilinear", align_corners=False)[0].numpy()
    m = F.interpolate(torch.from_numpy(mcrop[None, None].astype(np.float32)), size=(size,) * 3, mode="nearest")[0, 0].numpy()
    image = np.stack([t[0], t[1], t[1] - t[0]]).astype(np.float16)
    if return_info:  # physical size is lost by the resize, so keep it as tabular features
        info = {"tumour_mm3": float(mask.sum()), "extent_mm": float((hi - lo).max()), "crop_mm": float(2 * half),
                "n_components_ok": int(mask.sum() > 0)}
        return image, m.astype(np.uint8), info
    return image, m.astype(np.uint8)


class Prepared:
    """Folder of <case_id>.npz (image, mask) + table.csv + radiomics.csv. Arrays are cached in RAM."""

    def __init__(self, root: str, cache: bool = True):
        self.root = root
        self.table = pd.read_csv(os.path.join(root, "table.csv"))
        self.radiomics_names = list(pd.read_csv(os.path.join(root, "radiomics.csv"), index_col=0, nrows=1).columns)
        self.radiomics = pd.read_csv(os.path.join(root, "radiomics.csv"), index_col=0).loc[self.table.case_id].to_numpy(np.float32)
        self._img, self._msk = {}, {}
        if cache:
            for i, c in enumerate(self.table.case_id):
                z = np.load(os.path.join(root, f"{c}.npz"))
                self._img[i], self._msk[i] = z["image"], z["mask"]

    def image(self, i: int) -> np.ndarray:
        if i not in self._img:
            self._img[i] = np.load(os.path.join(self.root, f"{self.table.case_id[i]}.npz"))["image"]
        return self._img[i].astype(np.float32)

    def mask(self, i: int) -> np.ndarray:
        if i not in self._msk:
            self._msk[i] = np.load(os.path.join(self.root, f"{self.table.case_id[i]}.npz"))["mask"]
        return self._msk[i]


class InMemory:
    def __init__(self, table, images, masks, radiomics):
        self.table, self._im, self._m, self.radiomics = table.reset_index(drop=True), images, masks, radiomics

    def image(self, i):
        return self._im[i]

    def mask(self, i):
        return self._m[i]
