"""Radiomic features.

`simple_features` is a dependency-free set (first-order + shape) used for tests and as a
fallback. For the paper, use `pyradiomics_features`, which calls PyRadiomics with a fixed,
IBSI-oriented configuration. The PyRadiomics path has NOT been run in the build
environment; check it on the first Kaggle session.
"""
from __future__ import annotations

import numpy as np

SIMPLE_NAMES = [
    f"{ch}_{stat}"
    for ch in ("post", "sub")
    for stat in ("mean", "std", "p10", "p50", "p90", "skew", "kurt", "energy")
] + ["volume_vox", "bbox_extent", "surface_ratio", "rim_core_ratio"]


def _first_order(v: np.ndarray) -> list[float]:
    if v.size == 0:
        return [np.nan] * 8
    mu, sd = float(v.mean()), float(v.std() + 1e-8)
    z = (v - mu) / sd
    return [
        mu,
        sd,
        float(np.percentile(v, 10)),
        float(np.percentile(v, 50)),
        float(np.percentile(v, 90)),
        float((z**3).mean()),
        float((z**4).mean()),
        float((v**2).mean()),
    ]


def simple_features(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """image: (3, D, H, W) pre/post/subtraction; mask: (D, H, W)."""
    m = mask.astype(bool)
    feats = _first_order(image[1][m]) + _first_order(image[2][m])
    vol = float(m.sum())
    if vol == 0:
        return np.array(feats + [0.0, np.nan, np.nan, np.nan], dtype=np.float32)
    idx = np.argwhere(m)
    extent = float((idx.max(0) - idx.min(0) + 1).max())
    # surface voxels: mask voxels with at least one face neighbour outside the mask
    pad = np.pad(m, 1)
    inner = pad[1:-1, 1:-1, 1:-1].copy()
    for ax in range(3):
        for sh in (-1, 1):
            inner &= np.roll(pad, sh, axis=ax)[1:-1, 1:-1, 1:-1]
    surface = m & ~inner
    rim = image[2][surface].mean() if surface.any() else np.nan
    core = image[2][inner].mean() if inner.any() else np.nan
    feats += [vol, extent, float(surface.sum() / vol), float(rim / (abs(core) + 1e-6))]
    return np.array(feats, dtype=np.float32)


def extract_simple(images: np.ndarray, masks: np.ndarray) -> np.ndarray:
    return np.stack([simple_features(i, m) for i, m in zip(images, masks)])


def pyradiomics_features(image_path: str, mask_path: str, bin_width: float = 0.1) -> dict:
    """PyRadiomics on a z-scored image. Untested here; requires `pyradiomics`."""
    from radiomics import featureextractor  # type: ignore

    ext = featureextractor.RadiomicsFeatureExtractor(
        binWidth=bin_width, resampledPixelSpacing=[1, 1, 1], interpolator="sitkBSpline", normalize=False
    )
    ext.disableAllFeatures()
    for cls in ("firstorder", "shape", "glcm", "glrlm", "glszm", "gldm", "ngtdm"):
        ext.enableFeatureClassByName(cls)
    out = ext.execute(image_path, mask_path)
    return {k: float(v) for k, v in out.items() if not k.startswith("diagnostics")}
