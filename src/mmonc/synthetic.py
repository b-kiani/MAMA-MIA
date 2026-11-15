"""Synthetic cohort that mimics the *shape* of MAMA-MIA so the pipeline can be tested.

Nothing produced from this module is a research result. The generator plants a strong
clinical signal (receptor subtype) and a weak imaging signal, so tests can check that
the code recovers something, not that the science is right.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COLLECTIONS = {  # name: (share, pCR base logit shift, mean delay pre->1st post in s)
    "ISPY2": (0.65, 0.0, 145.0),
    "DUKE": (0.19, -0.4, 241.0),
    "ISPY1": (0.12, -0.2, 390.0),
    "NACT": (0.04, -0.6, 442.0),
}


def make_cohort(n: int = 400, size: int = 24, seed: int = 0, img_effect: float = 0.45):
    """Return (table, images, masks).

    table  : DataFrame, one row per case
    images : float32 array (n, 3, size, size, size) -> pre, first post, subtraction
    masks  : uint8 array (n, size, size, size)
    """
    rng = np.random.default_rng(seed)
    names = list(COLLECTIONS)
    probs = np.array([COLLECTIONS[k][0] for k in names])
    coll = rng.choice(names, size=n, p=probs / probs.sum())

    age = np.clip(rng.normal(49, 10, n), 22, 80)
    hr = rng.binomial(1, 0.55, n)
    her2 = rng.binomial(1, 0.25, n)
    multifocal = rng.binomial(1, 0.36, n).astype(float)
    menopausal = (age + rng.normal(0, 4, n) > 51).astype(float)
    # menopausal status is absent in the two oldest collections, as in the real data
    menopausal[np.isin(coll, ["ISPY1", "NACT"])] = np.nan
    multifocal[rng.random(n) < 0.05] = np.nan

    latent_img = rng.normal(0, 1, n)  # weak imaging signal
    shift = np.array([COLLECTIONS[c][1] for c in coll])
    logit = -0.9 - 1.3 * hr + 1.1 * her2 - 0.015 * (age - 49) + img_effect * latent_img + shift
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    delay = np.array([COLLECTIONS[c][2] for c in coll]) * rng.uniform(0.8, 1.25, n)
    dt = np.stack([np.zeros(n), delay, delay * rng.uniform(1.5, 2.0, n)], axis=1)

    grid = np.stack(np.meshgrid(*[np.arange(size)] * 3, indexing="ij"), axis=0)
    images = np.empty((n, 3, size, size, size), dtype=np.float32)
    masks = np.empty((n, size, size, size), dtype=np.uint8)
    for i in range(n):
        centre = rng.uniform(size * 0.35, size * 0.65, 3)
        radius = rng.uniform(size * 0.12, size * 0.25)
        d = np.sqrt(((grid - centre[:, None, None, None]) ** 2).sum(0))
        m = d < radius
        pre = rng.normal(0, 1, (size,) * 3)
        # enhancement depends on the latent signal and (confound) on contrast timing
        enh = (1.0 + 0.35 * latent_img[i]) * (1 - np.exp(-delay[i] / 200.0))
        post = pre + m * enh * 2.0 + rng.normal(0, 0.3, (size,) * 3)
        images[i] = np.stack([pre, post, post - pre]).astype(np.float32)
        masks[i] = m

    t = rng.exponential(60 * np.exp(-0.5 * (1 - y) - 0.3 * (1 - hr)), n)
    c = rng.uniform(12, 96, n)
    table = pd.DataFrame(
        {
            "case_id": [f"{c_}_{i:04d}" for i, c_ in enumerate(coll)],
            "collection": coll,
            "pcr": y,
            "age": age,
            "hr": hr,
            "her2": her2,
            "menopausal": menopausal,
            "multifocal": multifocal,
            "ethnicity": rng.choice(["white", "black", "asian", "other"], n, p=[0.75, 0.16, 0.06, 0.03]),
            "vendor": rng.choice(["GE", "Siemens", "Philips"], n, p=[0.64, 0.27, 0.09]),
            "field_strength": rng.choice([1.5, 3.0], n, p=[0.72, 0.28]),
            "dt0": dt[:, 0],
            "dt1": dt[:, 1],
            "dt2": dt[:, 2],
            "rfs_months": np.minimum(t, c),
            "rfs_event": (t <= c).astype(int),
        }
    )
    return table, images, masks


def make_cohort2(n: int = 96, size: int = 16, plane: int = 32, seed: int = 0):
    """Round-2 test cohort (three phases, slices, kinetic features). Code test only."""
    from .pipeline2 import InMemory2
    from .prep2 import KIN_NAMES

    tb, im, mk = make_cohort(n, size=size, seed=seed)
    rng = np.random.default_rng(seed + 1)
    pre, post = im[:, 0] * 0.1 + 1.0, im[:, 1] * 0.1 + 1.0
    late = post + rng.normal(0, 0.02, post.shape).astype(np.float32)
    vols = [np.stack([pre[i], post[i], late[i]]).astype(np.float16) for i in range(n)]
    sls = [rng.normal(1, 0.1, (12, 3, plane, plane)).astype(np.float16) for _ in range(n)]
    kin = rng.normal(0, 1, (n, len(KIN_NAMES))).astype(np.float32)
    kin[rng.random(kin.shape) < 0.05] = np.nan
    tb["dt_last"] = tb["dt2"]
    return InMemory2(tb, vols, [m for m in mk], sls, kin)
