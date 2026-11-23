"""Round-2 preprocessing: three DCE phases (pre, early, late), kinetic features, 2.5D slices.

    python -m mmonc.prep2 --out /kaggle/working/prepared2 --workers 4 [--limit 20]

Per case it writes <case>.npz with
  vol    (3, S, S, S) float16   pre, early, late, each divided by the mean pre-contrast signal of the crop
  mask   (S, S, S) uint8
  sl     (12, 3, P, P) float16  four slices through the tumour in each of the three planes
  slm    (12, P, P) uint8       the tumour mask on those slices
and it writes table.csv (as round 1, plus n_phases and dt_late) and kinetics.csv.
Resumable: cases already written are skipped.
"""
from __future__ import annotations

import argparse
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from .data import read_image
from .prep import _file, build_table, find_root

KIN_NAMES = ["log_vol_mm3", "extent_mm", "surface_ratio", "log_ftv_mm3", "ftv_frac", "pe_mean", "pe_median",
             "pe_p10", "pe_p90", "pe_sd", "pe_entropy", "pe_local_sd", "pe_late_mean", "ser_mean", "ser_median",
             "ser_p90", "frac_washout", "frac_plateau", "frac_persistent", "peri_pe_mean", "peri_ratio",
             "rim_core_ratio"]
SER_NAMES = ["pe_late_mean", "ser_mean", "ser_median", "ser_p90", "frac_washout", "frac_plateau", "frac_persistent"]


def kinetic_features(pre, early, late, mask, has_late=True, sigma=1.0, floor=0.2, ser_floor=0.05, ser_clip=5.0,
                     mask_op=0, min_enh=10):
    """Enhancement kinetics inside a tumour mask on a 1 mm grid, exactly as implemented:

      base = max(G_sigma * pre, floor * mean(pre > 0 in the crop))     (smoothed, floored denominator)
      PE   = (early - pre) / base ;  PE_late = (late - pre) / base
      SER  = clip(PE / max(PE_late, ser_floor), 0, ser_clip), on voxels with PE >= 0.7, only if >= min_enh such voxels
      PE-threshold volume = number of mask voxels with PE >= 0.7 (not the official I-SPY FTV, which adds
      background filtering and connectivity criteria).
    The defaults reproduce the round-2 features; the keyword arguments give the sensitivity variants
    (mask_op > 0 dilates and < 0 erodes the mask by that many voxels).
    """
    from scipy import ndimage as ndi

    m = mask.astype(bool)
    if mask_op > 0:
        m = ndi.binary_dilation(m, iterations=mask_op)
    elif mask_op < 0:
        e = ndi.binary_erosion(m, iterations=-mask_op)
        m = e if e.sum() >= 10 else m
    ref = float(pre[pre > 0].mean()) if (pre > 0).any() else 1.0
    sm = ndi.gaussian_filter(pre.astype(np.float32), sigma) if sigma > 0 else pre.astype(np.float32)
    base = np.maximum(sm, floor * ref)
    pe = (early - pre) / base
    pel = (late - pre) / base
    v = float(m.sum())
    er = ndi.binary_erosion(m, iterations=2)
    rim, core = m & ~er, er
    shell = ndi.binary_dilation(m, iterations=5) & ~ndi.binary_dilation(m, iterations=1)
    x = pe[m]
    enh = m & (pe >= 0.7)
    f = dict.fromkeys(KIN_NAMES, np.nan)
    idx = np.argwhere(m)
    f["log_vol_mm3"] = np.log1p(v)
    f["extent_mm"] = float((idx.max(0) - idx.min(0) + 1).max())
    f["surface_ratio"] = float((m & ~ndi.binary_erosion(m)).sum() / max(v, 1.0) ** (2 / 3))
    f["log_ftv_mm3"] = np.log1p(float(enh.sum()))
    f["ftv_frac"] = float(enh.sum() / max(v, 1.0))
    f["pe_mean"], f["pe_median"], f["pe_sd"] = float(x.mean()), float(np.median(x)), float(x.std())
    f["pe_p10"], f["pe_p90"] = (float(q) for q in np.percentile(x, [10, 90]))
    h = np.histogram(np.clip(x, 0, 3), bins=32, range=(0, 3))[0].astype(float)
    p = h[h > 0] / h.sum()
    f["pe_entropy"] = float(-(p * np.log(p)).sum())
    mu = ndi.uniform_filter(pe, 3)
    sd = np.sqrt(np.maximum(ndi.uniform_filter(pe * pe, 3) - mu * mu, 0))
    f["pe_local_sd"] = float(sd[m].mean())
    f["peri_pe_mean"] = float(pe[shell].mean()) if shell.any() else np.nan
    f["peri_ratio"] = f["peri_pe_mean"] / (abs(f["pe_mean"]) + 1e-3)
    if core.sum() >= 10 and rim.sum() >= 10:
        f["rim_core_ratio"] = float(pe[rim].mean() / (abs(pe[core].mean()) + 1e-3))
    if has_late:
        f["pe_late_mean"] = float(pel[m].mean())
        if enh.sum() >= min_enh:
            ser = pe[enh] / np.maximum(pel[enh], ser_floor)
            ser = np.clip(ser, 0, ser_clip)
            f["ser_mean"], f["ser_median"], f["ser_p90"] = float(ser.mean()), float(np.median(ser)), float(np.percentile(ser, 90))
            f["frac_washout"] = float((ser > 1.1).mean())
            f["frac_plateau"] = float(((ser >= 0.9) & (ser <= 1.1)).mean())
            f["frac_persistent"] = float((ser < 0.9).mean())
    return f


def crop_case(phase_paths, mask_path, margin=0.5, min_mm=48):
    """Cube around the tumour on a 1 mm grid. Returns (phases[k, D, D, D] float32, mask[D, D, D] bool)."""
    import SimpleITK as sitk

    mimg = read_image(mask_path, sitk.sitkUInt8)
    marr = sitk.GetArrayFromImage(mimg) > 0
    if not marr.any():
        raise ValueError(f"empty mask: {mask_path}")
    sp = np.array(mimg.GetSpacing())  # x, y, z
    idx = np.argwhere(marr)[:, ::-1]  # -> x, y, z
    lo, hi = idx.min(0), idx.max(0) + 1
    side = max(float(((hi - lo) * sp).max()) * (1 + margin), float(min_mm))
    half = np.ceil(side / 2 / sp).astype(int)
    centre = (lo + hi) // 2
    size = np.array(mimg.GetSize())
    a, b = np.maximum(centre - half, 0), np.minimum(centre + half, size)

    def roi(img, interp):
        r = sitk.RegionOfInterest(img, [int(v) for v in (b - a)], [int(v) for v in a])
        new = [max(1, int(round(n * s))) for n, s in zip(r.GetSize(), r.GetSpacing())]
        r = sitk.Resample(r, new, sitk.Transform(), interp, r.GetOrigin(), (1.0, 1.0, 1.0), r.GetDirection(), 0, r.GetPixelID())
        return sitk.GetArrayFromImage(r)

    m = roi(mimg, sitk.sitkNearestNeighbor) > 0
    ph = [roi(read_image(p, sitk.sitkFloat32), sitk.sitkLinear) for p in phase_paths]
    D = int(max(m.shape))
    pad = [((D - n) // 2, D - n - (D - n) // 2) for n in m.shape]
    return np.stack([np.pad(p, pad) for p in ph]).astype(np.float32), np.pad(m, pad)


def pack(ph, m, size=64, plane=112, has_late=True):
    """Resize the cube for the 3D models and cut 12 slices for the 2.5D model."""
    import torch
    import torch.nn.functional as F

    pre = ph[0]
    ref = float(pre[pre > 0].mean()) if (pre > 0).any() else 1.0
    x = torch.from_numpy(np.clip(ph / ref, 0, 20))[None]
    vol = F.interpolate(x, size=(size,) * 3, mode="trilinear", align_corners=False)[0].numpy().astype(np.float16)
    mk = F.interpolate(torch.from_numpy(m[None, None].astype(np.float32)), size=(size,) * 3, mode="nearest")[0, 0].numpy().astype(np.uint8)
    big = F.interpolate(x, size=(plane,) * 3, mode="trilinear", align_corners=False)[0].numpy()
    mb = F.interpolate(torch.from_numpy(m[None, None].astype(np.float32)), size=(plane,) * 3, mode="nearest")[0, 0].numpy() > 0
    sl, slm = [], []
    for ax in range(3):
        pos = np.flatnonzero(mb.any(axis=tuple(i for i in range(3) if i != ax)))
        for q in (0.2, 0.4, 0.6, 0.8):
            k = int(pos[min(len(pos) - 1, int(q * len(pos)))])
            sl.append(np.take(big, k, axis=ax + 1))
            slm.append(np.take(mb, k, axis=ax))
    return vol, mk, np.stack(sl).astype(np.float16), np.stack(slm).astype(np.uint8)


def _one(args):
    root, cid, size = args
    files = sorted(glob.glob(os.path.join(root, "images", cid, f"{cid}_*")))
    if len(files) < 2:
        raise ValueError(f"{cid}: {len(files)} phases")
    pick = [files[0], files[1], files[-1]]
    ph, m = crop_case(pick, _file(os.path.join(root, "segmentations", "expert"), cid))
    has_late = len(files) > 2
    kin = kinetic_features(ph[0], ph[1], ph[2], m, has_late)
    vol, mk, sl, slm = pack(ph, m, size, has_late=has_late)
    return cid, vol, mk, sl, slm, kin, len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    root = a.root or find_root()
    os.makedirs(a.out, exist_ok=True)
    tb = build_table(root)
    ids = list(tb.case_id)
    if a.limit:
        ids = list(tb.groupby("collection").head(max(1, a.limit // 4)).case_id)
    todo = [c for c in ids if not os.path.exists(os.path.join(a.out, f"{c}.npz"))]
    print(f"root {root}\n{len(ids)} cases, {len(todo)} to do", flush=True)
    t0, done, fails = time.time(), 0, []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(_one, (root, c, a.size)): c for c in todo}
        for f in as_completed(futs):
            c = futs[f]
            try:
                cid, vol, mk, sl, slm, kin, n = f.result()
                np.savez_compressed(os.path.join(a.out, f"{cid}.npz"), vol=vol, mask=mk, sl=sl, slm=slm, n_phases=np.int16(n),
                                    kin=np.array([kin[k] for k in KIN_NAMES], np.float32))
            except Exception as e:  # noqa: BLE001
                fails.append((c, repr(e)[:160]))
            done += 1
            if done % 50 == 0 or done == len(todo):
                el = time.time() - t0
                print(f"{done}/{len(todo)}  {el/60:.1f} min  eta {el/done*(len(todo)-done)/60:.1f} min  fails {len(fails)}", flush=True)
    for c, e in fails[:20]:
        print("FAIL", c, e)
    have = [c for c in ids if os.path.exists(os.path.join(a.out, f"{c}.npz"))]
    kin, nph = {}, {}
    for c in have:
        z = np.load(os.path.join(a.out, f"{c}.npz"))
        kin[c], nph[c] = z["kin"], int(z["n_phases"])
    k = pd.DataFrame.from_dict(kin, orient="index", columns=KIN_NAMES)
    k.index.name = "case_id"
    k.to_csv(os.path.join(a.out, "kinetics.csv"))
    t = tb[tb.case_id.isin(have)].copy()
    t["n_phases_found"] = t.case_id.map(nph)
    t.to_csv(os.path.join(a.out, "table.csv"), index=False)
    pd.DataFrame(fails, columns=["case_id", "error"]).to_csv(os.path.join(a.out, "failures.csv"), index=False)
    print(f"wrote {len(have)} cases; {len(fails)} failures; {(time.time()-t0)/60:.1f} min", flush=True)
    print("phases found:", t.groupby("collection").n_phases_found.describe()[["min", "50%", "max"]].to_string(), flush=True)
    print(k.describe().T[["count", "mean", "std"]].round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
