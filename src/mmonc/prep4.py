"""Third-review preprocessing: orientation audit, common anatomical orientation, fixed field of view,
kinetic sensitivity variants (paper/analysis_plan_revision3.md).

    python -m mmonc.prep4 --out /kaggle/working/prep4 --workers 4 [--limit 20]

Every case is read once. Three versions of the round-2 inputs are written, each in its own folder
with the layout of mmonc.prep2 (so pipeline2.Data2 reads them):
  orig/     as round 2: arrays in the stored voxel order (no reorientation). Identical to mmonc.prep2.
  lps/      every image and mask reoriented to LPS (sitk.DICOMOrient) before cropping, so that array
            axes are (superior-inferior, anterior-posterior, right-left) in every collection; phases are
            resampled onto the mask grid if their geometry differs.
  lps_fov/  as lps/, but a fixed physical field of view (FOV_MM cube centred on the tumour) instead of the
            tumour-scaled cube, so image-only models keep absolute size.
Also written:
  audit.csv              per case: stored orientation code of each phase and the mask, sizes, spacings,
                         origins, direction agreement, crop coverage, fixed-FOV truncation, and the
                         approximate tissue coverage of the larger-context (prep3) box.
  kinetics_variants.csv  per case and variant: the kinetic features under the implementation variants.
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
from .prep2 import KIN_NAMES, kinetic_features, pack

FOV_MM = 96.0
VARIANTS = {  # name -> kinetic_features keyword arguments (prespecified, analysis_plan_revision3.md)
    "default": {},
    "no_smoothing": {"sigma": 0.0},
    "floor_0.05": {"floor": 0.05},
    "floor_0.5": {"floor": 0.5},
    "ser_unclipped": {"ser_floor": 0.01, "ser_clip": 1e3},
    "mask_eroded": {"mask_op": -1},
    "mask_dilated": {"mask_op": 1},
}
VERSIONS = ("orig", "lps", "lps_fov")


def _code(img):
    import SimpleITK as sitk

    return sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(img.GetDirection())


def _geom(img):
    return np.array(img.GetSize()), np.array(img.GetSpacing()), np.array(img.GetOrigin()), np.array(img.GetDirection())


def crop(phases, mimg, margin=0.5, min_mm=48.0, fixed_mm=None):
    """Cube around the tumour on a 1 mm grid from SimpleITK images that share the mask grid.
    Returns phases[k, D, D, D], mask[D, D, D], info. The cube is clipped to the image and zero-padded."""
    import SimpleITK as sitk

    marr = sitk.GetArrayFromImage(mimg) > 0
    if not marr.any():
        raise ValueError("empty mask")
    sp = np.array(mimg.GetSpacing())
    idx = np.argwhere(marr)[:, ::-1]
    lo, hi = idx.min(0), idx.max(0) + 1
    extent = float(((hi - lo) * sp).max())
    side = float(fixed_mm) if fixed_mm else max(extent * (1 + margin), float(min_mm))
    half = np.ceil(side / 2 / sp).astype(int)
    centre = (lo + hi) // 2
    size = np.array(mimg.GetSize())
    a, b = np.maximum(centre - half, 0), np.minimum(centre + half, size)
    info = {"side_mm": side, "extent_mm": extent, "coverage": float(np.prod(b - a) / np.prod(2 * half)),
            "tumour_truncated": bool(np.any(lo < a) or np.any(hi > b))}

    def roi(img, interp):
        r = sitk.RegionOfInterest(img, [int(v) for v in (b - a)], [int(v) for v in a])
        new = [max(1, int(round(n * s))) for n, s in zip(r.GetSize(), r.GetSpacing())]
        r = sitk.Resample(r, new, sitk.Transform(), interp, r.GetOrigin(), (1.0, 1.0, 1.0), r.GetDirection(), 0, r.GetPixelID())
        return sitk.GetArrayFromImage(r)

    m = roi(mimg, sitk.sitkNearestNeighbor) > 0
    ph = [roi(p, sitk.sitkLinear) for p in phases]
    D = int(max(m.shape))
    pad = [((D - n) // 2, D - n - (D - n) // 2) for n in m.shape]
    return np.stack([np.pad(p, pad) for p in ph]).astype(np.float32), np.pad(m, pad), info


def context_coverage(pre_lps, m_lps):
    """Approximate tissue coverage of the larger-context box of mmonc.prep3 (48 x 160 x 160 voxels at 1.5 mm,
    centred on the tumour). Tissue = pre-contrast signal above the Otsu threshold; only the side of the body
    that holds the tumour and the region no more than 30 mm posterior to the tumour centre are counted."""
    import SimpleITK as sitk

    from .prep3 import MM, SHAPE, _iso

    pre = _iso(pre_lps, sitk.sitkLinear)
    m = sitk.GetArrayFromImage(_iso(m_lps, sitk.sitkNearestNeighbor)) > 0
    fg = sitk.GetArrayFromImage(sitk.OtsuThreshold(pre, 0, 1)) > 0
    n = [min(x, y) for x, y in zip(fg.shape, m.shape)]
    fg, m = fg[:n[0], :n[1], :n[2]], m[:n[0], :n[1], :n[2]]
    if not m.any() or not fg.any():
        return {"ctx_coverage": np.nan, "ctx_si_tissue_mm": np.nan}
    c = np.argwhere(m).mean(0).round().astype(int)
    mid = int(np.argwhere(fg)[:, 2].mean())
    side = np.arange(fg.shape[2]) >= mid if c[2] >= mid else np.arange(fg.shape[2]) < mid
    ant = np.arange(fg.shape[1]) <= c[1] + int(30 / MM)
    t = fg & side[None, None, :] & ant[None, :, None]
    box = np.zeros_like(t)
    sl = tuple(slice(max(c[ax] - SHAPE[ax] // 2, 0), max(c[ax] - SHAPE[ax] // 2 + SHAPE[ax], 0)) for ax in range(3))
    box[sl] = True
    zs = np.flatnonzero(t.any((1, 2)))
    if not t.any():
        return {"ctx_coverage": np.nan, "ctx_si_tissue_mm": np.nan}
    return {"ctx_coverage": float((t & box).sum() / t.sum()),
            "ctx_si_tissue_mm": float((zs.max() - zs.min() + 1) * MM) if len(zs) else np.nan}


def _one(args):
    import SimpleITK as sitk

    root, cid, size = args
    files = sorted(glob.glob(os.path.join(root, "images", cid, f"{cid}_*")))
    if len(files) < 2:
        raise ValueError(f"{cid}: {len(files)} phases")
    pick = [files[0], files[1], files[-1]]
    has_late = len(files) > 2
    mimg = read_image(_file(os.path.join(root, "segmentations", "expert"), cid), sitk.sitkUInt8)
    phs = [read_image(p, sitk.sitkFloat32) for p in pick]
    au = {"case_id": cid, "n_phase_files": len(files), "code_mask": _code(mimg)}
    gm = _geom(mimg)
    au.update({"size_mask": "x".join(map(str, gm[0])), "spacing_mask": "x".join(f"{v:.3f}" for v in gm[1])})
    match = True
    for k, (lab, p) in enumerate(zip(("pre", "early", "late"), phs)):
        g = _geom(p)
        au[f"code_{lab}"] = _code(p)
        au[f"size_eq_{lab}"] = bool(np.array_equal(g[0], gm[0]))
        au[f"spacing_maxdiff_{lab}"] = float(np.abs(g[1] - gm[1]).max())
        au[f"origin_maxdiff_mm_{lab}"] = float(np.abs(g[2] - gm[2]).max())
        au[f"direction_maxdiff_{lab}"] = float(np.abs(g[3] - gm[3]).max())
        match &= au[f"size_eq_{lab}"] and au[f"spacing_maxdiff_{lab}"] < 1e-3 and au[f"origin_maxdiff_mm_{lab}"] < 0.5 and au[f"direction_maxdiff_{lab}"] < 1e-3
    au["geometry_match"] = bool(match)
    out = {}
    # orig: as round 2 (index-space crop of the stored arrays)
    ph, m, info = crop(phs, mimg)
    out["orig"] = pack(ph, m, size, has_late=has_late)
    kin_orig = kinetic_features(ph[0], ph[1], ph[2], m, has_late)
    au.update({f"orig_{k}": v for k, v in info.items()})
    # lps: common anatomical orientation; phases put on the mask grid if their geometry differs
    ml = sitk.DICOMOrient(mimg, "LPS")
    pl = [sitk.DICOMOrient(p, "LPS") for p in phs]
    if not match:
        pl = [sitk.Resample(p, ml, sitk.Transform(), sitk.sitkLinear, 0.0, p.GetPixelID()) for p in pl]
    au["code_after"] = _code(ml)
    ph, m, info = crop(pl, ml)
    out["lps"] = pack(ph, m, size, has_late=has_late)
    au.update({f"lps_{k}": v for k, v in info.items()})
    kin = {v: kinetic_features(ph[0], ph[1], ph[2], m, has_late, **kw) for v, kw in VARIANTS.items()}
    d = np.array([kin["default"][k] - kin_orig[k] for k in KIN_NAMES], float)
    au["kin_orig_vs_lps_maxabs"] = float(np.nanmax(np.abs(d))) if np.isfinite(d).any() else np.nan
    # lps_fov: fixed physical field of view
    ph, m, info = crop(pl, ml, fixed_mm=FOV_MM)
    out["lps_fov"] = pack(ph, m, size, has_late=has_late)
    au.update({f"fov_{k}": v for k, v in info.items()})
    try:
        au.update(context_coverage(pl[0], ml))
    except Exception as e:  # noqa: BLE001
        au["ctx_error"] = repr(e)[:120]
    return cid, out, kin, kin_orig, au


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    root = a.root or find_root()
    for v in VERSIONS:
        os.makedirs(os.path.join(a.out, v), exist_ok=True)
    tb = build_table(root)
    ids = list(tb.case_id)
    if a.limit:
        ids = list(tb.groupby("collection").head(max(1, a.limit // 4)).case_id)
    print(f"root {root}\n{len(ids)} cases", flush=True)
    t0, done, fails, audit, kv, ko, nph = time.time(), 0, [], [], [], {}, {}
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(_one, (root, c, a.size)): c for c in ids}
        for f in as_completed(futs):
            c = futs[f]
            try:
                cid, out, kin, kin_orig, au = f.result()
                for v in VERSIONS:
                    vol, mk, sl, slm = out[v]
                    kk = kin_orig if v == "orig" else kin["default"]
                    np.savez_compressed(os.path.join(a.out, v, f"{cid}.npz"), vol=vol, mask=mk, sl=sl, slm=slm,
                                        n_phases=np.int16(au["n_phase_files"]), kin=np.array([kk[k] for k in KIN_NAMES], np.float32))
                audit.append(au)
                ko[cid] = kin_orig
                nph[cid] = au["n_phase_files"]
                for name, feats in kin.items():
                    kv.append({"case_id": cid, "variant": name, **feats})
            except Exception as e:  # noqa: BLE001
                fails.append((c, repr(e)[:160]))
            done += 1
            if done % 100 == 0 or done == len(ids):
                el = time.time() - t0
                print(f"{done}/{len(ids)}  {el / 60:.1f} min  eta {el / done * (len(ids) - done) / 60:.1f} min  fails {len(fails)}", flush=True)
    for c, e in fails[:20]:
        print("FAIL", c, e, flush=True)
    A = pd.DataFrame(audit).merge(tb[["case_id", "collection", "view"]], on="case_id")
    A.to_csv(os.path.join(a.out, "audit.csv"), index=False)
    K = pd.DataFrame(kv)
    K.to_csv(os.path.join(a.out, "kinetics_variants.csv"), index=False)
    have = set(A.case_id)
    t = tb[tb.case_id.isin(have)].copy()
    t["n_phases_found"] = t.case_id.map(nph)
    for v in VERSIONS:
        t.to_csv(os.path.join(a.out, v, "table.csv"), index=False)
        src = ko if v == "orig" else {c: g for c, g in zip(K[K.variant == "default"].case_id, K[K.variant == "default"][KIN_NAMES].to_dict("records"))}
        k = pd.DataFrame.from_dict(src, orient="index")[KIN_NAMES]
        k.index.name = "case_id"
        k.to_csv(os.path.join(a.out, v, "kinetics.csv"))
    pd.DataFrame(fails, columns=["case_id", "error"]).to_csv(os.path.join(a.out, "failures.csv"), index=False)
    print(f"wrote {len(have)} cases; {len(fails)} failures; {(time.time() - t0) / 60:.1f} min", flush=True)
    print(A.groupby(["collection", "view"]).agg(n=("case_id", "size"), code_pre=("code_pre", lambda s: s.value_counts().to_dict()),
                                                 code_mask=("code_mask", lambda s: s.value_counts().to_dict()),
                                                 geometry_match=("geometry_match", "mean"), code_after=("code_after", lambda s: s.value_counts().to_dict()),
                                                 orig_coverage=("orig_coverage", "mean"), fov_trunc=("fov_tumour_truncated", "mean"),
                                                 ctx_cov=("ctx_coverage", "median"), kin_diff=("kin_orig_vs_lps_maxabs", "median")).to_string(), flush=True)


if __name__ == "__main__":
    main()
