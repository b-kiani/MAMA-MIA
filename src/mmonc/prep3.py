"""Revision preprocessing: whole-breast early-subtraction volumes for the large-context baseline.

    python -m mmonc.prep3 --out /kaggle/working/prepared3 --workers 4

Per case: pre-contrast and first post-contrast phases, reoriented to LPS so that every
collection shares one anatomical frame, resampled to 1.5 mm isotropic, subtraction
(early - pre) divided by the mean pre-contrast signal, clipped to [0, 3] and stored as
uint8 (x 85; Joo et al. also used 8-bit inputs). A box of 48 x 160 x 160 voxels
(7.2 x 24 x 24 cm, superior-inferior x anterior-posterior x left-right) centred on the tumour
is kept, padded with zeros where the field of view is smaller. Writes wb.npy (N, 48, 160, 160)
and wb_ids.csv.
"""
from __future__ import annotations

import argparse
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .data import read_image
from .prep import _file, build_table, find_root

SHAPE = (48, 160, 160)
MM = 1.5


def _iso(img, interp, mm=MM):
    import SimpleITK as sitk

    img = sitk.DICOMOrient(img, "LPS")
    sp, sz = np.array(img.GetSpacing()), np.array(img.GetSize())
    new = [max(1, int(round(n * s / mm))) for n, s in zip(sz, sp)]
    return sitk.Resample(img, new, sitk.Transform(), interp, img.GetOrigin(), (mm, mm, mm), img.GetDirection(), 0,
                         img.GetPixelID())


def whole_case(pre_path, early_path, mask_path):
    import SimpleITK as sitk

    pre = sitk.GetArrayFromImage(_iso(read_image(pre_path, sitk.sitkFloat32), sitk.sitkLinear))
    early = sitk.GetArrayFromImage(_iso(read_image(early_path, sitk.sitkFloat32), sitk.sitkLinear))
    m = sitk.GetArrayFromImage(_iso(read_image(mask_path, sitk.sitkUInt8), sitk.sitkNearestNeighbor)) > 0
    n = [min(a, b, c) for a, b, c in zip(pre.shape, early.shape, m.shape)]
    pre, early, m = pre[:n[0], :n[1], :n[2]], early[:n[0], :n[1], :n[2]], m[:n[0], :n[1], :n[2]]
    ref = float(pre[pre > 0].mean()) if (pre > 0).any() else 1.0
    sub = np.clip((early - pre) / ref, 0, 3)
    c = np.argwhere(m).mean(0).round().astype(int) if m.any() else np.array(sub.shape) // 2
    out = np.zeros(SHAPE, np.float32)
    src, dst = [], []
    for ax in range(3):
        lo = c[ax] - SHAPE[ax] // 2
        a, b = max(lo, 0), min(lo + SHAPE[ax], sub.shape[ax])
        src.append(slice(a, b)); dst.append(slice(a - lo, b - lo))
    out[tuple(dst)] = sub[tuple(src)]
    return np.round(out * 85).astype(np.uint8)


def _one(args):
    root, cid = args
    files = sorted(glob.glob(os.path.join(root, "images", cid, f"{cid}_*")))
    if len(files) < 2:
        raise ValueError(f"{cid}: {len(files)} phases")
    return cid, whole_case(files[0], files[1], _file(os.path.join(root, "segmentations", "expert"), cid))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    root = a.root or find_root()
    os.makedirs(a.out, exist_ok=True)
    ids = list(build_table(root).case_id)
    if a.limit:
        ids = ids[:a.limit]
    arr = np.zeros((len(ids),) + SHAPE, np.uint8)
    ok = np.zeros(len(ids), bool)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for k, res in enumerate(ex.map(_safe, [(root, c) for c in ids], chunksize=4)):
            if res is not None:
                arr[k], ok[k] = res, True
            if (k + 1) % 100 == 0:
                print(f"whole-breast {k + 1}/{len(ids)} {(time.time() - t0) / 60:.1f} min", flush=True)
    np.save(os.path.join(a.out, "wb.npy"), arr)
    pd.DataFrame({"case_id": ids, "ok": ok}).to_csv(os.path.join(a.out, "wb_ids.csv"), index=False)
    print(f"wrote {ok.sum()}/{len(ids)} whole-breast volumes in {(time.time() - t0) / 60:.1f} min; "
          f"mean nonzero fraction {float((arr[ok] > 0).mean()):.3f}", flush=True)


def _safe(args):
    try:
        return _one(args)[1]
    except Exception as e:  # noqa: BLE001
        print("WB FAIL", args[1], repr(e)[:160], flush=True)
        return None


if __name__ == "__main__":
    main()
