"""Preprocess the MAMA-MIA release into cropped arrays + one patient table.

    python -m mmonc.prep --out /kaggle/working/prepared --workers 4 [--limit 20]

Resumable: cases already written are skipped.
"""
from __future__ import annotations

import argparse
import ast
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from .data import preprocess_case
from .radiomics import SIMPLE_NAMES, simple_features

MENO = {"pre": 0.0, "peri": 0.0, "post": 1.0}


def find_root(base="/kaggle/input"):
    hits = glob.glob(os.path.join(base, "**", "clinical_and_imaging_info.xlsx"), recursive=True)
    if not hits:
        raise FileNotFoundError("clinical_and_imaging_info.xlsx not found under " + base)
    return os.path.dirname(hits[0])


def _file(folder, stem):
    hits = sorted(glob.glob(os.path.join(folder, stem + ".*")))
    if not hits:
        raise FileNotFoundError(os.path.join(folder, stem))
    return hits[0]


def build_table(root):
    raw = pd.read_excel(os.path.join(root, "clinical_and_imaging_info.xlsx"))
    t = pd.DataFrame({"case_id": raw.patient_id, "collection": raw.dataset.str.upper(), "pcr": raw.pcr,
                      "age": raw.age, "hr": raw.hr, "er": raw.er, "pr": raw.pr, "her2": raw.her2,
                      "tumor_subtype": raw.tumor_subtype,
                      "menopausal": raw.menopause.astype(str).str.lower().str[:4].str.strip(" (").map(
                          lambda v: MENO.get(v.split()[0] if v.split() else v, np.nan)),
                      "multifocal": raw.multifocal_cancer, "bilateral": raw.bilateral_breast_cancer,
                      "ethnicity": raw.ethnicity, "bmi_group": raw.bmi_group, "vendor": raw.manufacturer,
                      "field_strength": raw.field_strength, "view": raw.view, "site": raw.site,
                      "slice_thickness": raw.slice_thickness, "num_phases": raw.num_phases,
                      "acquisition_date": raw.acquisition_date, "nac_agent": raw.nac_agent})
    times = raw.acquisition_times.apply(lambda v: ast.literal_eval(v) if isinstance(v, str) and v.startswith("[") else None)
    dt = np.full((len(raw), 3), np.nan)
    for i, tt in enumerate(times):
        if tt:
            tt = [float(x) for x in tt]
            dt[i, : min(3, len(tt))] = tt[:3]
    t["dt0"], t["dt1"], t["dt2"] = dt[:, 0], dt[:, 1], dt[:, 2]
    t["dt_last"] = [float(tt[-1]) if tt else np.nan for tt in times]
    # follow-up (days -> months); recurrence event = days_to_recurrence > 0 (0 means none recorded)
    fu = raw.days_to_follow_up / 30.44
    rec = raw.days_to_recurrence
    t["rfs_event"] = np.where(rec.isna() | fu.isna(), np.nan, (rec > 0).astype(float))
    t["rfs_months"] = np.where(t.rfs_event == 1, rec / 30.44, fu)
    return t


def _one(args):
    root, cid, size = args
    img_dir = os.path.join(root, "images", cid)
    phases = [_file(img_dir, f"{cid}_{k:04d}") for k in (0, 1)]
    mask = _file(os.path.join(root, "segmentations", "expert"), cid)
    image, m, info = preprocess_case(phases, mask, size=size, return_info=True)
    return cid, image, m, info


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
    if a.limit:  # stratified small sample for testing: a few per collection
        ids = list(tb.groupby("collection").head(max(1, a.limit // 4)).case_id)
    todo = [c for c in ids if not os.path.exists(os.path.join(a.out, f"{c}.npz"))]
    print(f"root {root}\n{len(ids)} cases, {len(todo)} to do", flush=True)
    t0, done, fails = time.time(), 0, []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(_one, (root, c, a.size)): c for c in todo}
        for f in as_completed(futs):
            c = futs[f]
            try:
                cid, image, m, info = f.result()
                np.savez_compressed(os.path.join(a.out, f"{cid}.npz"), image=image, mask=m,
                                    **{k: np.float32(v) for k, v in info.items()})
            except Exception as e:  # noqa: BLE001
                fails.append((c, repr(e)[:120]))
            done += 1
            if done % 50 == 0 or done == len(todo):
                el = time.time() - t0
                print(f"{done}/{len(todo)}  {el/60:.1f} min  eta {el/done*(len(todo)-done)/60:.1f} min  fails {len(fails)}", flush=True)
    for c, e in fails[:20]:
        print("FAIL", c, e)
    # table + radiomics for every case that has an array
    have = [c for c in ids if os.path.exists(os.path.join(a.out, f"{c}.npz"))]
    rad, extra = {}, {}
    for c in have:
        z = np.load(os.path.join(a.out, f"{c}.npz"))
        rad[c] = simple_features(z["image"].astype(np.float32), z["mask"])
        extra[c] = {k: float(z[k]) for k in ("tumour_mm3", "extent_mm", "crop_mm")}
    r = pd.DataFrame.from_dict(rad, orient="index", columns=SIMPLE_NAMES)
    e = pd.DataFrame.from_dict(extra, orient="index")
    r["log_tumour_mm3"] = np.log1p(e.tumour_mm3)
    r["extent_mm"] = e.extent_mm
    r = r.drop(columns=["volume_vox", "bbox_extent"])  # meaningless after resizing the crop
    r.index.name = "case_id"
    r.to_csv(os.path.join(a.out, "radiomics.csv"))
    tb[tb.case_id.isin(have)].to_csv(os.path.join(a.out, "table.csv"), index=False)
    pd.DataFrame(fails, columns=["case_id", "error"]).to_csv(os.path.join(a.out, "failures.csv"), index=False)
    print(f"wrote {len(have)} cases; {len(fails)} failures; {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
