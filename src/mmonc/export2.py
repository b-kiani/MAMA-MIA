"""Print round-2 outputs to the run log as gzip+base64 chunks (file download from Kaggle is unreliable).

    python -m mmonc.export2 --dir /kaggle/working/round2 --extra table.csv kinetics.csv

Lines look like:  X2 <name> <part>/<parts> <payload>.   mmonc.export2.decode(log_text, out_dir) restores the files.
"""
from __future__ import annotations

import argparse
import base64
import glob
import gzip
import os
import re

CHUNK = 6000


def emit(path, name=None):
    name = name or os.path.basename(path)
    b = base64.b64encode(gzip.compress(open(path, "rb").read(), 9)).decode()
    parts = [b[i:i + CHUNK] for i in range(0, len(b), CHUNK)] or [""]
    for i, p in enumerate(parts):
        print(f"X2 {name} {i + 1}/{len(parts)} {p}", flush=True)


def decode(text: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    files = {}
    for name, i, n, payload in re.findall(r"X2 (\S+) (\d+)/(\d+) ([A-Za-z0-9+/=]*)", text):
        files.setdefault(name, {})[int(i)] = (int(n), payload)
    done = []
    for name, parts in files.items():
        n = next(iter(parts.values()))[0]
        if len(parts) != n:
            print(f"incomplete {name}: {len(parts)}/{n}")
            continue
        raw = gzip.decompress(base64.b64decode("".join(parts[i][1] for i in range(1, n + 1))))
        open(os.path.join(out_dir, name), "wb").write(raw)
        done.append(name)
    return done


def samples(data_dir, out_path, per_class=1):
    """Example cases for a figure: per collection, the pCR and the non-pCR case of median tumour volume.

    Saves the in-plane slice through the tumour (pre, early, late; signal relative to pre-contrast) and its mask.
    """
    import numpy as np
    import pandas as pd

    t = pd.read_csv(os.path.join(data_dir, "table.csv"))
    k = pd.read_csv(os.path.join(data_dir, "kinetics.csv"), index_col=0)
    t = t[t.pcr.notna() & t.hr.notna() & t.her2.notna()].copy()
    t["v"] = k.loc[t.case_id, "log_vol_mm3"].to_numpy()
    img, msk, meta = [], [], []
    for (c, y), g in t.groupby(["collection", "pcr"]):
        g = g.assign(d=(g.v - g.v.median()).abs()).sort_values(["d", "case_id"]).head(per_class)
        for r in g.itertuples():
            z = np.load(os.path.join(data_dir, f"{r.case_id}.npz"))
            img.append(z["sl"][1]); msk.append(z["slm"][1])
            meta.append(f"{r.case_id},{c},{int(y)},{int(r.hr)},{int(r.her2)},{r.age:.0f},{r.field_strength},{r.view}")
    np.savez_compressed(out_path, img=np.stack(img), msk=np.stack(msk), meta=np.array(meta))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--samples-from")
    a = ap.parse_args()
    if a.samples_from:
        try:
            samples(a.samples_from, os.path.join(a.dir, "samples.npz"))
            emit(os.path.join(a.dir, "samples.npz"))
        except Exception as e:  # noqa: BLE001
            print("SAMPLES FAILED", repr(e), flush=True)
    for f in sorted(glob.glob(os.path.join(a.dir, "*.csv"))) + list(a.extra):
        if os.path.exists(f):
            emit(f)
    print("X2_DONE", flush=True)


if __name__ == "__main__":
    main()
