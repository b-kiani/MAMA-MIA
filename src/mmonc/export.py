"""Print compact, plot-ready results to stdout (one JSON object per line, prefixed EXPORT).

    python -m mmonc.export --final /kaggle/input/.../final

Used to move results out of Kaggle through the run log when file download is unavailable.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from .metrics import net_benefit, net_benefit_all, reliability_points
from .report import load_runs

MAIN = ["Clinical", "Clinical + radiomics", "Image only (ResNet)", "Image only (DenseNet-121)",
        "Image only (EfficientNet-B0)", "Image only (ViT)", "Concatenation fusion", "CARF (proposed)"]


def _thin(x, y, k=60):
    idx = np.unique(np.linspace(0, len(x) - 1, min(k, len(x))).astype(int))
    return [round(float(v), 4) for v in np.asarray(x)[idx]], [round(float(v), 4) for v in np.asarray(y)[idx]]


def emit(kind, **kw):
    print("EXPORT " + json.dumps({"kind": kind, **kw}, separators=(",", ":")), flush=True)


def curves(df, cohort):
    y = df.y.to_numpy()
    for m in [c for c in MAIN if c in df.columns]:
        p = df[m].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        rx, ry = _thin(fpr, tpr)
        pr, rc, _ = precision_recall_curve(y, p)
        px, py = _thin(rc[::-1], pr[::-1])
        cx, cy = reliability_points(y, p)
        t = np.linspace(0.10, 0.60, 26)
        emit("curves", cohort=cohort, model=m, roc=[rx, ry], pr=[px, py],
             cal=[[round(float(v), 4) for v in cx], [round(float(v), 4) for v in cy]],
             dca=[[round(float(v), 3) for v in t], [round(float(v), 4) for v in net_benefit(y, p, t)]])
    t = np.linspace(0.10, 0.60, 26)
    emit("dca_all", cohort=cohort, t=[round(float(v), 3) for v in t],
         nb=[round(float(v), 4) for v in net_benefit_all(y, t)], prevalence=round(float(y.mean()), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True)
    a = ap.parse_args()
    runs = sorted(d for d in glob.glob(os.path.join(a.final, "runs", "*")) if os.path.isdir(d))
    oof, ext = load_runs([d for d in runs if os.path.basename(d) in ("run1", "run3a", "run3b", "run3c")])
    curves(ext, "external")
    curves(oof, "internal")
    for c in sorted(ext.collection.unique()):
        curves(ext[ext.collection == c], c)
    if "gate:CARF" in ext.columns:
        g = pd.concat([oof.assign(collection="ISPY2"), ext])
        emit("gate_hist", **{c: np.histogram(gg["gate:CARF"].dropna(), bins=20, range=(0, 0.4))[0].tolist()
                              for c, gg in g.groupby("collection")})
    for f in sorted(glob.glob(os.path.join(a.final, "*.csv"))) + sorted(glob.glob(os.path.join(a.final, "runs", "run1", "table6*.csv"))):
        t = pd.read_csv(f)
        if len(t) <= 400:
            emit("table", name=os.path.relpath(f, a.final), csv=t.round(4).to_csv(index=False))
    print("EXPORT_DONE", flush=True)


if __name__ == "__main__":
    main()
