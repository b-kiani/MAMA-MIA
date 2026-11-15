"""Statistics for the round-2 experiments, computed from per-case predictions.

    python -m mmonc.report2 --dir results/round2/raw --out results/round2

--dir holds the files restored by mmonc.export2.decode: preds_devext_*.csv, preds_controls_*.csv,
preds_loco_*.csv and table.csv.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import metrics, stats

META = ("case_id", "collection", "y", "set")


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def load_devext(d):
    out = None
    for f in sorted(glob.glob(os.path.join(d, "preds_devext_*.csv"))):
        t = pd.read_csv(f)
        out = t if out is None else out.merge(t.drop(columns=["collection", "y", "set"]), on="case_id")
    return out


def models_of(df):
    return [c for c in df.columns if c not in META and not c.startswith("gate:")]


def devext_tables(P, n_boot=2000):
    dev, ext = P[P.set == "dev"], P[P.set == "ext"]
    rows, cmp_, per = [], [], []
    for m in models_of(P):
        thr = metrics.youden_threshold(dev.y, dev[m])
        a = metrics.summary(dev.y.to_numpy(), dev[m].to_numpy(), thr, n_boot)
        b = metrics.summary(ext.y.to_numpy(), ext[m].to_numpy(), thr, n_boot)
        rows.append({"model": m, "internal_auc": a["auc"], "internal_lo": a["auc_lo"], "internal_hi": a["auc_hi"],
                     "external_auc": b["auc"], "external_lo": b["auc_lo"], "external_hi": b["auc_hi"],
                     "ext_slope": b["slope"], "ext_intercept": b["intercept"], "ext_brier": b["brier"],
                     "ext_sensitivity": b["sensitivity"], "ext_specificity": b["specificity"],
                     "ext_balanced_accuracy": b["balanced_accuracy"]})
        for c, g in ext.groupby("collection"):
            per.append({"model": m, "collection": c, "n": len(g), "events": int(g.y.sum()), "auc": roc_auc_score(g.y, g[m])})
        if m != "Clinical":
            d = stats.delong(ext.y, ext["Clinical"], ext[m])
            di = stats.delong(dev.y, dev["Clinical"], dev[m])
            cmp_.append({"model": m, "internal_diff": di["diff"], "internal_ci_lo": di["ci_lo"], "internal_ci_hi": di["ci_hi"],
                         "diff": d["diff"], "se": d["se"], "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"], "p": d["p"],
                         "verdict": stats.verdict(d, 0.03),
                         "detectable_diff_80pct_power": (1.96 + 0.8416) * d["se"],
                         "corr_with_clinical_logit": float(np.corrcoef(_logit(ext["Clinical"]), _logit(ext[m]))[0, 1])})
    c = pd.DataFrame(cmp_)
    if len(c):
        c["p_holm"] = stats.holm(c.p)
    return pd.DataFrame(rows), c, pd.DataFrame(per)


def _lr_test(y, base, extra):
    """Likelihood-ratio test of adding `extra` (n,) to a logistic model on `base` (n, k)."""
    def ll(X):
        m = LogisticRegression(C=1e6, max_iter=2000).fit(X, y)
        p = np.clip(m.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
        return float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum()), m
    l0, _ = ll(base)
    l1, m1 = ll(np.column_stack([base, extra]))
    return {"lr_chi2": 2 * (l1 - l0), "p": float(sps.chi2.sf(max(2 * (l1 - l0), 0), 1)), "coef": float(m1.coef_[0, -1])}


def added_value(P, T, n_boot=1000):
    """Does an image score add to the clinical logit in external data, with no gate in the way?

    1. Likelihood-ratio test of adding the image score to the clinical logit.
    2. Stacking: logistic model on (clinical logit, image logit) refitted *in the external data*
       with 5-fold cross-validation - an optimistic upper bound for what the image score can add.
    Both overall and within receptor subtype (clinical logit replaced by age within a subtype).
    """
    ext = P[P.set == "ext"].merge(T[["case_id", "age", "hr", "her2", "tumor_subtype"]], on="case_id", how="left")
    y = ext.y.to_numpy()
    zc = _logit(ext["Clinical"])
    img = [m for m in models_of(P) if m.startswith("Image only") or m in ("Kinetic radiomics",)]
    rows = []
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    for m in img:
        zi = _logit(ext[m])
        r = _lr_test(y, zc.reshape(-1, 1), zi)
        st = cross_val_predict(LogisticRegression(C=1e6, max_iter=2000), np.column_stack([zc, zi]), y, cv=cv, method="predict_proba")[:, 1]
        base = cross_val_predict(LogisticRegression(C=1e6, max_iter=2000), zc.reshape(-1, 1), y, cv=cv, method="predict_proba")[:, 1]
        d = stats.delong(y, base, st)
        rows.append({"group": "All external", "n": len(y), "events": int(y.sum()), "image_score": m, "image_auc": roc_auc_score(y, ext[m]),
                     **r, "stack_auc_gain": d["diff"], "stack_ci_lo": d["ci_lo"], "stack_ci_hi": d["ci_hi"]})
        hr, her2 = ext.hr.to_numpy(float), ext.her2.to_numpy(float)
        groups = {"HR+/HER2-": (hr == 1) & (her2 == 0), "HER2+": her2 == 1, "Triple negative": (hr == 0) & (her2 == 0)}
        for g, s in groups.items():
            if min(y[s].sum(), (1 - y[s]).sum()) < 10:
                continue
            age = np.nan_to_num(ext.age.to_numpy(float)[s], nan=np.nanmean(ext.age))
            b = np.column_stack([age, hr[s]]) if g == "HER2+" else age.reshape(-1, 1)
            r = _lr_test(y[s], b, zi[s])
            a = metrics.bootstrap_ci(roc_auc_score, y[s], ext[m].to_numpy()[s], n_boot=n_boot)
            rows.append({"group": g, "n": int(s.sum()), "events": int(y[s].sum()), "image_score": m, "image_auc": a[0],
                         "image_auc_lo": a[1], "image_auc_hi": a[2], **r})
    t = pd.DataFrame(rows)
    if len(t):
        t["p_holm"] = np.nan
        for m in img:  # Holm across the subtype tests of each image score
            s = (t.image_score == m) & (t.group != "All external")
            if s.any():
                t.loc[s, "p_holm"] = stats.holm(t.loc[s, "p"])
    return t


def subtype_auc(P, T, n_boot=1000):
    ext = P[P.set == "ext"].merge(T[["case_id", "hr", "her2"]], on="case_id", how="left")
    y, hr, her2 = ext.y.to_numpy(), ext.hr.to_numpy(float), ext.her2.to_numpy(float)
    groups = {"HR+/HER2-": (hr == 1) & (her2 == 0), "HER2+": her2 == 1, "Triple negative": (hr == 0) & (her2 == 0)}
    rows = []
    for g, s in groups.items():
        for m in models_of(P):
            a = metrics.bootstrap_ci(roc_auc_score, y[s], ext[m].to_numpy()[s], n_boot=n_boot)
            rows.append({"group": g, "n": int(s.sum()), "events": int(y[s].sum()), "pcr_rate": float(y[s].mean()),
                         "model": m, "auc": a[0], "lo": a[1], "hi": a[2]})
    return pd.DataFrame(rows)


def recalibration(P):
    """Intercept-only update per external collection, estimated with 5-fold cross-validation."""
    ext = P[P.set == "ext"]
    rows = []
    for m in [x for x in ("Clinical", "CARF, kinetic", "CARF, pretrained 2.5D") if x in ext]:
        for c, g in ext.groupby("collection"):
            y, z = g.y.to_numpy(), _logit(g[m])
            new = np.zeros(len(y))
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(z, y):
                a = 0.0
                for _ in range(50):
                    q = metrics.sigmoid(a + z[tr])
                    a += (y[tr] - q).sum() / max(1e-9, (q * (1 - q)).sum())
                new[te] = metrics.sigmoid(a + z[te])
            b0, b1 = metrics.calibration(y, metrics.sigmoid(z)), metrics.calibration(y, new)
            rows.append({"model": m, "collection": c, "n": len(y), "intercept_before": b0["intercept"], "intercept_after": b1["intercept"],
                         "brier_before": b0["brier"], "brier_after": b1["brier"], "ece_before": b0["ece"], "ece_after": b1["ece"]})
    return pd.DataFrame(rows)


def controls_table(d, n_boot=1000):
    fs = sorted(glob.glob(os.path.join(d, "preds_controls_*.csv")))
    if not fs:
        return pd.DataFrame()
    C = pd.concat([pd.read_csv(f) for f in fs])
    rows = []
    for (t, s), g in C.groupby(["target", "set"]):
        if g.y.nunique() < 2:
            continue
        a = metrics.bootstrap_ci(roc_auc_score, g.y.to_numpy(), g.p.to_numpy(), n_boot=n_boot)
        rows.append({"target": t, "cohort": "I-SPY2 cross-validation" if s == "dev" else "External", "n": len(g),
                     "positives": int(g.y.sum()), "auc": a[0], "lo": a[1], "hi": a[2]})
        if s == "ext":
            for c, gg in g.groupby("collection"):
                if gg.y.nunique() == 2 and min(gg.y.sum(), (1 - gg.y).sum()) >= 10:
                    rows.append({"target": t, "cohort": c, "n": len(gg), "positives": int(gg.y.sum()), "auc": roc_auc_score(gg.y, gg.p)})
    return pd.DataFrame(rows)


def loco_tables(d):
    fs = sorted(glob.glob(os.path.join(d, "preds_loco_*.csv")))
    if not fs:
        return pd.DataFrame(), pd.DataFrame()
    L = pd.concat([pd.read_csv(f) for f in fs])
    ms = [c for c in L.columns if c not in META and not c.startswith("gate:")]
    rows = []
    for c, g in L.groupby("collection"):
        for m in ms:
            if g[m].isna().any():
                continue
            a = metrics.bootstrap_ci(roc_auc_score, g.y.to_numpy(), g[m].to_numpy(), n_boot=1000)
            r = {"held_out": c, "n": len(g), "events": int(g.y.sum()), "model": m, "auc": a[0], "lo": a[1], "hi": a[2]}
            if m != "Clinical":
                dd = stats.delong(g.y, g["Clinical"], g[m])
                r.update(diff=dd["diff"], se=dd["se"], ci_lo=dd["ci_lo"], ci_hi=dd["ci_hi"], p=dd["p"])
            rows.append(r)
    t = pd.DataFrame(rows)
    pooled = []
    for m in [x for x in ms if x != "Clinical"]:
        s = t[(t.model == m) & t.se.notna()]
        if len(s) < 2:
            continue
        w = 1 / s.se ** 2
        est, se = float((w * s["diff"]).sum() / w.sum()), float(np.sqrt(1 / w.sum()))
        q = float((w * (s["diff"] - est) ** 2).sum())
        pooled.append({"model": m, "collections": len(s), "pooled_diff": est, "ci_lo": est - 1.96 * se, "ci_hi": est + 1.96 * se,
                       "p": float(2 * sps.norm.sf(abs(est / se))), "i2": max(0.0, (q - (len(s) - 1)) / q) if q > 0 else 0.0})
    return t, pd.DataFrame(pooled)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    T = pd.read_csv(os.path.join(a.dir, "table.csv"))
    P = load_devext(a.dir)
    t_perf, t_cmp, t_per = devext_tables(P, a.n_boot)
    tabs = {"r2_performance": t_perf, "r2_vs_clinical": t_cmp, "r2_per_collection": t_per,
            "r2_added_value": added_value(P, T), "r2_subtype_auc": subtype_auc(P, T), "r2_recalibration": recalibration(P),
            "r2_controls": controls_table(a.dir)}
    tabs["r2_loco"], tabs["r2_loco_pooled"] = loco_tables(a.dir)
    g = [c for c in P.columns if c.startswith("gate:")]
    if g:
        tabs["r2_gate"] = P.groupby(["set", "collection"])[g].agg(["mean", "std"]).round(4).reset_index()
    for k, t in tabs.items():
        if len(t):
            t.to_csv(os.path.join(a.out, k + ".csv"), index=False)
            print(f"\n== {k}\n{t.round(4).to_string(index=False)}")
    P.drop(columns=[]).to_csv(os.path.join(a.out, "r2_predictions.csv"), index=False)


if __name__ == "__main__":
    main()
