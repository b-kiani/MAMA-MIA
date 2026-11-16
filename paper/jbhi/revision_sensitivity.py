"""Sensitivity analyses for outcome, treatment and case-mix differences (reviewer point 8).

    python paper/jbhi/revision_sensitivity.py   ->  results/revision/rev_sens_*.csv

Tabular models only (no GPU): they are refitted exactly as in the main analysis.
1. pCR rate by receptor subtype and collection (case-mix and treatment-era shift)
2. treatment-harmonised development: clinical, clinical + FTV and clinical + kinetic radiomics
   fitted on the I-SPY2 control arm only (paclitaxel, or paclitaxel + trastuzumab if HER2+)
3. complete-case external evaluation (receptor status recorded)
4. external evaluation standardised to the I-SPY2 subtype mix (inverse-probability weights)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
from mmonc import metrics, stats  # noqa: E402
from mmonc.models import make_logreg  # noqa: E402
from mmonc.prep2 import KIN_NAMES  # noqa: E402

RAW = os.path.join(ROOT, "results", "round2", "raw")
OUT = os.path.join(ROOT, "results", "revision")
NAMES = {"DUKE": "Duke", "ISPY1": "I-SPY1", "NACT": "NACT", "ISPY2": "I-SPY2"}
EXTC = ["DUKE", "ISPY1", "NACT"]
CONTROL = ("Paclitaxel", "Paclitaxel + Trastuzumab")


def subtype(r):
    if pd.isna(r.hr) or pd.isna(r.her2):
        return "Unknown"
    return "HER2+" if r.her2 == 1 else ("HR+/HER2-" if r.hr == 1 else "Triple negative")


def boot_ci(fn, n, n_boot=2000, seed=0, strata=None):
    rng = np.random.default_rng(seed)
    out = []
    groups = [np.arange(n)] if strata is None else [np.flatnonzero(strata == s) for s in np.unique(strata)]
    for _ in range(n_boot):
        b = np.concatenate([rng.choice(g, len(g)) for g in groups])
        try:
            out.append(fn(b))
        except ValueError:
            pass
    return np.percentile(out, [2.5, 97.5])


def main():
    T = pd.read_csv(os.path.join(RAW, "table.csv"))
    K = pd.read_csv(os.path.join(RAW, "kinetics.csv"), index_col=0).loc[T.case_id, KIN_NAMES].to_numpy(float)
    keep = T.pcr.notna().to_numpy()
    T, K = T[keep].reset_index(drop=True), K[keep]
    y = T.pcr.to_numpy().astype(int)
    Xc = T[["age", "hr", "her2"]].to_numpy(float)
    ftv = K[:, [KIN_NAMES.index("log_ftv_mm3")]]
    dev = (T.collection == "ISPY2").to_numpy()
    ext = ~dev
    ctl = dev & T.nac_agent.isin(CONTROL).to_numpy()
    T["subtype"] = T.apply(subtype, axis=1)

    # 1. pCR by subtype and collection
    rows = []
    for c in ["ISPY2", "DUKE", "ISPY1", "NACT"]:
        for grp in [("I-SPY2 control arm", ctl)] if c == "ISPY2" else []:
            pass
        for label, mask in ([(NAMES[c], (T.collection == c).to_numpy())] + ([("I-SPY2 control arm", ctl)] if c == "ISPY2" else [])):
            for s in ["HR+/HER2-", "HER2+", "Triple negative", "Unknown", "All"]:
                m = mask & ((T.subtype == s).to_numpy() if s != "All" else True)
                if m.sum():
                    rows.append({"cohort": label, "subtype": s, "n": int(m.sum()), "pcr": int(y[m].sum()), "pcr_rate": y[m].mean()})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "rev_sens_pcr_by_subtype.csv"), index=False)

    # 2. development on all of I-SPY2 vs on its control arm only
    specs = [("Clinical", Xc, "l2", 1.0), ("Clinical + FTV", np.hstack([Xc, ftv]), "l2", 1.0),
             ("Clinical + kinetic radiomics", np.hstack([Xc, K]), "l1", 0.3)]
    rows, preds = [], {}
    for dname, dmask in [("All I-SPY2 arms", dev), ("I-SPY2 control arm only", ctl)]:
        for label, X, pen, C in specs:
            p = make_logreg(C=C, penalty=pen).fit(X[dmask], y[dmask]).predict_proba(X[ext])[:, 1]
            preds[(dname, label)] = p
        ye, ge = y[ext], T.collection.to_numpy()[ext]
        pc = preds[(dname, "Clinical")]
        for label, *_ in specs:
            p = preds[(dname, label)]
            for cname, m in [("All external", np.ones(ext.sum(), bool))] + [(NAMES[c], ge == c) for c in EXTC]:
                cal = metrics.calibration(ye[m], p[m])
                lo, hi = boot_ci(lambda b: roc_auc_score(ye[m][b], p[m][b]), int(m.sum()), strata=ye[m])
                r = {"development": dname, "n_dev": int(dmask.sum()), "events_dev": int(y[dmask].sum()), "model": label, "collection": cname,
                     "n": int(m.sum()), "auc": roc_auc_score(ye[m], p[m]), "auc_lo": lo, "auc_hi": hi,
                     "observed": ye[m].mean(), "mean_predicted": p[m].mean(), "e_over_o": p[m].mean() / ye[m].mean(),
                     "intercept": cal["intercept"], "slope": cal["slope"], "brier": cal["brier"]}
                if label != "Clinical":
                    d = stats.delong(ye[m], pc[m], p[m])
                    r.update({"diff": d["diff"], "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"], "p": d["p"]})
                rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(OUT, "rev_sens_control_arm.csv"), index=False)

    # 3 and 4 use the stored per-patient predictions of every model
    P = pd.read_csv(os.path.join(ROOT, "results", "round2", "r2_predictions.csv"))
    extra = os.path.join(OUT, "rev_all_predictions.csv")
    if os.path.exists(extra):
        P = pd.read_csv(extra)
    E = P[P.set == "ext"].merge(T[["case_id", "subtype"]], on="case_id", how="left").reset_index(drop=True)
    ye = E.y.to_numpy()
    models = [c for c in P.columns if c not in ("case_id", "collection", "y", "set", "Clinical") and not c.startswith("gate:") and E[c].notna().all()]
    known = (E.subtype != "Unknown").to_numpy()
    # weights: make the external subtype mix equal to I-SPY2's (unknown-subtype patients keep weight 1)
    mix_dev = T[dev].subtype.value_counts(normalize=True)
    mix_ext = E[known].subtype.value_counts(normalize=True)
    w = np.where(known, E.subtype.map(lambda s: mix_dev.get(s, 0) / mix_ext.get(s, 1)).to_numpy(), 1.0).astype(float)
    rows = []
    for m in models:
        d = stats.delong(ye[known], E.Clinical.to_numpy()[known], E[m].to_numpy()[known])
        rows.append({"analysis": "Complete cases (receptor status recorded)", "model": m, "n": int(known.sum()), "events": int(ye[known].sum()),
                     "auc_clinical": d["auc1"], "auc_model": d["auc2"], "diff": d["diff"], "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"]})
        pc, pm = E.Clinical.to_numpy(), E[m].to_numpy()
        f = lambda b: roc_auc_score(ye[b], pm[b], sample_weight=w[b]) - roc_auc_score(ye[b], pc[b], sample_weight=w[b])  # noqa: E731
        lo, hi = boot_ci(f, len(ye), strata=ye)
        rows.append({"analysis": "Standardised to the I-SPY2 subtype mix", "model": m, "n": len(ye), "events": int(ye.sum()),
                     "auc_clinical": roc_auc_score(ye, pc, sample_weight=w), "auc_model": roc_auc_score(ye, pm, sample_weight=w),
                     "diff": f(np.arange(len(ye))), "ci_lo": lo, "ci_hi": hi})
    C = pd.DataFrame(rows)
    C.to_csv(os.path.join(OUT, "rev_sens_complete_case.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)
    print(pd.read_csv(os.path.join(OUT, "rev_sens_pcr_by_subtype.csv")).round(3).to_string(index=False))
    print(S.round(3).to_string(index=False))
    print(C.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
