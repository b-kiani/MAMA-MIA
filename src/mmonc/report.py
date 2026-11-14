"""Merge finished runs into the paper's tables and figures, and add the secondary analyses.

    python -m mmonc.report --prepared /path/prepared --runs run1 run3a run3b run3c --out results/final

Reads only saved predictions (no retraining) plus the prepared patient table. Thresholds are
re-derived from development (ISPY2 out-of-fold) predictions only.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import figures, metrics, stats
from .models import make_logreg

PRIMARY = "CARF (proposed)"
CLIN = "Clinical"


# --------------------------------------------------------------------------- loading
def load_runs(runs):
    oof, ext = None, None
    for r in runs:
        o = pd.read_csv(os.path.join(r, "predictions_internal_oof.csv"))
        e_path = os.path.join(r, "predictions_external.csv")
        e = pd.read_csv(e_path) if os.path.exists(e_path) else None
        if oof is None:
            oof, ext = o, e
            continue
        new = [c for c in o.columns if c not in oof.columns]
        oof = oof.merge(o[["case_id"] + new], on="case_id", how="left")
        if e is not None and ext is not None:
            new = [c for c in e.columns if c not in ext.columns]
            ext = ext.merge(e[["case_id"] + new], on="case_id", how="left")
    return oof, ext


def model_cols(df):
    return [c for c in df.columns if c not in ("case_id", "y", "collection") and not c.startswith("gate:")]


# --------------------------------------------------------------------------- helpers
def auc_se(y, p):
    """Hanley-McNeil standard error of the AUC."""
    a = roc_auc_score(y, p)
    n1, n0 = int(np.sum(y)), int(len(y) - np.sum(y))
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    return a, float(np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0)))


def random_effects(estimates, ses):
    """DerSimonian-Laird pooling on the logit-AUC scale. Returns pooled AUC, CI, I2, prediction interval."""
    a = np.clip(np.asarray(estimates), 1e-3, 1 - 1e-3)
    th = np.log(a / (1 - a))
    v = (np.asarray(ses) / (a * (1 - a))) ** 2
    w = 1 / v
    fixed = np.sum(w * th) / np.sum(w)
    q = np.sum(w * (th - fixed) ** 2)
    k = len(th)
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (q - (k - 1)) / c) if c > 0 else 0.0
    ws = 1 / (v + tau2)
    mu = np.sum(ws * th) / np.sum(ws)
    se = np.sqrt(1 / np.sum(ws))
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0
    tcrit = sps.t.ppf(0.975, max(1, k - 2))
    pi = (mu - tcrit * np.sqrt(tau2 + se ** 2), mu + tcrit * np.sqrt(tau2 + se ** 2))
    ex = lambda z: float(1 / (1 + np.exp(-z)))  # noqa: E731
    return {"k": k, "pooled_auc": ex(mu), "lo": ex(mu - 1.96 * se), "hi": ex(mu + 1.96 * se),
            "i2": float(i2), "tau2": float(tau2), "pi_lo": ex(pi[0]), "pi_hi": ex(pi[1])}


def fmt(a, lo, hi, d=2):
    return f"{a:.{d}f} ({lo:.{d}f}-{hi:.{d}f})"


# --------------------------------------------------------------------------- tables
def table1(tb):
    rows = []
    for c, g in list(tb.groupby("collection")) + [("All", tb)]:
        r = {"Collection": c, "Patients": len(g), "pCR, n (%)": f"{int(g.pcr.sum())} ({100*g.pcr.mean():.1f})",
             "Age, mean (SD)": f"{g.age.mean():.1f} ({g.age.std():.1f})",
             "HR positive, %": f"{100*g.hr.mean():.1f}", "HER2 positive, %": f"{100*g.her2.mean():.1f}",
             "HR/HER2 missing, %": f"{100*g[['hr','her2']].isna().any(axis=1).mean():.1f}",
             "3 T, %": f"{100*(g.field_strength >= 3).mean():.1f}",
             "Vendor GE/Siemens/Philips, %": "/".join(f"{100*(g.vendor.astype(str).str.upper()==v).mean():.0f}" for v in ("GE", "SIEMENS", "PHILIPS")),
             "Plane": "/".join(sorted(g.view.dropna().astype(str).str.lower().unique()))[:20],
             "Sites": int(g.site.nunique()),
             "Black/African American, %": f"{100*g.ethnicity.astype(str).str.contains('afric', case=False).mean():.1f}"}
        rows.append(r)
    return pd.DataFrame(rows)


def perf_table(df, thr, cohort_col=None, n_boot=2000):
    rows = []
    groups = [("Pooled external" if cohort_col else "ISPY2 (out-of-fold)", np.ones(len(df), bool))]
    if cohort_col:
        groups += [(c, (df[cohort_col] == c).to_numpy()) for c in sorted(df[cohort_col].unique())]
    for m in model_cols(df):
        for name, s in groups:
            y, p = df.y.to_numpy()[s], df[m].to_numpy()[s]
            if min(y.sum(), (1 - y).sum()) < 5:
                continue
            r = metrics.summary(y, p, thr[m], n_boot=n_boot)
            rows.append({"model": m, "cohort": name, **r})
    return pd.DataFrame(rows)


def comparisons(ext, ref=CLIN):
    rows = []
    y = ext.y.to_numpy()
    thr_ref = 0.5
    for m in model_cols(ext):
        if m == ref:
            continue
        d = stats.delong(y, ext[ref], ext[m])
        rows.append({"model": m, "vs": ref, "auc_ref": d["auc1"], "auc": d["auc2"], "diff": d["diff"],
                     "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"], "p": d["p"],
                     **stats.nri_idi(y, ext[ref].to_numpy(), ext[m].to_numpy(), thr_ref)})
    t = pd.DataFrame(rows)
    t["p_holm"] = stats.holm(t.p)
    return t


def subgroups(df, tb, models, n_boot=500):
    d = df.merge(tb[["case_id", "age", "ethnicity", "vendor", "field_strength", "tumor_subtype", "view"]], on="case_id", how="left")
    d["age_band"] = pd.cut(d.age, [0, 40, 50, 60, 120], right=False, labels=["<40", "40-49", "50-59", ">=60"])
    d["ethnicity"] = d.ethnicity.astype(str).str.lower().str.slice(0, 16)
    d["vendor"] = d.vendor.astype(str).str.upper()
    rows = []
    for var in ["age_band", "ethnicity", "vendor", "field_strength", "tumor_subtype", "view"]:
        for lvl, g in d.groupby(var, observed=True):
            y = g.y.to_numpy()
            if min(y.sum(), (1 - y).sum()) < 5:
                continue
            for m in models:
                a = metrics.bootstrap_ci(roc_auc_score, y, g[m].to_numpy(), n_boot=n_boot)
                rows.append({"variable": var, "level": str(lvl), "model": m, "n": len(g), "events": int(y.sum()),
                             "auc": a[0], "lo": a[1], "hi": a[2]})
    return pd.DataFrame(rows)


def site_heterogeneity(oof, tb, models):
    d = oof.merge(tb[["case_id", "site"]], on="case_id", how="left")
    rows, pooled = [], []
    for m in models:
        est, ses, sites = [], [], []
        for s, g in d.groupby("site"):
            y = g.y.to_numpy()
            if min(y.sum(), (1 - y).sum()) < 5:
                continue
            a, se = auc_se(y, g[m].to_numpy())
            est.append(a); ses.append(se); sites.append(s)
            rows.append({"model": m, "site": s, "n": len(g), "events": int(y.sum()), "auc": a, "se": se})
        if len(est) >= 3:
            pooled.append({"model": m, **random_effects(est, ses)})
    return pd.DataFrame(rows), pd.DataFrame(pooled)


def temporal(oof, tb, models):
    d = oof.merge(tb[["case_id", "acquisition_date"]], on="case_id", how="left")
    yr = pd.to_datetime(d.acquisition_date, errors="coerce").dt.year
    d["period"] = np.where(yr.isna(), "unknown", np.where(yr <= yr.median(), f"<= {int(yr.median()) if yr.notna().any() else 0}", f"> {int(yr.median()) if yr.notna().any() else 0}"))
    rows = []
    for per, g in d.groupby("period"):
        y = g.y.to_numpy()
        if min(y.sum(), (1 - y).sum()) < 5:
            continue
        for m in models:
            a = metrics.bootstrap_ci(roc_auc_score, y, g[m].to_numpy(), n_boot=500)
            rows.append({"period": per, "model": m, "n": len(g), "events": int(y.sum()), "auc": a[0], "lo": a[1], "hi": a[2]})
    return pd.DataFrame(rows)


def clinical_variants(tb, rad, dev_ids, ext_ids):
    """Tabular-only sensitivity models, fitted on ISPY2, scored on external. CPU only."""
    t = tb.set_index("case_id")
    sub = pd.get_dummies(t.tumor_subtype.astype(str), prefix="sub", dtype=float)
    sets = {
        "Clinical (age, HR, HER2)": t[["age", "hr", "her2"]],
        "Clinical extended (+ ER, PR, subtype)": pd.concat([t[["age", "hr", "her2", "er", "pr"]], sub], axis=1),
        "Clinical + tumour size": pd.concat([t[["age", "hr", "her2"]], rad[["log_tumour_mm3", "extent_mm"]]], axis=1),
        "Clinical + radiomics (L1)": pd.concat([t[["age", "hr", "her2"]], rad], axis=1),
    }
    rows, preds = [], {}
    yd, ye = t.loc[dev_ids, "pcr"].to_numpy(int), t.loc[ext_ids, "pcr"].to_numpy(int)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    for name, X in sets.items():
        pen = "l1" if "radiomics" in name else "l2"
        m = make_logreg(C=0.3 if pen == "l1" else 1.0, penalty=pen)
        po = cross_val_predict(m, X.loc[dev_ids].to_numpy(float), yd, cv=cv, method="predict_proba")[:, 1]
        pe = m.fit(X.loc[dev_ids].to_numpy(float), yd).predict_proba(X.loc[ext_ids].to_numpy(float))[:, 1]
        preds[name] = pe
        a_i = metrics.bootstrap_ci(roc_auc_score, yd, po, n_boot=1000)
        a_e = metrics.bootstrap_ci(roc_auc_score, ye, pe, n_boot=1000)
        rows.append({"model": name, "internal_auc": fmt(*a_i), "external_auc": fmt(*a_e), "ext_auc_value": a_e[0]})
    return pd.DataFrame(rows), preds


def leave_one_collection_out(tb, rad):
    """Tabular models only: train on three collections, test on the fourth (sensitivity analysis)."""
    t = tb.set_index("case_id")
    X = {"Clinical": t[["age", "hr", "her2"]], "Clinical + radiomics": pd.concat([t[["age", "hr", "her2"]], rad], axis=1)}
    rows = []
    for held in sorted(t.collection.unique()):
        tr, te = t.index[t.collection != held], t.index[t.collection == held]
        ytr, yte = t.loc[tr, "pcr"].to_numpy(int), t.loc[te, "pcr"].to_numpy(int)
        if min(yte.sum(), (1 - yte).sum()) < 5:
            continue
        for name, Xs in X.items():
            m = make_logreg(C=0.3 if "radiomics" in name else 1.0, penalty="l1" if "radiomics" in name else "l2")
            p = m.fit(Xs.loc[tr].to_numpy(float), ytr).predict_proba(Xs.loc[te].to_numpy(float))[:, 1]
            a = metrics.bootstrap_ci(roc_auc_score, yte, p, n_boot=1000)
            rows.append({"held_out": held, "model": name, "n": len(te), "events": int(yte.sum()), "auc": fmt(*a)})
    return pd.DataFrame(rows)


def survival(ext, tb, models):
    from lifelines import CoxPHFitter

    d = ext.merge(tb[["case_id", "rfs_months", "rfs_event", "hr", "her2"]], on="case_id", how="left")
    d = d.dropna(subset=["rfs_months", "rfs_event"])
    d = d[d.rfs_months > 0]
    rows = []
    for m in models:
        risk = 1 - d[m].to_numpy()
        c = metrics.c_index(d.rfs_months.to_numpy(), d.rfs_event.to_numpy().astype(int), risk)
        cox = CoxPHFitter().fit(pd.DataFrame({"T": d.rfs_months, "E": d.rfs_event, "risk_sd": (risk - risk.mean()) / risk.std(),
                                              "hr": d.hr.fillna(d.hr.median()), "her2": d.her2.fillna(d.her2.median())}),
                                "T", "E")
        s = cox.summary.loc["risk_sd"]
        rows.append({"model": m, "n": len(d), "events": int(d.rfs_event.sum()), "c_index": c,
                     "hr_per_sd_adj": float(s["exp(coef)"]), "hr_lo": float(s["exp(coef) lower 95%"]),
                     "hr_hi": float(s["exp(coef) upper 95%"]), "p": float(s["p"])})
    return pd.DataFrame(rows), d


def gate_analysis(ext, oof, tb):
    rows = []
    if "gate:CARF" not in ext.columns:
        return pd.DataFrame(rows)
    allg = pd.concat([oof.assign(collection="ISPY2")[["case_id", "collection", "y", "gate:CARF", CLIN, PRIMARY]],
                      ext[["case_id", "collection", "y", "gate:CARF", CLIN, PRIMARY]]])
    allg = allg.merge(tb[["case_id", "tumor_subtype"]], on="case_id", how="left")
    for c, g in allg.groupby("collection"):
        lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))  # noqa: E731
        rows.append({"group": f"collection={c}", "n": len(g), "gate_mean": g["gate:CARF"].mean(),
                     "gate_sd": g["gate:CARF"].std(),
                     "mean_abs_imaging_shift_logit": float(np.abs(lg(g[PRIMARY]) - lg(g[CLIN])).mean())})
    for c, g in allg.groupby("tumor_subtype"):
        rows.append({"group": f"subtype={c}", "n": len(g), "gate_mean": g["gate:CARF"].mean(), "gate_sd": g["gate:CARF"].std()})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig = os.path.join(a.out, "figures")
    tb = pd.read_csv(os.path.join(a.prepared, "table.csv"))
    rad = pd.read_csv(os.path.join(a.prepared, "radiomics.csv"), index_col=0)
    oof, ext = load_runs(a.runs)
    models = model_cols(oof)
    thr = {m: metrics.youden_threshold(oof.y, oof[m]) for m in models}
    json.dump({"thresholds": thr, "runs": a.runs}, open(os.path.join(a.out, "thresholds.json"), "w"), indent=1)
    out = {}
    out["table1_cohort"] = table1(tb[tb.pcr.notna()])
    out["table2_internal"] = perf_table(oof, thr, n_boot=a.n_boot)
    main_models = [m for m in [CLIN, "Radiomics", "Clinical + radiomics", "Image only (ResNet)", "Image only (DenseNet-121)",
                               "Image only (EfficientNet-B0)", "Image only (ViT)", "Concatenation fusion", PRIMARY] if m in models]
    out["tableS_site_auc"], out["tableS_site_pooled"] = site_heterogeneity(oof, tb, [m for m in (CLIN, PRIMARY) if m in models])
    out["tableS_temporal"] = temporal(oof, tb, [m for m in (CLIN, PRIMARY) if m in models])
    out["tableS_loco_tabular"] = leave_one_collection_out(tb[tb.pcr.notna()], rad)
    if ext is not None:
        out["table3_external"] = perf_table(ext, thr, cohort_col="collection", n_boot=a.n_boot)
        out["table5_vs_clinical"] = comparisons(ext)
        if PRIMARY in models:
            best_img = max([m for m in models if m.startswith("Image only")], key=lambda m: roc_auc_score(ext.y, ext[m]), default=None)
            prim = stats.delong(ext.y, ext[CLIN], ext[PRIMARY])
            out["primary"] = pd.DataFrame([{**prim, "verdict": stats.verdict(prim, 0.03)}])
            if best_img:
                d2 = stats.delong(ext.y, ext[best_img], ext[PRIMARY])
                out["tableS_carf_vs_best_image"] = pd.DataFrame([{"best_image_model": best_img, **d2}])
        out["tableS_subgroups"] = subgroups(ext, tb, [m for m in (CLIN, PRIMARY) if m in models])
        cv_tab, _ = clinical_variants(tb[tb.pcr.notna()], rad, list(oof.case_id), list(ext.case_id))
        out["tableS_clinical_variants"] = cv_tab
        out["tableS_gate"] = gate_analysis(ext, oof, tb)
        try:
            out["tableS_survival"], sd = survival(ext, tb, [m for m in (CLIN, PRIMARY) if m in models])
            if PRIMARY in models and len(sd) >= 30:
                figures.kaplan_meier(sd.rfs_months.to_numpy(), sd.rfs_event.to_numpy().astype(int),
                                     1 - sd[PRIMARY].to_numpy(), os.path.join(fig, "fig6_km_external"))
        except Exception as e:  # noqa: BLE001
            print("survival analysis skipped:", e)
        shown = {m: ext[m].to_numpy() for m in main_models if m in (CLIN, "Clinical + radiomics", "Image only (ResNet)", "Concatenation fusion", PRIMARY)}
        figures.roc_pr(ext.y, shown, os.path.join(fig, "fig2_roc_pr_external"), title="(pooled external)")
        figures.calibration_plot(ext.y, shown, os.path.join(fig, "fig3_calibration_external"))
        figures.decision_curve(ext.y, shown, os.path.join(fig, "fig4_decision_curve_external"))
        pe = out["table3_external"]
        figures.confusion({m: pe[(pe.model == m) & (pe.cohort == "Pooled external")].iloc[0] for m in shown},
                          os.path.join(fig, "fig5_confusion_external"))
        figures.forest(out["tableS_subgroups"], os.path.join(fig, "figS_subgroups_forest"))
    figures.roc_pr(oof.y, {m: oof[m].to_numpy() for m in main_models}, os.path.join(fig, "figS_roc_internal"), title="(ISPY2 OOF)")
    figures.workflow(os.path.join(fig, "fig1_workflow"))
    for k, v in out.items():
        v.to_csv(os.path.join(a.out, k + ".csv"), index=False)
    print(json.dumps({k: len(v) for k, v in out.items()}))


if __name__ == "__main__":
    main()
