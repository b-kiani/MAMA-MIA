"""Third-review analyses (paper/analysis_plan_revision3.md) from results/revision3/raw.

    python paper/jbhi/revision3_analyses.py   ->  results/revision3/*.csv, paper/jbhi/rev3_numbers.json

Env: REV3_RAW (raw folder; default results/revision3/raw), REV3_TABLE (case table; default the round-2 table).
"""
from __future__ import annotations

import glob
import json
import os
import sys
import tarfile

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
from mmonc import metrics, stats  # noqa: E402
from mmonc.models import make_logreg  # noqa: E402
from mmonc.pipeline5 import BETAS, CONTROL_ARM, REGIMEN_GROUPS  # noqa: E402
from mmonc.prep2 import KIN_NAMES  # noqa: E402

RAW = os.environ.get("REV3_RAW", os.path.join(ROOT, "results", "revision3", "raw"))
OUT = os.path.join(ROOT, "results", "revision3")
TABLE = os.environ.get("REV3_TABLE", os.path.join(ROOT, "results", "round2", "raw", "table.csv"))
EXT = ("DUKE", "ISPY1", "NACT")
LAB = {"Clinical": "Clinical (age, HR, HER2)", "s25d": "Image only, pretrained 2.5D", "lf": "Late fusion (clinical + 2.5D image)",
       "carfm_25d": "CARF-mix, pretrained 2.5D", "carfm_k": "CARF-mix, kinetic 3D", "image_only_k": "Image only, kinetic 3D",
       "ckin": "Clinical + kinetic radiomics"}


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def dl(y, a, b):
    try:
        return stats.delong(y, a, b)
    except ValueError:
        return {"diff": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan, "se": np.nan}


def auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan


# ------------------------------------------------------------------------- loading
def load():
    tgz = os.path.join(RAW, "preds.tar.gz")
    if os.path.exists(tgz) and not os.path.isdir(os.path.join(RAW, "preds")):
        with tarfile.open(tgz) as t:
            t.extractall(RAW, filter="data")
    S = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(RAW, "runs_*.csv")))], ignore_index=True)
    S = S.drop_duplicates("run_id", keep="last").set_index("run_id")
    return S


_CACHE = {}


def P(S, rid):
    if rid not in S.index:
        return None
    if rid not in _CACHE:
        _CACHE[rid] = pd.read_csv(os.path.join(RAW, "preds", S.loc[rid, "file"]))
    return _CACHE[rid]


def rid(model, variant="lps", seed=0, labels="real", tab="clinkin", subset="all", folds="cv"):
    return "|".join([model, variant, f"s{seed}", labels, tab, subset, folds])


def late_fusion(D):
    """Cross-fitted stacking of the clinical and image-only logits (analysis plan, section 2)."""
    X = np.c_[logit(D.zc), logit(D.p)]
    y, dev = D.y.to_numpy(), (D.set == "dev").to_numpy()
    out = np.full(len(D), np.nan)
    f = D.fold.to_numpy()
    for k in np.unique(f[dev]):
        tr, te = dev & (f != k), dev & (f == k)
        out[te] = LogisticRegression(C=1e4, max_iter=2000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    out[~dev] = LogisticRegression(C=1e4, max_iter=2000).fit(X[dev], y[dev]).predict_proba(X[~dev])[:, 1]
    return out


def ckin(D, K):
    """Clinical + kinetic radiomics (L1 logistic, C = 0.3, as round 2) on the folds of run D. K: kinetic table by case."""
    tb = TB.set_index("case_id").loc[D.case_id]
    X = np.hstack([tb[["age", "hr", "her2"]].to_numpy(float), K.loc[D.case_id, KIN_NAMES].to_numpy(float)])
    y, dev, f = D.y.to_numpy(), (D.set == "dev").to_numpy(), D.fold.to_numpy()
    out = np.full(len(D), np.nan)
    for k in np.unique(f[dev]):
        tr, te = dev & (f != k), dev & (f == k)
        out[te] = make_logreg(C=0.3, penalty="l1").fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    out[~dev] = make_logreg(C=0.3, penalty="l1").fit(X[dev], y[dev]).predict_proba(X[~dev])[:, 1]
    return out


def compare(D, col, ref="zc", per_coll=True):
    """External and I-SPY2 out-of-fold AUC and paired difference from `ref` (both columns of D)."""
    e, d = D[D.set == "ext"], D[D.set == "dev"]
    r = {"ext_auc": auc(e.y, e[col]), "oof_auc": auc(d.y, d[col]), "ext_ref": auc(e.y, e[ref]), "oof_ref": auc(d.y, d[ref])}
    x = dl(e.y, e[ref], e[col]); r.update(ext_diff=x["diff"], ext_lo=x["ci_lo"], ext_hi=x["ci_hi"], ext_p=x["p"], ext_se=x["se"])
    x = dl(d.y, d[ref], d[col]); r.update(oof_diff=x["diff"], oof_lo=x["ci_lo"], oof_hi=x["ci_hi"])
    if per_coll:
        for c in EXT:
            m = (e.collection == c).to_numpy()
            if m.sum() > 5:
                x = dl(e.y[m], e[ref][m], e[col][m])
                r.update({f"diff_{c}": x["diff"], f"lo_{c}": x["ci_lo"], f"hi_{c}": x["ci_hi"]})
    return r


def calib(D, col):
    out = {}
    for c in EXT:
        e = D[(D.set == "ext") & (D.collection == c)]
        if len(e) > 5 and e.y.nunique() > 1:
            k = metrics.calibration(e.y.to_numpy(), e[col].to_numpy())
            out.update({f"int_{c}": k["intercept"], f"slope_{c}": k["slope"]})
    return out


def with_lf(D):
    D = D.copy()
    D["lf"] = late_fusion(D)
    return D


# ------------------------------------------------------------------------- 1 audit
def audit():
    f = os.path.join(RAW, "audit.csv")
    if not os.path.exists(f):
        return None, {}
    A = pd.read_csv(f)
    g = A.groupby("collection")
    vc = lambda s: "; ".join(f"{k} {v}" for k, v in s.value_counts().items())  # noqa: E731
    T = pd.DataFrame({
        "n": g.size(), "view": g.view.agg(vc), "stored_code_pre": g.code_pre.agg(vc), "stored_code_mask": g.code_mask.agg(vc),
        "phase_mask_geometry_match": g.geometry_match.mean(), "code_after": g.code_after.agg(vc),
        "crop_inside_image_lt1": g.orig_coverage.apply(lambda s: float((s < 0.999).mean())),
        "crop_coverage_min": g.orig_coverage.min(), "tumour_truncated": g.orig_tumour_truncated.mean(),
        "fov96_tumour_truncated": g.fov_tumour_truncated.mean(), "tumour_extent_mm_median": g.lps_extent_mm.median(),
        "ctx_coverage_median": g.ctx_coverage.median(), "ctx_coverage_p10": g.ctx_coverage.quantile(0.1),
        "kinetic_orig_vs_lps_maxabs_median": g.kin_orig_vs_lps_maxabs.median(), "kinetic_orig_vs_lps_maxabs_max": g.kin_orig_vs_lps_maxabs.max()})
    T.to_csv(os.path.join(OUT, "rev3_audit_summary.csv"))
    N = {"n": int(len(A)), "geometry_match": float(A.geometry_match.mean()), "codes": A.groupby(["collection", "code_pre"]).size().to_dict(),
         "fov_trunc": float(A.fov_tumour_truncated.mean()), "ctx_cov_median": float(A.ctx_coverage.median()),
         "crop_clipped": float((A.orig_coverage < 0.999).mean())}
    N["codes"] = {f"{k[0]}:{k[1]}": int(v) for k, v in N["codes"].items()}
    return T, N


# ------------------------------------------------------------------------- 2 orientation
def orientation(S):
    rows = []
    for seed in range(5):
        a, b = P(S, rid("s25d", "orig", seed)), P(S, rid("s25d", "lps", seed))
        c, d = P(S, rid("carfm_25d", "orig", seed)), P(S, rid("carfm_25d", "lps", seed))
        for name, o, l, col in (("s25d", a, b, "p"), ("lf", a, b, "lf"), ("carfm_25d", c, d, "p")):
            if o is None or l is None:
                continue
            if col == "lf":
                o, l = with_lf(o), with_lf(l)
            M = o[["case_id", "collection", "y", "set"]].assign(orig=o[col].to_numpy(), lps=l.set_index("case_id").loc[o.case_id, col].to_numpy())
            r = {"model": name, "seed": seed, **compare(M, "lps", "orig")}
            r.update({f"orig_{k}": v for k, v in calib(M, "orig").items()})
            r.update({f"lps_{k}": v for k, v in calib(M, "lps").items()})
            rows.append(r)
    # 3D models: lps (this run) vs orig (round-3 runs with the same code and seed)
    r2 = os.path.join(ROOT, "results", "revision2", "raw")
    for name, f, col in (("carfm_k", "preds_nested_h1a.csv", "nested:CARF-mix, kinetic"), ("image_only_k", "preds_opt_h0a.csv", "opt:base")):
        l = P(S, rid(name))
        if l is None or not os.path.exists(os.path.join(r2, f)):
            continue
        o = pd.read_csv(os.path.join(r2, f)).set_index("case_id")
        if col not in o or not l.case_id.isin(o.index).all():
            continue
        M = l[["case_id", "collection", "y", "set"]].assign(lps=l.p.to_numpy(), orig=o.loc[l.case_id, col].to_numpy())
        r = {"model": name, "seed": 0, **compare(M, "lps", "orig")}
        r.update({f"orig_{k}": v for k, v in calib(M, "orig").items()}); r.update({f"lps_{k}": v for k, v in calib(M, "lps").items()})
        rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, "rev3_orientation.csv"), index=False)
    return T


# ------------------------------------------------------------------------- 3 repeated training
def repeats(S, variant="lps"):
    K = KV[KV.variant == "default"].set_index("case_id") if KV is not None else None
    rows, folds = [], []
    for seed in range(5):
        a, c = P(S, rid("s25d", variant, seed)), P(S, rid("carfm_25d", variant, seed))
        if a is not None:
            a = with_lf(a)
            rows.append({"model": "Clinical", "seed": seed, **compare(a, "zc")})
            for col, nm in (("p", "s25d"), ("lf", "lf")):
                rows.append({"model": nm, "seed": seed, **compare(a, col)})
            if K is not None and TB is not None:
                a["ck"] = ckin(a, K)
                rows.append({"model": "ckin", "seed": seed, **compare(a, "ck")})
        for nm in ("carfm_25d", "carfm_k"):
            d = P(S, rid(nm, variant, seed))
            if d is None:
                continue
            e = d[d.set == "ext"]
            r = {"model": nm, "seed": seed, **compare(d, "p"), "corr_abs_ext": float(np.nanmean(np.abs(e["corr"]))),
                 "corr_abs_oof": float(np.nanmean(np.abs(d[d.set == "dev"]["corr"]))), "gate_ext": float(np.nanmean(e.gate)),
                 "corr_abs_p90_ext": float(np.nanpercentile(np.abs(e["corr"]), 90))}
            rows.append(r)
        for nm in ("s25d", "carfm_25d", "carfm_k"):
            d = P(S, rid(nm, variant, seed))
            if d is None:
                continue
            e = d[d.set == "ext"]
            for k in [c for c in d.columns if c.startswith("pf")]:
                x = dl(e.y, e.zc, e[k])
                folds.append({"model": nm, "seed": seed, "fold_model": int(k[2:]), "ext_auc": auc(e.y, e[k]), "ext_diff": x["diff"]})
            folds.append({"model": nm, "seed": seed, "fold_model": -1, "ext_auc": auc(e.y, e.p), "ext_diff": dl(e.y, e.zc, e.p)["diff"]})
    j = os.path.join(os.path.dirname(RAW), "raw_3d", "carfm_k_repeats.json")
    if variant == "lps" and os.path.exists(j):  # follow-up run (seeds 1-4 of CARF-mix 3D); statistics computed from its exported predictions
        for x in json.load(open(j))["runs"]:
            rows.append({"model": "carfm_k", "seed": x["seed"], "ext_auc": x["ext_auc"], "oof_auc": x["oof_auc"], "ext_ref": x["ext_clin"], "oof_ref": x["oof_clin"],
                         "ext_diff": x["ext_diff"], "ext_lo": x["ext_lo"], "ext_hi": x["ext_hi"], "oof_diff": x["oof_diff"],
                         "diff_DUKE": x["d_DUKE"], "diff_ISPY1": x["d_ISPY1"], "diff_NACT": x["d_NACT"],
                         "corr_abs_ext": x["corr_abs_ext"], "corr_abs_oof": x["corr_abs_oof"], "gate_ext": x["gate_ext"]})
            folds += [{"model": "carfm_k", "seed": x["seed"], "fold_model": k, "ext_auc": a} for k, a in enumerate(x["fold_auc"])]
            folds.append({"model": "carfm_k", "seed": x["seed"], "fold_model": -1, "ext_auc": x["ext_auc"], "ext_diff": x["ext_diff"]})
    R, Fo = pd.DataFrame(rows), pd.DataFrame(folds)
    R.to_csv(os.path.join(OUT, f"rev3_repeats_{variant}.csv"), index=False)
    Fo.to_csv(os.path.join(OUT, f"rev3_fold_models_{variant}.csv"), index=False)
    if len(R):
        agg = R.groupby("model").agg(n_seeds=("seed", "size"), oof_mean=("oof_auc", "mean"), oof_sd=("oof_auc", "std"),
                                     ext_mean=("ext_auc", "mean"), ext_sd=("ext_auc", "std"), ext_min=("ext_auc", "min"), ext_max=("ext_auc", "max"),
                                     diff_mean=("ext_diff", "mean"), diff_sd=("ext_diff", "std"), diff_min=("ext_diff", "min"), diff_max=("ext_diff", "max"),
                                     lo_min=("ext_lo", "min"), hi_max=("ext_hi", "max"), n_ci_excl0=("ext_lo", lambda s: int((s > 0).sum())))
        if len(Fo):
            fm = Fo[Fo.fold_model >= 0].groupby("model").agg(fold_ext_mean=("ext_auc", "mean"), fold_ext_sd=("ext_auc", "std"),
                                                             fold_ext_min=("ext_auc", "min"), fold_ext_max=("ext_auc", "max"))
            agg = agg.join(fm)
        agg.to_csv(os.path.join(OUT, f"rev3_repeats_{variant}_summary.csv"))
    return R, Fo


# ------------------------------------------------------------------------- 4 signal recovery and null controls
def recovery(S):
    rows = []
    for d in (0, 1):
        for k in BETAS:
            lab = f"sig-{k}-{d}"
            a, c = P(S, rid("s25d", labels=lab, tab="clin")), P(S, rid("carfm_25d", labels=lab, tab="clin"))
            info = S.loc[rid("s25d", labels=lab, tab="clin")] if a is not None else (S.loc[rid("carfm_25d", labels=lab, tab="clin")] if c is not None else None)
            if info is None:
                continue
            base = {"level": k, "draw": d, "beta": BETAS[k], "oracle_ext": info["oracle_ext"], "oracle_dev": info["oracle_dev"]}
            if a is not None:
                a = with_lf(a)
                rows.append({**base, "model": "lf", **compare(a, "lf", per_coll=False)})
                rows.append({**base, "model": "s25d", **compare(a, "p", per_coll=False)})
            if c is not None:
                rows.append({**base, "model": "carfm_25d", **compare(c, "p", per_coll=False), "corr_abs_ext": float(np.nanmean(np.abs(c[c.set == "ext"]["corr"])))})
    for perm in (0, 1):
        for nm in ("s25d", "carfm_25d"):
            a = P(S, rid(nm, labels=f"null-{perm}", tab="clin"))
            if a is None:
                continue
            if nm == "s25d":
                a = with_lf(a)
                rows.append({"level": "permuted", "draw": perm, "model": "lf", **compare(a, "lf", per_coll=False)})
            rows.append({"level": "permuted", "draw": perm, "model": nm, **compare(a, "p", per_coll=False)})
    R = pd.DataFrame(rows)
    if len(R) and "oracle_ext" in R:
        R["recovery_ext"] = R.ext_diff / R.oracle_ext.where(R.oracle_ext.abs() > 1e-9)
        R["recovery_oof"] = R.oof_diff / R.oracle_dev.where(R.oracle_dev.abs() > 1e-9)
    R.to_csv(os.path.join(OUT, "rev3_recovery.csv"), index=False)
    return R


# ------------------------------------------------------------------------- 5 ablation
def ablation(S):
    rows = []
    combos = [(m, b, g) for m in (0, 1) for b in (0, 1) for g in (0, 1)]
    settings = [("real, seed 0", dict(seed=0)), ("real, seed 1", dict(seed=1)), ("planted 0.05", dict(labels="sig-0.05-0", tab="clin"))]
    for sname, kw in settings:
        for m, b, g in combos:
            name = "carfm_25d" if (m, b, g) == (1, 1, 1) else f"abl25d_m{m}b{b}g{g}"
            d = P(S, rid(name, **kw))
            if d is None:
                continue
            e = d[d.set == "ext"]
            r = {"setting": sname, "mix": m, "bound": b, "gate_reg": g, **compare(d, "p", per_coll=False),
                 "corr_abs_ext": float(np.nanmean(np.abs(e["corr"]))), "gate_ext": float(np.nanmean(e.gate)),
                 "slope_ext": metrics.calibration(e.y.to_numpy(), e.p.to_numpy())["slope"],
                 "int_ext": metrics.calibration(e.y.to_numpy(), e.p.to_numpy())["intercept"]}
            if "sig" in kw.get("labels", ""):
                r["oracle_ext"] = S.loc[rid(name, **kw), "oracle_ext"]
            rows.append(r)
        a = P(S, rid("s25d", **kw))
        if a is not None:
            a = with_lf(a)
            e = a[a.set == "ext"]
            r = {"setting": sname, "mix": "late fusion", **compare(a, "lf", per_coll=False),
                 "slope_ext": metrics.calibration(e.y.to_numpy(), e.lf.to_numpy())["slope"], "int_ext": metrics.calibration(e.y.to_numpy(), e.lf.to_numpy())["intercept"]}
            if "sig" in kw.get("labels", ""):
                r["oracle_ext"] = S.loc[rid("s25d", **kw), "oracle_ext"]
            rows.append(r)
    R = pd.DataFrame(rows)
    if "oracle_ext" in R:
        R["recovery_ext"] = R.ext_diff / R.oracle_ext
    R.to_csv(os.path.join(OUT, "rev3_ablation.csv"), index=False)
    return R


# ------------------------------------------------------------------------- 6 fixed field of view
def fov(S):
    rows = []
    for seed in range(3):
        for nm, col in (("s25d", "p"), ("s25d", "lf"), ("carfm_25d", "p")):
            o, l = P(S, rid(nm, "lps", seed)), P(S, rid(nm, "lps_fov", seed))
            if o is None or l is None:
                continue
            if col == "lf":
                o, l = with_lf(o), with_lf(l)
            M = o[["case_id", "collection", "y", "set", "zc"]].assign(tumour_scaled=o[col].to_numpy(), fixed=l.set_index("case_id").loc[o.case_id, col].to_numpy())
            rows.append({"model": "lf" if col == "lf" else nm, "seed": seed, **compare(M, "fixed", "tumour_scaled"),
                         "fixed_vs_clin": compare(M, "fixed", "zc", per_coll=False)["ext_diff"]})
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "rev3_fov.csv"), index=False)
    return R


# ------------------------------------------------------------------------- 7 treatment
def treatment(S):
    out = {}
    tb = TB.set_index("case_id")
    grp_of = {a: g for g, aa in REGIMEN_GROUPS.items() for a in aa}
    # (a) within regimen groups: seed-averaged OOF predictions (lps)
    acc = None
    for seed in range(5):
        a, c = P(S, rid("s25d", seed=seed)), P(S, rid("carfm_25d", seed=seed))
        if a is None or c is None:
            continue
        a = with_lf(a)
        d = a[a.set == "dev"][["case_id", "y", "zc", "lf"]].merge(c[c.set == "dev"][["case_id", "p"]], on="case_id")
        acc = d if acc is None else acc.assign(zc=acc.zc + d.set_index("case_id").loc[acc.case_id, "zc"].to_numpy(),
                                                 lf=acc.lf + d.set_index("case_id").loc[acc.case_id, "lf"].to_numpy(),
                                                 p=acc.p + d.set_index("case_id").loc[acc.case_id, "p"].to_numpy(), n=acc.get("n", 1) + 1)
    rows = []
    if acc is not None:
        acc["group"] = tb.loc[acc.case_id, "nac_agent"].map(grp_of).to_numpy()
        for g, d in [("all", acc)] + list(acc.groupby("group")):
            for col in ("lf", "p"):
                x = dl(d.y, d.zc, d[col])
                rows.append({"analysis": "within group (5-fold OOF, seed-averaged)", "group": g, "n": len(d), "events": int(d.y.sum()),
                             "model": "lf" if col == "lf" else "carfm_25d", "auc_clin": auc(d.y, d.zc), "auc_model": auc(d.y, d[col]),
                             "diff": x["diff"], "lo": x["ci_lo"], "hi": x["ci_hi"]})
    # (b) regimen-group-held-out folds
    for nm in ("s25d", "carfm_25d"):
        d = P(S, rid(nm, folds="regimen"))
        if d is None:
            continue
        if nm == "s25d":
            d = with_lf(d)
        col = "lf" if nm == "s25d" else "p"
        d = d[d.set == "dev"].copy()
        d["group"] = tb.loc[d.case_id, "nac_agent"].map(grp_of).to_numpy()
        for g, x0 in [("all", d)] + list(d.groupby("group")):
            x = dl(x0.y, x0.zc, x0[col])
            rows.append({"analysis": "regimen group held out", "group": g, "n": len(x0), "events": int(x0.y.sum()), "model": "lf" if col == "lf" else nm,
                         "auc_clin": auc(x0.y, x0.zc), "auc_model": auc(x0.y, x0[col]), "diff": x["diff"], "lo": x["ci_lo"], "hi": x["ci_hi"]})
    # (c) matched training size: control arm vs random all-arm subsets
    for sub in ["ctl"] + [f"rnd{r}" for r in range(5)]:
        d = P(S, rid("s25d", subset=sub))
        if d is None:
            continue
        d = with_lf(d)
        r = compare(d, "lf", per_coll=False)
        rows.append({"analysis": "matched training size (external)", "group": sub, "n": int((d.set == "dev").sum()), "model": "lf",
                     "auc_clin": r["ext_ref"], "auc_model": r["ext_auc"], "diff": r["ext_diff"], "lo": r["ext_lo"], "hi": r["ext_hi"],
                     "oof_clin": r["oof_ref"], "oof_model": r["oof_auc"]})
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "rev3_treatment.csv"), index=False)
    out["rows"] = R
    # (d) richer clinical comparator
    a = P(S, rid("s25d"))
    if a is not None:
        a = with_lf(a)
        t = tb.loc[a.case_id]
        cat = pd.get_dummies(t[["ethnicity", "bmi_group"]].astype("object"), dummy_na=False).astype(float)
        X = np.hstack([t[["age", "er", "pr", "her2", "menopausal", "multifocal"]].to_numpy(float), cat.to_numpy()])
        y, dev, f = a.y.to_numpy(), (a.set == "dev").to_numpy(), a.fold.to_numpy()
        rich = np.full(len(a), np.nan)
        for k in np.unique(f[dev]):
            tr, te = dev & (f != k), dev & (f == k)
            rich[te] = make_logreg(C=1.0).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        rich[~dev] = make_logreg(C=1.0).fit(X[dev], y[dev]).predict_proba(X[~dev])[:, 1]
        a["rich"] = rich
        b = a.assign(zc=rich)
        a["lf_rich"] = late_fusion(b)
        rr = [{"comparison": "richer clinical vs clinical", **compare(a, "rich", "zc")},
              {"comparison": "late fusion over richer clinical vs richer clinical", **compare(a, "lf_rich", "rich")}]
        miss = t[["menopausal", "multifocal", "bmi_group", "ethnicity", "pr"]].isna().groupby(t.collection.to_numpy()).mean()
        pd.DataFrame(rr).to_csv(os.path.join(OUT, "rev3_richer_clinical.csv"), index=False)
        miss.to_csv(os.path.join(OUT, "rev3_richer_clinical_missing.csv"))
        out["rich"] = rr
    return out


# ------------------------------------------------------------------------- 8 kinetic variants
def kinetic_variants(S):
    a = P(S, rid("s25d"))
    if a is None or KV is None:
        return None
    rows = []
    base = KV[KV.variant == "default"].set_index("case_id")
    for v, K in KV.groupby("variant"):
        K = K.set_index("case_id")
        a2 = a.copy()
        a2["ck"] = ckin(a2, K)
        ch = (K.loc[base.index, KIN_NAMES] - base[KIN_NAMES]).abs().median()
        rows.append({"variant": v, **compare(a2, "ck"), "features_changed": int((ch > 1e-6).sum()),
                     "pe_mean_median_abs_change": float(ch["pe_mean"]), "ser_mean_median_abs_change": float(ch["ser_mean"]),
                     "frac_washout_median_abs_change": float(ch["frac_washout"])})
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "rev3_kinetic_variants.csv"), index=False)
    return R


# ------------------------------------------------------------------------- 9 decision-specific
def decision(S, R):
    a = P(S, rid("s25d"))
    if a is None or R is None or not len(R):
        return None
    a = with_lf(a)
    c = P(S, rid("carfm_25d"))
    a["carfm"] = c.set_index("case_id").loc[a.case_id, "p"].to_numpy() if c is not None else np.nan
    K = KV[KV.variant == "default"].set_index("case_id") if KV is not None else None
    if K is not None:
        a["ck"] = ckin(a, K)
    s0 = R[(R.seed == 0) & R.model.isin(["lf", "carfm_25d", "ckin"])]
    best = s0.sort_values("oof_auc").iloc[-1].model  # best fusion by I-SPY2 out-of-fold AUC, seed 0
    col = {"lf": "lf", "carfm_25d": "carfm", "ckin": "ck"}[best]
    e = a[a.set == "ext"]
    rows = []
    for nm, cc in (("Clinical", "zc"), (best, col)):
        for t in (0.2, 0.3, 0.4):
            k = metrics.classification(e.y.to_numpy(), e[cc].to_numpy(), t)
            ppv = k["tp"] / max(1, k["tp"] + k["fp"]); npv = k["tn"] / max(1, k["tn"] + k["fn"])
            rows.append({"model": nm, "threshold": t, "sensitivity": k["sensitivity"], "specificity": k["specificity"], "ppv": ppv, "npv": npv,
                         "net_benefit": float(metrics.net_benefit(e.y.to_numpy(), e[cc].to_numpy(), [t])[0]),
                         "net_benefit_all": float(metrics.net_benefit_all(e.y.to_numpy(), [t])[0])})
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(OUT, "rev3_decision.csv"), index=False)
    return D


def main():
    global TB, KV
    os.makedirs(OUT, exist_ok=True)
    TB = pd.read_csv(TABLE) if os.path.exists(TABLE) else None
    kf = os.path.join(RAW, "kinetics_variants.csv")
    KV = pd.read_csv(kf) if os.path.exists(kf) else None
    S = load()
    N = {"runs": int(len(S)), "minutes_total": float(S.minutes.sum())}
    _, N["audit"] = audit()
    O = orientation(S)
    N["orientation"] = O.round(4).to_dict(orient="records")
    R, Fo = repeats(S, "lps")
    Ro, _ = repeats(S, "orig")
    N["repeats_lps"] = pd.read_csv(os.path.join(OUT, "rev3_repeats_lps_summary.csv")).round(4).to_dict(orient="records") if len(R) else []
    N["repeats_orig"] = pd.read_csv(os.path.join(OUT, "rev3_repeats_orig_summary.csv")).round(4).to_dict(orient="records") if len(Ro) else []
    N["recovery"] = recovery(S).round(4).to_dict(orient="records")
    N["ablation"] = ablation(S).round(4).to_dict(orient="records")
    N["fov"] = fov(S).round(4).to_dict(orient="records")
    if TB is not None and "nac_agent" in TB:
        tr = treatment(S)
        N["treatment"] = tr["rows"].round(4).to_dict(orient="records")
        N["richer_clinical"] = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in tr.get("rich", [])]
    kv = kinetic_variants(S) if TB is not None else None
    N["kinetic_variants"] = kv.round(4).to_dict(orient="records") if kv is not None else []
    dd = decision(S, R) if TB is not None else None
    N["decision"] = dd.round(4).to_dict(orient="records") if dd is not None else []
    json.dump(N, open(os.path.join(HERE, "rev3_numbers.json"), "w"), indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    print(json.dumps({k: (v if not isinstance(v, list) else f"{len(v)} rows") for k, v in N.items()}, indent=1, default=str))


TB, KV = None, None
if __name__ == "__main__":
    main()
