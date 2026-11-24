"""Fourth-review analyses (paper/analysis_plan_revision4.md): rescoring of all locked external predictions against the
harmonized pathologic endpoint, harmonized repeated-training summaries, and the extended signal-recovery analysis.

    python paper/jbhi/rev4_analyses.py  ->  results/revision4/*.csv, paper/jbhi/rev4_numbers.json
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
from mmonc import metrics, stats  # noqa: E402
from mmonc.models import ClinicalAnchor, make_logreg  # noqa: E402
from mmonc.pipeline5 import BETAS  # noqa: E402

import revision3_analyses as r3  # noqa: E402

OUT = os.path.join(ROOT, "results", "revision4")
R3 = os.path.join(ROOT, "results", "revision3")
RAW3 = os.path.join(R3, "raw")
EXT = ("DUKE", "ISPY1", "NACT")
H = pd.read_csv(os.path.join(R3, "rev3_outcome_check.csv")).set_index("case_id").harmonized


def harmonize(D):
    """Replace the external labels by the harmonized pathologic endpoint; drop external patients without one."""
    D = D.copy()
    e = (D.set == "ext").to_numpy()
    h = D.case_id.map(H)
    D.loc[e, "y"] = h[e]
    return D[~(e & h.isna().to_numpy())].reset_index(drop=True)


def boot(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    i0, i1 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    v = [roc_auc_score(y[b], p[b]) for b in (np.r_[rng.choice(i0, len(i0)), rng.choice(i1, len(i1))] for _ in range(n))]
    return roc_auc_score(y, p), *np.percentile(v, [2.5, 97.5])


def ext_metrics(E, col, ref):
    y = E.y.to_numpy().astype(int)
    a = boot(y, E[col].to_numpy())
    d = stats.delong(y, E[ref].to_numpy(), E[col].to_numpy())
    k = metrics.calibration(y, E[col].to_numpy())
    r = {"auc": a[0], "auc_lo": a[1], "auc_hi": a[2], "diff": d["diff"], "lo": d["ci_lo"], "hi": d["ci_hi"], "p": d["p"], "se": d["se"],
         "cal_int": k["intercept"], "cal_slope": k["slope"], "brier": k["brier"]}
    for c in EXT:
        m = (E.collection == c).to_numpy()
        dc = stats.delong(y[m], E[ref].to_numpy()[m], E[col].to_numpy()[m])
        kc = metrics.calibration(y[m], E[col].to_numpy()[m])
        r.update({f"auc_{c}": roc_auc_score(y[m], E[col].to_numpy()[m]), f"diff_{c}": dc["diff"], f"lo_{c}": dc["ci_lo"], f"hi_{c}": dc["ci_hi"],
                  f"int_{c}": kc["intercept"], f"slope_{c}": kc["slope"]})
    return r


def locked_matrix(endpoint):
    """Every locked model of rounds 1-3 (rev_all_predictions), rescored on the external patients."""
    P = pd.read_csv(os.path.join(ROOT, "results", "revision", "rev_all_predictions.csv"))
    E = P[P.set == "ext"].copy()
    if endpoint == "harmonized":
        E = harmonize(E)
    rows = []
    for m in [c for c in E.columns if c not in ("case_id", "collection", "y", "set") and not c.startswith("gate:")]:
        if E[m].isna().any():
            continue
        ref = "R1: Clinical" if m.startswith("R1:") else "Clinical"
        rows.append({"endpoint": endpoint, "model": m, "reference": ref, "n": len(E), "events": int(E.y.sum()), **ext_metrics(E, m, ref)})
    return pd.DataFrame(rows)


def seeded(endpoint):
    """Reoriented runs of round 3 (seeds 0-4): 2.5D image-only, late fusion (round-3 stacking; external predictions only use
    I-SPY2 data), CARF-mix 2.5D, clinical + kinetic radiomics, CARF-mix kinetic 3D (seed 0 here; seeds 1-4 from rev4 raw)."""
    S = r3.load()
    r3.TB = pd.read_csv(os.path.join(ROOT, "results", "round2", "raw", "table.csv"))
    r3.KV = pd.read_csv(os.path.join(RAW3, "kinetics_variants.csv"))
    K = r3.KV[r3.KV.variant == "default"].set_index("case_id")
    rows = []
    for seed in range(5):
        a = r3.P(S, r3.rid("s25d", seed=seed))
        if a is not None:
            a = r3.with_lf(a)
            a["ck"] = r3.ckin(a, K)
            A = harmonize(a) if endpoint == "harmonized" else a
            E = A[A.set == "ext"]
            for col, nm in (("zc", "Clinical"), ("p", "s25d"), ("lf", "lf"), ("ck", "ckin")):
                rows.append({"endpoint": endpoint, "model": nm, "seed": seed, **ext_metrics(E, col, "zc")})
        for nm in ("carfm_25d", "carfm_k"):
            d = r3.P(S, r3.rid(nm, seed=seed))
            if d is None:
                continue
            D = harmonize(d) if endpoint == "harmonized" else d
            E = D[D.set == "ext"]
            r = {"endpoint": endpoint, "model": nm, "seed": seed, **ext_metrics(E, "p", "zc"), "corr_abs_ext": float(np.nanmean(np.abs(E["corr"])))}
            r["fold_auc_min"] = min(roc_auc_score(E.y, E[f"pf{k}"]) for k in range(5))
            r["fold_auc_max"] = max(roc_auc_score(E.y, E[f"pf{k}"]) for k in range(5))
            rows.append(r)
    extra = os.path.join(OUT, "raw", "carfm_k_seeds1-4_" + endpoint + ".json")
    if os.path.exists(extra):  # computed in the browser from the exported predictions of Kaggle version 8
        for x in json.load(open(extra))["runs"]:
            rows.append({"endpoint": endpoint, "model": "carfm_k", **x})
    return pd.DataFrame(rows)


def recovery_extended(n_draws=500):
    """Signal recovery with oracle and recovered gains against the same refitted clinical model, and a clinical +
    planted-scalar logistic model that receives the engineered signal directly (500 draws per level)."""
    T = pd.read_csv(os.path.join(ROOT, "results", "round2", "raw", "table.csv"))
    K = pd.read_csv(os.path.join(RAW3, "kinetics_variants.csv"))
    K = K[K.variant == "default"].set_index("case_id")
    T = T[T.pcr.notna()].reset_index(drop=True)
    T = T[T.case_id.isin(K.index)].reset_index(drop=True)
    Xc = T[["age", "hr", "her2"]].to_numpy(float)
    dev = (T.collection == "ISPY2").to_numpy(); ext = ~dev
    y = T.pcr.to_numpy().astype(int)
    anc = ClinicalAnchor(n_splits=5); anc.fit(Xc[dev], y[dev]); z = anc.logit(Xc)
    pe = K.loc[T.case_id, "pe_mean"].to_numpy(float)
    Xi = np.where(np.isnan(Xc), np.nanmean(Xc[dev], 0), Xc)
    pe = np.where(np.isfinite(pe), pe, np.nanmean(pe[dev]))
    r = pe - LinearRegression().fit(Xi[dev], pe[dev]).predict(Xi)
    s = (r - r[dev].mean()) / r[dev].std()
    # (a) the two draws used by the pipeline: same-comparator oracle and recovered gains, plus the scalar model
    S = r3.load()
    rows = []
    for d in (0, 1):
        for lev, beta in BETAS.items():
            rid = r3.rid("carfm_25d", labels=f"sig-{lev}-{d}", tab="clin")
            D = r3.P(S, rid)
            if D is None:
                continue
            D = D.merge(T[["case_id"]].assign(s=s, z=z), on="case_id")
            a = r3.with_lf(r3.P(S, r3.rid("s25d", labels=f"sig-{lev}-{d}", tab="clin")).merge(D[["case_id"]], on="case_id"))
            E, Ea = D[D.set == "ext"], a[a.set == "ext"]
            ys = E.y.to_numpy()
            dv = D[D.set == "dev"]
            Xs = np.c_[np.nan_to_num(T.set_index("case_id").loc[dv.case_id, ["age", "hr", "her2"]].to_numpy(float), nan=0.0), dv.s]
            Xe = np.c_[np.nan_to_num(T.set_index("case_id").loc[E.case_id, ["age", "hr", "her2"]].to_numpy(float), nan=0.0), E.s]
            ps = make_logreg(C=1.0).fit(Xs, dv.y).predict_proba(Xe)[:, 1]
            o_same = stats.delong(ys, E.zc.to_numpy(), metrics.sigmoid((E.z + beta * E.s).to_numpy()))
            sc = stats.delong(ys, E.zc.to_numpy(), ps)
            cm = stats.delong(ys, E.zc.to_numpy(), E.p.to_numpy())
            lf = stats.delong(Ea.y.to_numpy(), Ea.zc.to_numpy(), Ea.lf.to_numpy())
            rows.append({"level": lev, "draw": d, "beta": beta,
                         "oracle_original": roc_auc_score(ys, E.z + beta * E.s) - roc_auc_score(ys, E.z),
                         "oracle_same": o_same["diff"], "oracle_same_lo": o_same["ci_lo"], "oracle_same_hi": o_same["ci_hi"],
                         "scalar": sc["diff"], "scalar_lo": sc["ci_lo"], "scalar_hi": sc["ci_hi"],
                         "carfm": cm["diff"], "carfm_lo": cm["ci_lo"], "carfm_hi": cm["ci_hi"],
                         "lf": lf["diff"], "lf_lo": lf["ci_lo"], "lf_hi": lf["ci_hi"]})
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "rev4_recovery_two_draws.csv"), index=False)
    # (b) detectability for a model that receives the signal directly: many independent outcome draws
    Xd_all = np.c_[np.nan_to_num(Xc, nan=0.0)]
    out = []
    for lev, beta in BETAS.items():
        det, gains, orc = 0, [], []
        for i in range(n_draws):
            u = np.random.default_rng(50000 + i).random(len(y))
            ys = (u < metrics.sigmoid(z + beta * s)).astype(int)
            a0 = ClinicalAnchor(n_splits=5); a0.fit(Xc[dev], ys[dev]); zc = a0.logit(Xc[ext])
            ps = LogisticRegression(C=1.0, max_iter=1000).fit(np.c_[Xd_all[dev], s[dev]], ys[dev]).predict_proba(np.c_[Xd_all[ext], s[ext]])[:, 1]
            dd = stats.delong(ys[ext], zc, ps)
            det += dd["ci_lo"] > 0
            gains.append(dd["diff"])
            orc.append(roc_auc_score(ys[ext], z[ext] + beta * s[ext]) - roc_auc_score(ys[ext], zc))
        out.append({"level": lev, "beta": beta, "draws": n_draws, "oracle_same_mean": float(np.mean(orc)), "scalar_gain_mean": float(np.mean(gains)),
                    "scalar_gain_p10": float(np.percentile(gains, 10)), "scalar_gain_p90": float(np.percentile(gains, 90)), "detected": det / n_draws})
    Dt = pd.DataFrame(out)
    Dt.to_csv(os.path.join(OUT, "rev4_scalar_detectability.csv"), index=False)
    return R, Dt


def main():
    os.makedirs(OUT, exist_ok=True)
    N = {}
    M = pd.concat([locked_matrix("harmonized"), locked_matrix("distributed")], ignore_index=True)
    M.to_csv(os.path.join(OUT, "rev4_locked_matrix.csv"), index=False)
    Sd = pd.concat([seeded("harmonized"), seeded("distributed")], ignore_index=True)
    Sd.to_csv(os.path.join(OUT, "rev4_seeded.csv"), index=False)
    R, Dt = recovery_extended()
    N["clinical_harmonized"] = M[(M.endpoint == "harmonized") & (M.model == "Clinical")].iloc[0].to_dict() if False else None
    print(M[M.model.isin(["R1: CARF (proposed)", "Clinical + FTV", "Clinical + kinetic radiomics", "CARF-mix, pretrained 2.5D", "CARF-mix, kinetic"])]
          [["endpoint", "model", "auc", "diff", "lo", "hi", "cal_int", "cal_slope"]].round(3).to_string())
    print(Sd.groupby(["endpoint", "model"]).agg(n=("seed", "size"), auc=("auc", "mean"), dmin=("diff", "min"), dmax=("diff", "max"), lo=("lo", "min"), hi=("hi", "max"),
                                                  ci=("cal_int", "mean"), cs=("cal_slope", "mean")).round(3).to_string())
    print(R.round(3).to_string())
    print(Dt.round(3).to_string())


if __name__ == "__main__":
    main()
