"""Stronger clinical comparators (second review, point 5). Post hoc; selected on I-SPY2 only.

    python paper/jbhi/revision_clinical.py  ->  results/revision/rev_clinical_*.csv

Candidates (all L2 logistic, C=1, median imputation + missing indicators inside each fit):
  A  age + HR + HER2 (main analysis)
  B  A + HR x HER2 interaction (equivalent to four receptor-subtype indicators)
  C  A + natural cubic spline of age (3 knots at the 10th, 50th, 90th development percentiles)
  D  B + spline of age
  E  D + age x HR and age x HER2
Selection: highest mean AUC over 20 repeats of fivefold cross-validation on I-SPY2.
The selected model then replaces the clinical reference for (i) all stored external predictions
(paired DeLong) and (ii) refitted tabular additions (FTV, kinetic radiomics).
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
from mmonc import metrics, stats  # noqa: E402
from mmonc.models import make_logreg  # noqa: E402
from mmonc.prep2 import KIN_NAMES  # noqa: E402

OUT = os.path.join(ROOT, "results", "revision")
RAW = os.path.join(ROOT, "results", "round2", "raw")


def ns_basis(x, knots):
    """Natural cubic spline basis (Harrell's truncated-power form), len(knots) - 2 nonlinear terms."""
    k = np.asarray(knots, float)
    K = len(k)
    def tp(u):
        return np.where(u > 0, u, 0.0) ** 3
    cols = []
    for j in range(K - 2):
        cols.append(tp(x - k[j]) - tp(x - k[K - 2]) * (k[K - 1] - k[j]) / (k[K - 1] - k[K - 2])
                    + tp(x - k[K - 1]) * (k[K - 2] - k[j]) / (k[K - 1] - k[K - 2]))
    return np.column_stack(cols) / (k[-1] - k[0]) ** 2


def design(T, spec, knots):
    age, hr, her2 = T.age.to_numpy(float), T.hr.to_numpy(float), T.her2.to_numpy(float)
    cols = [age, hr, her2]
    if spec in ("B", "D", "E"):
        cols.append(hr * her2)
    if spec in ("C", "D", "E"):
        a = np.where(np.isnan(age), np.nanmedian(age), age)
        S = ns_basis(a, knots)
        S[np.isnan(age)] = np.nan
        cols += list(S.T)
    if spec == "E":
        cols += [age * hr, age * her2]
    return np.column_stack(cols)


def main():
    T = pd.read_csv(os.path.join(RAW, "table.csv"))
    K = pd.read_csv(os.path.join(RAW, "kinetics.csv"), index_col=0).loc[T.case_id, KIN_NAMES].to_numpy(float)
    keep = T.pcr.notna().to_numpy()
    T, K = T[keep].reset_index(drop=True), K[keep]
    y = T.pcr.to_numpy().astype(int)
    dev, ext = (T.collection == "ISPY2").to_numpy(), (T.collection != "ISPY2").to_numpy()
    knots = np.nanpercentile(T.age[dev], [10, 50, 90])
    names = {"A": "Age + HR + HER2 (main)", "B": "+ HR x HER2", "C": "+ spline of age", "D": "+ HR x HER2 + spline of age",
             "E": "+ HR x HER2 + spline of age + age x receptor"}
    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=0)
    rows, ext_pred, X = [], {}, {}
    for s in names:
        X[s] = design(T, s, knots)
        aucs = []
        for tr, te in rcv.split(X[s][dev], y[dev]):
            m = make_logreg(1.0).fit(X[s][dev][tr], y[dev][tr])
            aucs.append(roc_auc_score(y[dev][te], m.predict_proba(X[s][dev][te])[:, 1]))
        aucs = np.array(aucs)
        p = make_logreg(1.0).fit(X[s][dev], y[dev]).predict_proba(X[s][ext])[:, 1]
        ext_pred[s] = p
        rows.append({"model": s, "description": names[s], "n_terms": X[s].shape[1], "cv_auc_mean": aucs.mean(),
                     "cv_auc_sd_over_repeats": aucs.reshape(20, 5).mean(1).std(), "ext_auc": roc_auc_score(y[ext], p)})
    R = pd.DataFrame(rows)
    best = R.sort_values("cv_auc_mean", ascending=False).model.iloc[0]
    R["selected"] = R.model == best
    ye, ge = y[ext], T.collection.to_numpy()[ext]
    base = ext_pred["A"]
    for r in R.itertuples():
        d = stats.delong(ye, base, ext_pred[r.model])
        R.loc[r.Index, ["diff_vs_main", "ci_lo", "ci_hi"]] = [d["diff"], d["ci_lo"], d["ci_hi"]]
        cal = metrics.calibration(ye, ext_pred[r.model])
        R.loc[r.Index, ["intercept", "slope", "e_over_o"]] = [cal["intercept"], cal["slope"], ext_pred[r.model].mean() / ye.mean()]
    R.to_csv(os.path.join(OUT, "rev_clinical_candidates.csv"), index=False)

    # imaging-based models against the selected stronger baseline
    S = ext_pred[best]
    P = pd.read_csv(os.path.join(OUT, "rev_all_predictions.csv"))
    E = P[P.set == "ext"].reset_index(drop=True)
    idx = pd.Series(np.arange(len(T)), index=T.case_id)
    S_e = S[(idx.loc[E.case_id].to_numpy() - np.flatnonzero(ext)[0]) * 0 + np.searchsorted(np.flatnonzero(ext), idx.loc[E.case_id].to_numpy())]
    assert np.allclose(E.y.to_numpy(), ye[np.searchsorted(np.flatnonzero(ext), idx.loc[E.case_id].to_numpy())])
    out = []
    models = [c for c in P.columns if c not in ("case_id", "collection", "y", "set") and not c.startswith("gate:") and E[c].notna().all()]
    for m in models:
        d = stats.delong(E.y.to_numpy(), S_e, E[m].to_numpy())
        out.append({"model": m, "reference": f"stronger clinical ({best})", "auc_reference": d["auc1"], "auc_model": d["auc2"],
                    "diff": d["diff"], "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"], "p": d["p"], "refit": False})
    # tabular additions refitted on top of the stronger clinical design
    ftv = K[:, [KIN_NAMES.index("log_ftv_mm3")]]
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    for label, Xa, pen, C in [("Stronger clinical + FTV", np.hstack([X[best], ftv]), "l2", 1.0),
                              ("Stronger clinical + kinetic radiomics", np.hstack([X[best], K]), "l1", 0.3)]:
        mdl = make_logreg(C=C, penalty=pen)
        oof = cross_val_predict(mdl, Xa[dev], y[dev], cv=cv, method="predict_proba")[:, 1]
        oof0 = cross_val_predict(make_logreg(1.0), X[best][dev], y[dev], cv=cv, method="predict_proba")[:, 1]
        p = mdl.fit(Xa[dev], y[dev]).predict_proba(Xa[ext])[:, 1]
        d = stats.delong(ye, S, p)
        di = stats.delong(y[dev], oof0, oof)
        out.append({"model": label, "reference": f"stronger clinical ({best})", "auc_reference": d["auc1"], "auc_model": d["auc2"],
                    "diff": d["diff"], "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"], "p": d["p"], "refit": True,
                    "internal_diff": di["diff"], "internal_lo": di["ci_lo"], "internal_hi": di["ci_hi"]})
        for c in ("DUKE", "ISPY1", "NACT"):
            mk = ge == c
            dc = stats.delong(ye[mk], S[mk], p[mk])
            out.append({"model": label, "reference": f"stronger clinical ({best}), {c}", "auc_reference": dc["auc1"], "auc_model": dc["auc2"],
                        "diff": dc["diff"], "ci_lo": dc["ci_lo"], "ci_hi": dc["ci_hi"], "p": dc["p"], "refit": True})
    O = pd.DataFrame(out)
    O.to_csv(os.path.join(OUT, "rev_clinical_vs_stronger.csv"), index=False)
    pd.set_option("display.width", 250)
    print(R.round(4).to_string(index=False))
    print(O.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
