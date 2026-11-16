"""Re-analyses requested in review, computed from the per-patient predictions.

    python paper/jbhi/revision_analyses.py   ->  results/revision/*.csv

1. per-collection paired AUC differences (DeLong) and a collection-stratified AUC difference
   that uses only within-collection patient pairs (bootstrap stratified by collection)
2. LOCO pooling: weights, fixed- and random-effects estimates, equal-weight sensitivity
3. receptor-subgroup denominators
4. leave-one-external-collection-out sensitivity of the pooled external difference
5. external calibration (calibration-in-the-large, expected/observed)
6. decision curves: net-benefit difference from the clinical model with bootstrap CIs
7. decision rule with separate superiority and clinical-importance questions, margins 0.02/0.03/0.05
8. empirical size of the imaging correction g*delta of the original CARF models
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
from mmonc import metrics, stats  # noqa: E402

R2 = os.path.join(ROOT, "results", "round2")
REV = os.path.join(ROOT, "results", "revision")
OUT = REV
os.makedirs(OUT, exist_ok=True)
META = {"case_id", "collection", "y", "set"}
EXTC = ["DUKE", "ISPY1", "NACT"]
NAMES = {"DUKE": "Duke", "ISPY1": "I-SPY1", "NACT": "NACT", "ISPY2": "I-SPY2"}


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def load_predictions():
    """Round-2 per-patient predictions, plus round-1 and revision models when available."""
    P = pd.read_csv(os.path.join(R2, "r2_predictions.csv"))
    r1 = os.path.join(REV, "raw", "r1_predictions_external.csv")
    if os.path.exists(r1):
        R = pd.read_csv(r1)
        keep = [c for c in R.columns if c in ("case_id", "Clinical") or c.startswith(("CARF", "Image only", "Concatenation", "Radiomics", "Clinical +"))]
        R = R[keep].rename(columns={c: "R1: " + c for c in keep if c != "case_id"})
        P = P.merge(R, on="case_id", how="left")
    for f in sorted(glob.glob(os.path.join(REV, "raw", "preds_devext_g*.csv")) + glob.glob(os.path.join(REV, "raw", "preds_dino_*.csv"))):
        D = pd.read_csv(f)
        new = [c for c in D.columns if c not in META and c not in P.columns]
        if new:
            P = P.merge(D[["case_id"] + new], on="case_id", how="left")
    return P


def ref_of(P, m):
    return "R1: Clinical" if m.startswith("R1:") and "R1: Clinical" in P.columns else "Clinical"


def models_of(P):
    return [c for c in P.columns if c not in META and not c.startswith("gate:") and c not in ("Clinical", "R1: Clinical")]


def stratified_auc(y, p, g):
    """AUC over positive-negative pairs from the same collection only."""
    num = den = 0.0
    for c in np.unique(g):
        s = g == c
        if 0 < y[s].sum() < s.sum():
            w = y[s].sum() * (s.sum() - y[s].sum())
            num += w * roc_auc_score(y[s], p[s]); den += w
    return num / den


def per_collection(P, n_boot=2000):
    E = P[P.set == "ext"].reset_index(drop=True)
    y, g = E.y.to_numpy(), E.collection.to_numpy()
    rng = np.random.default_rng(0)
    idx_by = {c: np.flatnonzero(g == c) for c in EXTC}
    rows = []
    for m in models_of(P):
        if E[m].isna().any():
            continue
        for c in EXTC:
            s = g == c
            d = stats.delong(y[s], E[ref_of(P, m)].to_numpy()[s], E[m].to_numpy()[s])
            rows.append({"model": m, "collection": NAMES[c], "n": int(s.sum()), "events": int(y[s].sum()),
                         "auc_clinical": d["auc1"], "auc_model": d["auc2"], "diff": d["diff"], "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"], "p": d["p"]})
        d = stats.delong(y, E[ref_of(P, m)], E[m])
        est = stratified_auc(y, E[m].to_numpy(), g) - stratified_auc(y, E[ref_of(P, m)].to_numpy(), g)
        boots = []
        for _ in range(n_boot):
            b = np.concatenate([rng.choice(idx_by[c], len(idx_by[c])) for c in EXTC])
            try:
                boots.append(stratified_auc(y[b], E[m].to_numpy()[b], g[b]) - stratified_auc(y[b], E[ref_of(P, m)].to_numpy()[b], g[b]))
            except ValueError:
                pass
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({"model": m, "collection": "Pooled (all pairs)", "n": len(y), "events": int(y.sum()), "auc_clinical": d["auc1"],
                     "auc_model": d["auc2"], "diff": d["diff"], "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"], "p": d["p"]})
        rows.append({"model": m, "collection": "Within-collection pairs", "n": len(y), "events": int(y.sum()),
                     "auc_clinical": stratified_auc(y, E[ref_of(P, m)].to_numpy(), g), "auc_model": stratified_auc(y, E[m].to_numpy(), g),
                     "diff": est, "ci_lo": lo, "ci_hi": hi, "p": np.nan})
    return pd.DataFrame(rows)


def loco_weights():
    t = pd.read_csv(os.path.join(R2, "r2_loco.csv"))
    rows = []
    for m, s in t[t.se.notna()].groupby("model"):
        se = s.se.to_numpy()
        w = 1 / se ** 2
        fe = float((w * s["diff"]).sum() / w.sum())
        fe_se = float(np.sqrt(1 / w.sum()))
        q = float((w * (s["diff"] - fe) ** 2).sum())
        k = len(s)
        tau2 = max(0.0, (q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
        wr = 1 / (se ** 2 + tau2)
        re = float((wr * s["diff"]).sum() / wr.sum()); re_se = float(np.sqrt(1 / wr.sum()))
        wn = s.n.to_numpy() / s.n.sum()
        ew = float((wn * s["diff"]).sum())
        for (_, r), wi, wri in zip(s.iterrows(), w / w.sum(), wr / wr.sum()):
            rows.append({"model": m, "held_out": NAMES[r.held_out], "diff": r["diff"], "se": r.se, "weight_fixed": wi, "weight_random": wri})
        rows.append({"model": m, "held_out": "Fixed effect", "diff": fe, "ci_lo": fe - 1.96 * fe_se, "ci_hi": fe + 1.96 * fe_se})
        rows.append({"model": m, "held_out": "Random effects (DL)", "diff": re, "ci_lo": re - 1.96 * re_se, "ci_hi": re + 1.96 * re_se, "tau2": tau2})
        rows.append({"model": m, "held_out": "Sample-size weighted", "diff": ew})
    return pd.DataFrame(rows)


def subgroup_counts(P):
    T = pd.read_csv(os.path.join(R2, "raw", "table.csv"))[["case_id", "hr", "her2"]]
    E = P[P.set == "ext"].merge(T, on="case_id")
    ok = E.hr.notna() & E.her2.notna()
    out = E.assign(receptors_known=ok).groupby(["collection", "receptors_known"]).agg(n=("y", "size"), events=("y", "sum")).reset_index()
    return out


def leave_one_out(P):
    E = P[P.set == "ext"]
    rows = []
    for m in models_of(P):
        if E[m].isna().any():
            continue
        for c in [None] + EXTC:
            s = E if c is None else E[E.collection != c]
            d = stats.delong(s.y, s[ref_of(P, m)], s[m])
            rows.append({"model": m, "excluded": "none" if c is None else NAMES[c], "n": len(s), "diff": d["diff"], "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"]})
    return pd.DataFrame(rows)


def calibration(P):
    rows = []
    for st in ("dev", "ext"):
        E = P[P.set == st]
        for m in ["Clinical"] + models_of(P):
            if E[m].isna().any():
                continue
            for c, g in [("All", E)] + list(E.groupby("collection")):
                if st == "dev" and c != "All":
                    continue
                cal = metrics.calibration(g.y.to_numpy(), g[m].to_numpy())
                rows.append({"set": "I-SPY2 (out-of-fold)" if st == "dev" else "External", "collection": NAMES.get(c, c), "model": m,
                             "n": len(g), "observed_rate": g.y.mean(), "mean_predicted": g[m].mean(), "expected_over_observed": g[m].mean() / g.y.mean(),
                             "intercept": cal["intercept"], "slope": cal["slope"], "brier": cal["brier"]})
    return pd.DataFrame(rows)


def decision_curves(P, thresholds=(0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5), n_boot=2000):
    E = P[P.set == "ext"].reset_index(drop=True)
    y = E.y.to_numpy()
    rng = np.random.default_rng(1)
    B = [rng.integers(0, len(y), len(y)) for _ in range(n_boot)]
    rows = []
    for m in ["Clinical"] + models_of(P):
        if E[m].isna().any():
            continue
        p, pc = E[m].to_numpy(), E[ref_of(P, m)].to_numpy()
        nb = metrics.net_benefit(y, p, thresholds)
        nbc = metrics.net_benefit(y, pc, thresholds)
        nball = metrics.net_benefit_all(y, thresholds)
        bd = np.array([metrics.net_benefit(y[b], p[b], thresholds) - metrics.net_benefit(y[b], pc[b], thresholds) for b in B])
        bn = np.array([metrics.net_benefit(y[b], p[b], thresholds) - np.maximum(metrics.net_benefit_all(y[b], thresholds), 0) for b in B])
        for k, t in enumerate(thresholds):
            rows.append({"model": m, "threshold": t, "net_benefit": nb[k], "nb_treat_all": nball[k],
                         "nb_vs_best_default": nb[k] - max(nball[k], 0), "nbd_lo": np.percentile(bn[:, k], 2.5), "nbd_hi": np.percentile(bn[:, k], 97.5),
                         "nb_minus_clinical": nb[k] - nbc[k], "lo": np.percentile(bd[:, k], 2.5), "hi": np.percentile(bd[:, k], 97.5)})
    return pd.DataFrame(rows)


def decision_rule(per_coll, margins=(0.02, 0.03, 0.05)):
    s = per_coll[per_coll.collection == "Pooled (all pairs)"]
    rows = []
    for r in s.itertuples():
        for mg in margins:
            sup = "superior" if r.ci_lo > 0 else ("inferior" if r.ci_hi < 0 else "not shown")
            if r.ci_lo > mg:
                imp = "clinically important gain"
            elif r.ci_lo > -mg and r.ci_hi < mg:
                imp = "equivalent (no clinically important difference)"
            elif r.ci_hi < -mg:
                imp = "clinically important loss"
            elif r.ci_hi < mg:
                imp = "no clinically important gain (loss not excluded)"
            else:
                imp = "inconclusive"
            rows.append({"model": r.model, "margin": mg, "diff": r.diff, "ci_lo": r.ci_lo, "ci_hi": r.ci_hi, "statistical": sup, "clinical": imp})
    return pd.DataFrame(rows)


def gdelta(P):
    rows = []
    for m in [c for c in P.columns if c.startswith(("CARF", "R1: CARF")) and not c.startswith("gate:")]:
        for st in ("dev", "ext"):
            s = P[(P.set == st) & P[m].notna()]
            if not len(s):
                continue
            d = logit(s[m]) - logit(s[ref_of(P, m)])
            rows.append({"model": m, "set": "I-SPY2 (out-of-fold)" if st == "dev" else "External (5-fold average)", "n": len(s),
                         "mean_abs": float(np.abs(d).mean()), "p99_abs": float(np.percentile(np.abs(d), 99)), "max_abs": float(np.abs(d).max()),
                         "max_abs_prob_change": float(np.abs(s[m] - s[ref_of(P, m)]).max())})
    return pd.DataFrame(rows)


def main():
    P = load_predictions()
    P.to_csv(os.path.join(OUT, "rev_all_predictions.csv"), index=False)
    tabs = {"rev_per_collection": per_collection(P), "rev_loco_weights": loco_weights(), "rev_subgroup_counts": subgroup_counts(P),
            "rev_leave_one_out": leave_one_out(P), "rev_calibration": calibration(P), "rev_decision_curves": decision_curves(P),
            "rev_gdelta": gdelta(P)}
    tabs["rev_decision_rule"] = decision_rule(tabs["rev_per_collection"])
    for k, v in tabs.items():
        v.to_csv(os.path.join(OUT, k + ".csv"), index=False)
        print(f"== {k}")
        print(v.round(3).to_string(index=False, max_rows=80))


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    main()
