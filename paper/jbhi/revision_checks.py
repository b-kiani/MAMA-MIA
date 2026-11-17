"""Summaries of the pipeline checks run in revision (positive controls, learning curves, overfitting
test, input dependency), from results/revision/raw.

    python paper/jbhi/revision_checks.py [raw_dir]   ->  results/revision/rev_checks_*.csv, figures/fig_checks.*
"""
from __future__ import annotations

import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
REV = os.path.join(ROOT, "results", "revision")
RAW = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REV, "raw")
OUT = REV if len(sys.argv) < 2 else RAW
FIG = os.path.join(HERE, "figures") if len(sys.argv) < 2 else RAW
BLUE, ORANGE, GREEN, GREY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#6b7280", "#111827"
TARGET = {"pe_mean_high": "Mean enhancement above median", "frac_washout_high": "Washout fraction above median", "vendor_ge": "Scanner vendor (GE vs other)"}
NET = {"Image only, kinetic 3D ResNet": "Kinetic 3D ResNet", "Image only, pretrained 2.5D": "Pretrained 2.5D"}


def read(pattern):
    fs = sorted(glob.glob(os.path.join(RAW, pattern)))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else None


def auc_ci(y, p, n=2000, seed=0):
    y, p = np.asarray(y), np.asarray(p)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    i0, i1 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    v = [roc_auc_score(y[b], p[b]) for b in (np.r_[rng.choice(i0, len(i0)), rng.choice(i1, len(i1))] for _ in range(n))]
    return roc_auc_score(y, p), *np.percentile(v, [2.5, 97.5])


def positive_controls():
    D = read("preds_posctl_*.csv")
    if D is None:
        return None
    rows = []
    for (m, t), g in D.groupby(["model", "target"], sort=False):
        for st, lab in (("dev", "I-SPY2 cross-validation"), ("ext", "External")):
            s = g[g.set == st]
            if len(s):
                a, lo, hi = auc_ci(s.y, s.p)
                rows.append({"model": m, "target": t, "cohort": lab, "n": len(s), "positives": int(s.y.sum()), "auc": a, "lo": lo, "hi": hi})
                if st == "ext":
                    for c, h in s.groupby("collection"):
                        if h.y.nunique() > 1:
                            rows.append({"model": m, "target": t, "cohort": c, "n": len(h), "positives": int(h.y.sum()), "auc": roc_auc_score(h.y, h.p)})
    return pd.DataFrame(rows)


def learning_curves():
    D, T = read("preds_lc_*.csv"), read("lc_train_*.csv")
    if D is None:
        return None
    rows = []
    for (m, f), g in D.groupby(["model", "fraction"], sort=False):
        r = {"model": m, "fraction": f}
        for st in ("dev", "ext"):
            s = g[g.set == st]
            a, lo, hi = auc_ci(s.y, s.p)
            r.update({f"{st}_auc": a, f"{st}_lo": lo, f"{st}_hi": hi})
        if T is not None:
            t = T[(T.model == m) & (T.fraction == f)]
            r.update({"train_auc_mean": t.train_auc.mean(), "train_auc_min": t.train_auc.min(), "train_auc_max": t.train_auc.max(), "n_train": t.n_train.mean()})
        rows.append(r)
    # full training folds: the main round-2 models (same folds and protocol)
    P = pd.read_csv(os.path.join(ROOT, "results", "round2", "r2_predictions.csv"))
    for m in D.model.unique():
        if m in P:
            r = {"model": m, "fraction": 1.0}
            for st in ("dev", "ext"):
                s = P[P.set == st]
                a, lo, hi = auc_ci(s.y, s[m])
                r.update({f"{st}_auc": a, f"{st}_lo": lo, f"{st}_hi": hi})
            rows.append(r)
    return pd.DataFrame(rows).sort_values(["model", "fraction"])


def figure(pc, lc, ov):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1.0, 1.0, 1.0]})
    a = ax[0]  # positive controls
    if pc is not None:
        s = pc[pc.cohort.isin(["I-SPY2 cross-validation", "External"])]
        labs, k = [], 0
        for t in TARGET:
            for m in NET:
                for coh, col, off in (("I-SPY2 cross-validation", INK, -0.17), ("External", ORANGE, 0.17)):
                    r = s[(s.model == m) & (s.target == t) & (s.cohort == coh)]
                    if len(r):
                        r = r.iloc[0]
                        a.errorbar(r.auc, k + off, xerr=[[r.auc - r.lo], [r.hi - r.auc]], fmt="o", ms=3, color=col, elinewidth=0.9)
                labs.append(f"{TARGET[t].split(' (')[0]}\n{NET[m]}"); k += 1
        a.set_yticks(range(len(labs))); a.set_yticklabels(labs, fontsize=5.8); a.invert_yaxis()
        a.axvline(0.5, color=GREY, lw=0.7, ls="--"); a.set_xlim(0.4, 1.03)
        a.plot([], [], "o", ms=3, color=INK, label="I-SPY2 cross-validation"); a.plot([], [], "o", ms=3, color=ORANGE, label="External")
    a.set_xlabel("AUC (95% CI)"); a.set_title("a  Positive controls", fontsize=7.4, loc="left")
    a = ax[1]  # learning curves
    if lc is not None:
        for m, col in zip(NET, (BLUE, GREEN)):
            s = lc[lc.model == m]
            if not len(s):
                continue
            a.errorbar(100 * s.fraction, s.dev_auc, yerr=[s.dev_auc - s.dev_lo, s.dev_hi - s.dev_auc], fmt="o-", ms=3, lw=1, color=col)
            t = s[s["train_auc_mean"].notna()] if "train_auc_mean" in s else s.iloc[:0]
            a.plot(100 * t.fraction, t.train_auc_mean, "s--", ms=3, lw=1, color=col, mfc="white")
        a.axhline(0.5, color=GREY, lw=0.7, ls="--"); a.set_xticks([25, 50, 100]); a.set_ylim(0.4, 1.03)
    a.set_xlabel("Share of each training fold used, %"); a.set_ylabel("pCR AUC"); a.set_title("b  Learning curves", fontsize=7.4, loc="left")
    a = ax[2]  # overfitting test
    if ov is not None:
        for m, col in zip(NET, (BLUE, GREEN)):
            s = ov[ov.model == m]
            a.plot(s.epoch, s.heldout_auc, "o-", ms=2.5, lw=1, color=col)
            a.plot(s.epoch, s.train_auc, "s--", ms=2.5, lw=1, color=col, mfc="white")
        a.axhline(0.5, color=GREY, lw=0.7, ls="--"); a.set_ylim(0.4, 1.03)
    a.set_xlabel("Epoch"); a.set_ylabel("pCR AUC"); a.set_title("c  Overfitting test", fontsize=7.4, loc="left")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=BLUE, marker="o", ms=3, lw=1, label="Kinetic 3D ResNet, held-out patients"),
          Line2D([], [], color=BLUE, marker="s", ms=3, lw=1, ls="--", mfc="white", label="Kinetic 3D ResNet, training patients"),
          Line2D([], [], color=GREEN, marker="o", ms=3, lw=1, label="Pretrained 2.5D, held-out patients"),
          Line2D([], [], color=GREEN, marker="s", ms=3, lw=1, ls="--", mfc="white", label="Pretrained 2.5D, training patients")]
    ha = [Line2D([], [], color=INK, marker="o", ms=3, lw=0, label="I-SPY2 cross-validation"), Line2D([], [], color=ORANGE, marker="o", ms=3, lw=0, label="External")]
    fig.add_artist(fig.legend(handles=ha, loc="lower left", ncol=1, fontsize=6, frameon=False, bbox_to_anchor=(0.2, 0.0), title="Panel a", title_fontsize=6))
    fig.legend(handles=hs, loc="lower right", ncol=2, fontsize=6, frameon=False, bbox_to_anchor=(0.99, 0.0), title="Panels b and c", title_fontsize=6)
    for a in ax:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.tick_params(labelsize=6.2)
    fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=0.8)
    fig.savefig(os.path.join(FIG, "fig_checks.pdf")); fig.savefig(os.path.join(FIG, "fig_checks.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    pc, lc, ov, dep = positive_controls(), learning_curves(), read("overfit_*.csv"), read("dependency_*.csv")
    for name, t in (("rev_checks_positive_controls", pc), ("rev_checks_learning_curves", lc), ("rev_checks_overfit", ov)):
        if t is not None:
            t.to_csv(os.path.join(OUT, name + ".csv"), index=False)
            print("==", name); print(t.round(3).to_string(index=False))
    if dep is not None:
        if "set" not in dep:
            dep["set"] = "all"
        g = dep.groupby(["model", "set"], sort=False).agg(fits=("n", "size"), tab=("mean_abs_change_logit_tab_permuted", "mean"), img=("mean_abs_change_logit_image_permuted", "mean"),
                                                         mean_gd=("mean_abs_gdelta", "mean"), p99_gd=("p99_abs_gdelta", "max"), max_gd=("max_abs_gdelta", "max"),
                                                         gate=("mean_gate", "mean")).reset_index()
        g.to_csv(os.path.join(OUT, "rev_checks_dependency.csv"), index=False)
        print("== rev_checks_dependency"); print(g.round(4).to_string(index=False))
    figure(pc, lc, ov)
