"""Second-review results from results/revision2/raw (optimization study, nested refit, overfitting test).

    python paper/jbhi/revision2_analyses.py  ->  results/revision2/*.csv, paper/jbhi/rev_opt_summary.tex,
                                                rev_nested_summary.tex, rev2_numbers.json, figures/fig_opt.*
"""
from __future__ import annotations

import glob
import json
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
sys.path.insert(0, os.path.join(ROOT, "src"))
from mmonc import stats  # noqa: E402

RAW = os.environ.get("REV2_RAW", os.path.join(ROOT, "results", "revision2", "raw"))
OUT = os.path.join(ROOT, "results", "revision2")
REV = os.path.join(ROOT, "results", "revision")
NAMES = {"DUKE": "Duke", "ISPY1": "I-SPY1", "NACT": "NACT"}
CFG = {"base": "Round-2 protocol", "long_auc": "150 epochs, stop on AUC", "long_loss": "150 epochs, stop on loss",
       "fixed": "150 epochs, no early stopping", "small_loss": "Narrow network, stop on loss"}


def read(p):
    fs = sorted(glob.glob(os.path.join(RAW, p)))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else None


def merge_preds(pattern):
    fs = sorted(glob.glob(os.path.join(RAW, pattern)))
    P = None
    for f in fs:
        d = pd.read_csv(f)
        P = d if P is None else P.merge(d[[c for c in d.columns if c not in ("collection", "y", "set")]], on="case_id", how="outer")
    return P


def boot(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    i0, i1 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    v = [roc_auc_score(y[b], p[b]) for b in (np.r_[rng.choice(i0, len(i0)), rng.choice(i1, len(i1))] for _ in range(n))]
    return roc_auc_score(y, p), *np.percentile(v, [2.5, 97.5])


def f3(x):
    return "--" if pd.isna(x) else f"{x + 0.0:.3f}".replace("-0.000", "0.000").replace("-", "$-$")


def ci(a, lo, hi):
    return f"{f3(a)} ({f3(lo)} to {f3(hi)})"


def table(caption, label, cols, rows, spec, note):
    out = ["\\begin{table}[!htbp]", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
           f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(cols) + "\\\\", "\\midrule"] + [" & ".join(r) + "\\\\" for r in rows]
    out += ["\\bottomrule", f"\\multicolumn{{{len(cols)}}}{{p{{0.95\\linewidth}}}}{{\\scriptsize {note}}}\\\\", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(out)


def main():
    os.makedirs(OUT, exist_ok=True)
    N = {}
    ref = pd.read_csv(os.path.join(REV, "rev_all_predictions.csv"))[["case_id", "set", "y", "Clinical"]]
    # ------------------------------------------------------------ optimization study
    T = read("opt_train_*.csv")
    P = merge_preds("preds_opt_*.csv")
    if T is not None and P is not None:
        P = P.merge(ref[["case_id", "Clinical"]], on="case_id")
        D, E = P[P.set == "dev"], P[P.set == "ext"]
        rows, summ = [], []
        for c in CFG:
            col = f"opt:{c}"
            if col not in P or P[col].isna().any():
                continue
            t = T[T.config == c]
            a = boot(D.y.to_numpy(), D[col].to_numpy())
            b = boot(E.y.to_numpy(), E[col].to_numpy())
            d = stats.delong(E.y.to_numpy(), E.Clinical.to_numpy(), E[col].to_numpy())
            r = {"config": c, "train_auc_mean": t.train_auc.mean(), "train_auc_min": t.train_auc.min(), "train_auc_max": t.train_auc.max(),
                 "epochs_run_mean": t.epochs_run.mean(), "epoch_kept_mean": t.epoch_kept.mean(), "minutes_per_fold": t.minutes.mean(),
                 "oof_auc": a[0], "oof_lo": a[1], "oof_hi": a[2], "ext_auc": b[0], "ext_lo": b[1], "ext_hi": b[2],
                 "diff": d["diff"], "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"]}
            for k, nm in NAMES.items():
                m = (E.collection == k).to_numpy()
                dc = stats.delong(E.y.to_numpy()[m], E.Clinical.to_numpy()[m], E[col].to_numpy()[m])
                r.update({f"diff_{nm}": dc["diff"], f"lo_{nm}": dc["ci_lo"], f"hi_{nm}": dc["ci_hi"]})
            summ.append(r)
        S = pd.DataFrame(summ)
        S["selected"] = S.oof_auc == S.oof_auc.max()
        S.to_csv(os.path.join(OUT, "rev2_opt_summary.csv"), index=False)
        sel = S[S.selected].iloc[0]
        N["opt"] = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer, bool, np.bool_)) else str(v)) for k, v in sel.items()}
        N["opt_all"] = S.round(4).to_dict(orient="records")
        rows = [[CFG[r.config] + (" (selected)" if r.selected else ""), ci(r["diff"], r.ci_lo, r.ci_hi)]
                + [ci(r[f"diff_{c}"], r[f"lo_{c}"], r[f"hi_{c}"]) for c in ("Duke", "I-SPY1", "NACT")] for _, r in S.iterrows()]
        open(os.path.join(HERE, "rev_opt_summary.tex"), "w").write(table(
            "Optimization Study: External AUC Difference from the Clinical Model, Pooled and Within Each Collection (95\\% CI)", "tab:s_opt",
            ["Configuration", "Pooled external", "Duke", "I-SPY1", "NACT"],
            rows, "lcccc",
            "Paired DeLong differences. Training and internal results are in main Table V. The narrow-network configuration was not completed. Configurations were fixed before the run (paper/analysis\\_plan\\_revision2.md)."))
        # figure: training vs held-out AUC per configuration
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
        fig, ax = plt.subplots(figsize=(4.6, 2.4))
        x = np.arange(len(S))
        ax.bar(x - 0.2, S.train_auc_mean, 0.38, color="#2a78d6", label="Training patients (mean over folds)")
        ax.errorbar(x - 0.2, S.train_auc_mean, yerr=[S.train_auc_mean - S.train_auc_min, S.train_auc_max - S.train_auc_mean], fmt="none", ecolor="#111827", lw=0.8)
        ax.bar(x + 0.2, S.oof_auc, 0.38, color="#eb6834", label="I-SPY2 held-out folds")
        ax.errorbar(x + 0.2, S.oof_auc, yerr=[S.oof_auc - S.oof_lo, S.oof_hi - S.oof_auc], fmt="none", ecolor="#111827", lw=0.8)
        ax.axhline(0.8, color="#6b7280", ls=":", lw=0.8)
        ax.axhline(0.5, color="#6b7280", ls="--", lw=0.7)
        ax.set_xticks(x); ax.set_xticklabels([CFG[c].replace(", ", ",\n") for c in S.config], fontsize=6)
        ax.set_ylim(0.4, 1.02); ax.set_ylabel("pCR AUC")
        ax.legend(fontsize=6, frameon=False, loc="upper right")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "figures", "fig_opt.pdf")); fig.savefig(os.path.join(HERE, "figures", "fig_opt.png"), dpi=300)
        plt.close(fig)
    # ------------------------------------------------------------ nested refit
    Q = merge_preds("preds_nested_*.csv")
    if Q is not None:
        old = pd.read_csv(os.path.join(REV, "rev_all_predictions.csv"))
        Dq, Eq = Q[Q.set == "dev"], Q[Q.set == "ext"]
        Do, Eo = old[old.set == "dev"].set_index("case_id").loc[Dq.case_id], old[old.set == "ext"].set_index("case_id").loc[Eq.case_id]
        rows, out = [], []
        oc = boot(Dq.y.to_numpy(), Dq["nested:Clinical"].to_numpy())
        for col in [c for c in Q.columns if c.startswith("nested:") and c != "nested:Clinical"]:
            m = col.split(":", 1)[1]
            a = boot(Dq.y.to_numpy(), Dq[col].to_numpy())
            b = boot(Eq.y.to_numpy(), Eq[col].to_numpy())
            di = stats.delong(Dq.y.to_numpy(), Dq["nested:Clinical"].to_numpy(), Dq[col].to_numpy())
            de = stats.delong(Eq.y.to_numpy(), Eo.Clinical.to_numpy(), Eq[col].to_numpy())
            prev_int = roc_auc_score(Do.y, Do[m]) if m in Do else np.nan
            prev_ext = roc_auc_score(Eo.y, Eo[m]) if m in Eo else np.nan
            out.append({"model": m, "int_auc_nested": a[0], "int_lo": a[1], "int_hi": a[2], "int_auc_as_run": prev_int,
                        "int_diff_vs_clin": di["diff"], "int_diff_lo": di["ci_lo"], "int_diff_hi": di["ci_hi"],
                        "ext_auc_nested": b[0], "ext_lo": b[1], "ext_hi": b[2], "ext_auc_as_run": prev_ext,
                        "ext_diff": de["diff"], "ext_diff_lo": de["ci_lo"], "ext_diff_hi": de["ci_hi"]})
        R = pd.DataFrame(out)
        R.to_csv(os.path.join(OUT, "rev2_nested_summary.csv"), index=False)
        N["nested"] = R.round(4).to_dict(orient="records")
        N["nested_clinical_int"] = oc[0]
        rows = [[r.model, f"{r.int_auc_as_run:.3f}", f"{r.int_auc_nested:.3f} ({r.int_lo:.3f}--{r.int_hi:.3f})", ci(r.int_diff_vs_clin, r.int_diff_lo, r.int_diff_hi),
                 f"{r.ext_auc_as_run:.3f}", f"{r.ext_auc_nested:.3f} ({r.ext_lo:.3f}--{r.ext_hi:.3f})", ci(r.ext_diff, r.ext_diff_lo, r.ext_diff_hi)] for r in R.itertuples()]
        open(os.path.join(HERE, "rev_nested_summary.tex"), "w").write(table(
            "Anchored Models Refitted with Fully Nested Preprocessing (Post Hoc)", "tab:s_nested",
            ["Model", "Int., as run", "Int., nested (95\\% CI)", "Int. $\\Delta$ vs clinical", "Ext., as run", "Ext., nested (95\\% CI)", "Ext. $\\Delta$ vs clinical"],
            rows, "lcccccc",
            f"Internal: I-SPY2 out-of-fold; the nested clinical model's out-of-fold AUC was {oc[0]:.3f}. External $\\Delta$: paired DeLong difference from the clinical model fitted on all of I-SPY2. Training ran on the GPU with the round-2 protocol; small differences from the as-run values also reflect the new training code and random initialization."))
    # ------------------------------------------------------------ overfitting test
    O = read("overfit2_*.csv")
    if O is not None:
        O.to_csv(os.path.join(OUT, "rev2_overfit.csv"), index=False)
        N["overfit"] = O.groupby("lr").agg(final_train=("train_auc", "last"), max_train=("train_auc", "max"), final_held=("heldout_auc", "last")).round(3).reset_index().to_dict(orient="records")
    json.dump(N, open(os.path.join(HERE, "rev2_numbers.json"), "w"), indent=1, default=float)
    print(json.dumps(N, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
