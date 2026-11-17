"""Tables and figures of the revision, generated from the result files.

    python paper/jbhi/make_rev_tex.py

Writes paper/jbhi/rev_tab_*.tex (main text), rev_supp_tables.tex (supplement), figures/fig_percoll.*,
figures/fig_checks.* and rev_numbers.json. Rows that depend on the revision run on Kaggle
(results/revision/raw) are written as "pending" until those files exist.
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

R1, R2, REV = (os.path.join(ROOT, "results", d) for d in ("final", "round2", "revision"))
RAW = os.path.join(REV, "raw")
FIG = os.path.join(HERE, "figures")
PLACE = "!t"
BLUE, ORANGE, GREEN, GREY, INK, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#6b7280", "#111827", "#7c5cd6"
NUM = {}


def f2(x):
    return "--" if pd.isna(x) else f"{x:.2f}".replace("-0.00", "0.00").replace("-", "$-$")


def f3(x):
    return "--" if pd.isna(x) else f"{x + 0.0:.3f}".replace("-0.000", "0.000").replace("-", "$-$")


def ci(a, lo, hi, f=f3, dash=False):
    if pd.isna(a):
        return "--"
    if dash:
        return f"{f(a)} ({f(lo)}--{f(hi)})"
    return f"{f(a)} ({f(lo)} to {f(hi)})"


def cic(a, lo, hi):
    return "--" if pd.isna(a) else f"{f3(a)} ({f3(lo)}, {f3(hi)})"


def pval(p):
    if pd.isna(p):
        return "--"
    if p > 0.99:
        return "$>$.99"
    return "$<$.001" if p < 0.001 else (f"{p:.3f}"[1:] if p < 0.01 else f"{p:.2f}"[1:])


def table(caption, label, cols, rows, spec, note=None, size="\\footnotesize", width="0.94\\linewidth"):
    out = [f"\\begin{{table}}[{PLACE}]", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\centering", size,
           "\\setlength{\\tabcolsep}{" + ("2pt" if size == "\\scriptsize" else "3pt") + "}", f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(cols) + "\\\\", "\\midrule"]
    for r in rows:
        out.append("\\midrule" if r == "MID" else " & ".join(r) + "\\\\")
    out.append("\\bottomrule")
    if note:
        out.append(f"\\multicolumn{{{len(cols)}}}{{p{{{width}}}}}{{\\scriptsize {note}}}\\\\")
    out += ["\\end{tabular}", "\\end{table}", ""]
    return "\n".join(out)


def boot_auc(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    i0, i1 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    v = [roc_auc_score(y[b], p[b]) for b in (np.r_[rng.choice(i0, len(i0)), rng.choice(i1, len(i1))] for _ in range(n))]
    return roc_auc_score(y, p), *np.percentile(v, [2.5, 97.5])


def have(name):
    return os.path.exists(os.path.join(RAW, name))


def read_raw(pattern):
    fs = sorted(glob.glob(os.path.join(RAW, pattern)))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else None


# --------------------------------------------------------------------------------------------- inputs
PC = pd.read_csv(os.path.join(REV, "rev_per_collection.csv"))
ALLP = os.path.join(REV, "rev_all_predictions.csv")
P = pd.read_csv(ALLP) if os.path.exists(ALLP) else pd.read_csv(os.path.join(R2, "r2_predictions.csv"))
REVMODELS = ["CARF-mix, kinetic", "CARF-mix, pretrained 2.5D", "DINOv2 linear probe, image only", "DINOv2 linear probe + clinical",
             "Whole-breast 3D ResNet, image only", "Whole-breast 3D ResNet + clinical"]
SHORT = {"CARF-mix, kinetic": "CARF-mix, kinetic 3D", "DINOv2 linear probe, image only": "DINOv2 probe, image only",
         "DINOv2 linear probe + clinical": "DINOv2 probe + clinical", "Whole-breast 3D ResNet, image only": "Whole-breast 3D, image only",
         "Whole-breast 3D ResNet + clinical": "Whole-breast 3D + clinical", "R1: CARF (proposed)": "CARF, 3D (primary)",
         "CARF, kinetic": "CARF, kinetic 3D", "Image only, kinetic 3D ResNet": "Kinetic 3D ResNet, image only",
         "Image only, pretrained 2.5D": "Pretrained 2.5D, image only"}


def sh(m):
    if m in SHORT:
        return SHORT[m]
    if m.startswith("R1: "):
        m = m[4:].replace("Image only (ResNet)", "ResNet, image only").replace("Image only (DenseNet-121)", "DenseNet-121, image only")
        m = m.replace("Image only (EfficientNet-B0)", "EfficientNet-B0, image only").replace("Image only (ViT)", "Vision transformer, image only")
        return m + " (round 1)"
    return m


# --------------------------------------------------------------------------------------------- Table: all models
def tab_all():
    static = r"""Clinical (age, HR, HER2) & 1 & 0.66 (0.62--0.69) & 0.70 (0.65--0.75) & reference & --\\
\textbf{CARF, 3D (primary)} & 1 & 0.65 (0.62--0.69) & \textbf{0.70 (0.65--0.76)} & \textbf{0.004 ($-$0.001 to 0.009)} & .77\\
Clinical + radiomics & 1 & 0.71 (0.68--0.75) & 0.66 (0.61--0.71) & $-$0.042 ($-$0.089 to 0.006) & .51\\
Concatenation fusion & 1 & 0.69 (0.65--0.72) & 0.64 (0.58--0.70) & $-$0.062 ($-$0.118 to $-$0.006) & .23\\
Radiomics & 1 & 0.63 (0.59--0.66) & 0.51 (0.45--0.57) & $-$0.187 ($-$0.273 to $-$0.101) & $<$.001\\
ResNet, image only & 1 & 0.55 (0.51--0.59) & 0.52 (0.47--0.58) & $-$0.178 ($-$0.262 to $-$0.094) & $<$.001\\
DenseNet-121, image only & 1 & 0.55 (0.52--0.59) & 0.53 (0.47--0.59) & $-$0.173 ($-$0.255 to $-$0.090) & $<$.001\\
EfficientNet-B0, image only & 1 & 0.51 (0.47--0.55) & 0.56 (0.51--0.62) & $-$0.141 ($-$0.220 to $-$0.063) & .004\\
Vision transformer, image only & 1 & 0.57 (0.53--0.60) & 0.51 (0.46--0.57) & $-$0.189 ($-$0.272 to $-$0.107) & $<$.001\\
\midrule
Clinical + FTV & 2 & 0.67 (0.63--0.70) & 0.70 (0.65--0.75) & 0.000 ($-$0.018 to 0.019) & .96\\
CARF, kinetic 3D & 2 & 0.65 (0.62--0.69) & 0.70 (0.65--0.75) & 0.001 ($-$0.001 to 0.002) & .78\\
CARF, pretrained 2.5D & 2 & 0.65 (0.61--0.69) & 0.67 (0.62--0.72) & $-$0.030 ($-$0.082 to 0.022) & .78\\
Clinical + kinetic radiomics & 2 & 0.72 (0.69--0.75) & 0.66 (0.61--0.71) & $-$0.040 ($-$0.087 to 0.007) & .37\\
Concatenation fusion, kinetic & 2 & 0.69 (0.65--0.72) & 0.63 (0.57--0.68) & $-$0.075 ($-$0.136 to $-$0.014) & .08\\
Pretrained 2.5D, image only & 2 & 0.61 (0.57--0.65) & 0.56 (0.50--0.62) & $-$0.140 ($-$0.217 to $-$0.064) & .002\\
Kinetic radiomics & 2 & 0.65 (0.61--0.68) & 0.53 (0.47--0.59) & $-$0.173 ($-$0.255 to $-$0.091) & $<$.001\\
Kinetic 3D ResNet, image only & 2 & 0.57 (0.53--0.61) & 0.52 (0.46--0.58) & $-$0.183 ($-$0.265 to $-$0.100) & $<$.001\\
\midrule"""
    D, E = P[P.set == "dev"], P[P.set == "ext"]
    rows, ps, got = [], [], []
    for m in REVMODELS:
        if m in P and E[m].notna().all() and D[m].notna().all():
            a = boot_auc(D.y.to_numpy(), D[m].to_numpy())
            b = boot_auc(E.y.to_numpy(), E[m].to_numpy())
            d = stats.delong(E.y.to_numpy(), E.Clinical.to_numpy(), E[m].to_numpy())
            got.append((m, a, b, d)); ps.append(d["p"])
            NUM[m] = {"int": a[0], "ext": b[0], "diff": d["diff"], "lo": d["ci_lo"], "hi": d["ci_hi"], "p": d["p"]}
    ph = stats.holm(np.array(ps)) if ps else []
    for (m, a, b, d), q in zip(got, ph):
        rows.append(f"{sh(m)} & R & {ci(*a, f=f2, dash=True)} & {ci(*b, f=f2, dash=True)} & {ci(d['diff'], d['ci_lo'], d['ci_hi'])} & {pval(q)}\\\\")
    for m in REVMODELS:
        if m not in [g[0] for g in got]:
            rows.append(f"{sh(m)} & R & pending & pending & pending & --\\\\")
    NUM["n_rev_models"] = len(got)
    body = static + "\n" + "\n".join(rows)
    return ("\\begin{table}[!t]\n\\caption{Internal and External Discrimination of All Models}\n\\label{tab:all}\n\\centering\n\\footnotesize\n"
            "\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{llcccc}\n\\toprule\n"
            "Model & Stage & Internal AUC (95\\% CI) & External AUC (95\\% CI) & External $\\Delta$ vs clinical (95\\% CI) & Holm $P$\\\\\n\\midrule\n"
            + body + "\n\\bottomrule\n"
            "\\multicolumn{6}{p{0.94\\linewidth}}{\\scriptsize Stage 1: round~1 (CARF vs clinical is the confirmatory comparison; the others are secondary). Stage 2: round~2 (exploratory, adaptive). Stage R: revision (exploratory, post hoc). Internal: I-SPY2 fivefold cross-validation ($n=980$). External: Duke, I-SPY1 and NACT ($n=511$; 124 with pCR). $\\Delta$: paired DeLong difference. Confidence intervals are unadjusted; only the $P$ values are Holm-adjusted, within each stage (the stage-1 family also included the CARF seeds and ablations), so an interval excluding 0 can accompany an adjusted $P$ above .05. The unadjusted $P$ for the confirmatory comparison is .15. Exploratory $P$ values are descriptive.}\\\\\n"
            "\\end{tabular}\n\\end{table}\n")


# --------------------------------------------------------------------------------------------- Table: per collection
def tab_percoll():
    order = ["R1: CARF (proposed)", "CARF, kinetic", "CARF, pretrained 2.5D", "CARF-mix, kinetic", "CARF-mix, pretrained 2.5D", "Clinical + FTV",
             "Clinical + kinetic radiomics", "Concatenation fusion, kinetic", "DINOv2 linear probe + clinical", "Whole-breast 3D ResNet + clinical",
             "Image only, pretrained 2.5D", "Image only, kinetic 3D ResNet"]
    c0 = PC[PC.model == PC.model.iloc[0]].set_index("collection")
    rows = [["Clinical, AUC"] + [f3(c0.loc[c, "auc_clinical"]) for c in ("Duke", "I-SPY1", "NACT", "Within-collection pairs")], "MID"]
    for m in order:
        s = PC[PC.model == m].set_index("collection")
        if len(s):
            rows.append([sh(m)] + [ci(s.loc[c, "diff"], s.loc[c, "ci_lo"], s.loc[c, "ci_hi"]) for c in ("Duke", "I-SPY1", "NACT", "Within-collection pairs")])
        elif m.startswith(("R1", "CARF-mix")):
            rows.append([sh(m)] + ["pending"] * 4)
    return table("AUC Difference from the Clinical Model Within Each External Collection (95\\% CI)", "tab:percoll",
                 ["Model", "Duke ($n=280$; 64 pCR)", "I-SPY1 ($n=167$; 49)", "NACT ($n=64$; 11)", "Within-collection pairs"], rows, "lcccc",
                 "Per collection: paired DeLong difference. Within-collection pairs: difference in the collection-stratified AUC, which uses only pairs of patients from the same collection; bootstrap stratified by collection. All models are listed in the Supplementary Material.", size="\\scriptsize")


# --------------------------------------------------------------------------------------------- Table: sensitivity
def tab_sens():
    S = pd.read_csv(os.path.join(REV, "rev_sens_control_arm.csv"))
    C = pd.read_csv(os.path.join(REV, "rev_sens_complete_case.csv"))
    L = pd.read_csv(os.path.join(REV, "rev_leave_one_out.csv"))
    rows = []
    for dname, lab in [("All I-SPY2 arms", "All I-SPY2 arms (main analysis; 980, 316 pCR)"), ("I-SPY2 control arm only", "I-SPY2 control arm only (209, 39 pCR)")]:
        s = S[(S.development == dname) & (S.collection == "All external")].set_index("model")
        c = s.loc["Clinical"]
        rows.append([f"Development on: {lab}", ci(c.auc, c.auc_lo, c.auc_hi, dash=True), f"{c.e_over_o:.2f}; {f2(c.intercept)}; {c.slope:.2f}",
                     ci(s.loc["Clinical + FTV", "diff"], s.loc["Clinical + FTV", "ci_lo"], s.loc["Clinical + FTV", "ci_hi"]),
                     ci(s.loc["Clinical + kinetic radiomics", "diff"], s.loc["Clinical + kinetic radiomics", "ci_lo"], s.loc["Clinical + kinetic radiomics", "ci_hi"])])
    rows.append("MID")

    def pick(df, key, val, m):
        s = df[(df[key] == val) & (df.model == m)]
        return ci(s["diff"].iloc[0], s.ci_lo.iloc[0], s.ci_hi.iloc[0]) if len(s) else "--"

    rows2 = []
    for a, lab in [("Complete cases (receptor status recorded)", "Complete cases ($n=489$; 121 pCR)"),
                   ("Standardised to the I-SPY2 subtype mix", "Reweighted to the I-SPY2 subtype mix")]:
        auc = C[C.analysis == a].auc_clinical.iloc[0]
        rows2.append([lab, f3(auc), pick(C, "analysis", a, "CARF, kinetic"), pick(C, "analysis", a, "Clinical + FTV"), pick(C, "analysis", a, "Clinical + kinetic radiomics")])
    for ex in ("Duke", "I-SPY1", "NACT"):
        n = int(L[L.excluded == ex].n.iloc[0])
        rows2.append([f"External cohort without {ex} ($n={n}$)", "--", pick(L, "excluded", ex, "CARF, kinetic"), pick(L, "excluded", ex, "Clinical + FTV"),
                      pick(L, "excluded", ex, "Clinical + kinetic radiomics")])
    t1 = table("Sensitivity Analyses for Treatment, Missing Data and Case-Mix Differences Between Collections", "tab:sens",
               ["Analysis", "Clinical, external AUC", "E/O; intercept; slope", "$\\Delta$, clinical + FTV", "$\\Delta$, clinical + kinetic radiomics"],
               rows + [["", "Clinical, external AUC", "$\\Delta$, CARF, kinetic 3D", "$\\Delta$, clinical + FTV", "$\\Delta$, clinical + kinetic radiomics"], "MID"] + rows2,
               "p{4.3cm}cccc",
               "E/O: expected-to-observed ratio of pCR. $\\Delta$: external AUC difference from the clinical model fitted on the same development data (95\\% CI). Upper block: tabular models refitted on the stated development set; networks were not retrained. Lower block: models of the main analysis evaluated on the stated subset or weighting of the external cohort.", size="\\scriptsize")
    return t1


# --------------------------------------------------------------------------------------------- Figure: per collection
def fig_percoll():
    order = ["R1: CARF (proposed)", "CARF, kinetic", "CARF-mix, kinetic", "CARF, pretrained 2.5D", "CARF-mix, pretrained 2.5D", "Clinical + FTV",
             "Clinical + kinetic radiomics", "Concatenation fusion, kinetic", "DINOv2 linear probe + clinical", "Whole-breast 3D ResNet + clinical",
             "Image only, pretrained 2.5D", "Image only, kinetic 3D ResNet"]
    order = [m for m in order if (PC.model == m).any()]
    cols = ["Duke", "I-SPY1", "NACT", "Within-collection pairs"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 0.27 * len(order) + 1.0), sharey=True)
    for ax, c in zip(axes, cols):
        s = PC[PC.collection == c].set_index("model").loc[order]
        yy = np.arange(len(order))[::-1]
        ax.axvspan(-0.03, 0.03, color=BLUE, alpha=0.10, lw=0)
        ax.axvline(0, color=GREY, lw=0.7)
        ax.errorbar(s["diff"], yy, xerr=[s["diff"] - s.ci_lo, s.ci_hi - s["diff"]], fmt="o", ms=3, color=INK, ecolor=INK, elinewidth=0.9, capsize=0)
        ax.set_title(c if c != "Within-collection pairs" else "Within-collection pairs", fontsize=7.4)
        ax.set_xlim(-0.48, 0.2); ax.set_xticks([-0.4, -0.2, 0, 0.2])
        ax.tick_params(axis="x", labelsize=6.4)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="x", color="#e5e7eb", lw=0.5)
    axes[0].set_yticks(np.arange(len(order))[::-1]); axes[0].set_yticklabels([sh(m) for m in order], fontsize=6.8)
    fig.supxlabel("AUC difference from the clinical model (95% CI)", fontsize=7.2, y=0.02)
    fig.tight_layout(rect=(0, 0.03, 1, 1), w_pad=0.6)
    fig.savefig(os.path.join(FIG, "fig_percoll.pdf")); fig.savefig(os.path.join(FIG, "fig_percoll.png"), dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------------------------- Supplement
def supp_tables():
    global PLACE
    PLACE = "!htbp"
    parts = []
    # decision rule at three margins
    D = pd.read_csv(os.path.join(REV, "rev_decision_rule.csv"))
    lab = {"equivalent (no clinically important difference)": "equivalent", "no clinically important gain (loss not excluded)": "gain excluded",
           "clinically important loss": "important loss", "inconclusive": "inconclusive", "clinically important gain": "important gain"}
    rows = []
    for m in D.model.unique():
        s = D[D.model == m].set_index("margin")
        r = s.iloc[0]
        rows.append([sh(m), ci(r["diff"], r.ci_lo, r.ci_hi), r.statistical] + [lab.get(s.loc[k, "clinical"], s.loc[k, "clinical"]) for k in (0.02, 0.03, 0.05)])
    parts.append(table("Decision Rule with Separate Statistical and Clinical-Importance Questions, at Three Margins (Pooled External Cohort)", "tab:s_rule",
                       ["Model", "$\\Delta$ (95\\% CI)", "Statistical", "$m=0.02$", "$m=0.03$", "$m=0.05$"], rows, "lccccc",
                       "Statistical: superior or inferior if the 95\\% CI excludes 0; otherwise not shown. Clinical importance: equivalent, CI within $(-m,m)$; gain excluded, upper bound below $m$ (a loss is not excluded); important loss, upper bound at most $-m$."))
    # LOCO weights
    W = pd.read_csv(os.path.join(REV, "rev_loco_weights.csv"))
    rows = []
    for m in W.model.unique():
        for r in W[W.model == m].itertuples():
            if r.held_out in ("Duke", "I-SPY1", "I-SPY2", "NACT"):
                rows.append([sh(m), r.held_out, f"{r.diff:.4f}".replace("-", "$-$"), f"{r.se:.4f}", f"{100 * r.weight_fixed:.1f}", f"{100 * r.weight_random:.1f}"])
            else:
                rows.append([sh(m), r.held_out, f"{r.diff:.4f}".replace("-", "$-$") + ("" if pd.isna(r.ci_lo) else f" ({r.ci_lo:.4f} to {r.ci_hi:.4f})".replace("-0", "$-$0")), "", "", ""])
        rows.append("MID")
    parts.append(table("Leave-One-Collection-Out Validation: Differences, Standard Errors and Pooling Weights", "tab:s_locow",
                       ["Model", "Held out / estimate", "$\\Delta$", "SE", "Fixed weight, \\%", "Random weight, \\%"], rows[:-1], "llcccc",
                       "For kinetic CARF the DeLong standard error is close to zero because its predictions nearly equal the clinical model's; the inverse-variance weights are then degenerate and no pooled estimate is reported in the text.", size="\\scriptsize"))
    # sensitivity: leave-one-out, complete cases, standardised
    L = pd.read_csv(os.path.join(REV, "rev_leave_one_out.csv"))
    C = pd.read_csv(os.path.join(REV, "rev_sens_complete_case.csv"))
    rows, rows_b = [], []
    for m in L.model.unique():
        s = L[L.model == m].set_index("excluded")
        c = C[C.model == m].set_index("analysis")
        rows.append([sh(m)] + [ci(s.loc[k, "diff"], s.loc[k, "ci_lo"], s.loc[k, "ci_hi"]) for k in ("Duke", "I-SPY1", "NACT")])
        rows_b.append([sh(m)] + [ci(c.loc[k, "diff"], c.loc[k, "ci_lo"], c.loc[k, "ci_hi"]) for k in c.index] + [""])
    parts.append(table("Sensitivity of the Pooled External AUC Difference from the Clinical Model", "tab:s_sens",
                       ["Model", "Without Duke ($n=231$)", "Without I-SPY1 ($n=344$)", "Without NACT ($n=447$)"],
                       rows + ["MID", ["", "Complete cases ($n=489$)", "I-SPY2 subtype mix", ""], "MID"] + rows_b, "lccc",
                       "Complete cases: patients with recorded receptor status. I-SPY2 subtype mix: external patients reweighted so that the receptor-subtype distribution equals that of I-SPY2."))
    # decision curves
    DC = pd.read_csv(os.path.join(REV, "rev_decision_curves.csv"))
    rows = []
    for m in DC.model.unique():
        s = DC[DC.model == m].set_index("threshold")
        key = "nb_vs_best_default" if m == "Clinical" else "nb_minus_clinical"
        lo, hi = ("nbd_lo", "nbd_hi") if m == "Clinical" else ("lo", "hi")
        rows.append([("Clinical, vs better default" if m == "Clinical" else sh(m))] + [cic(s.loc[t, key], s.loc[t, lo], s.loc[t, hi]) for t in (0.2, 0.25, 0.3, 0.4)])
        if m == "Clinical":
            rows.append("MID")
    parts.append(table("Decision Curves: Net Benefit of the Clinical Model Against the Better Default Strategy, and Difference of Each Model from the Clinical Model (Pooled External Cohort; Bootstrap 95\\% CI)",
                       "tab:s_dca", ["Model", "Threshold 0.20", "0.25", "0.30", "0.40"], rows, "lcccc",
                       "Action: continue the planned standard regimen if the predicted probability of pCR is at or above the threshold; otherwise consider escalation or trial enrollment. Default strategies: the same management for all patients. Probabilities were not recalibrated.", size="\\scriptsize"))
    # calibration by collection
    K = pd.read_csv(os.path.join(REV, "rev_calibration.csv"))
    K = K[K.set == "External"]
    rows = []
    for m in K.model.unique():
        s = K[K.model == m].set_index("collection")
        rows.append([sh(m)] + [f"{s.loc[c, 'expected_over_observed']:.2f}; {f2(s.loc[c, 'intercept'])}; {f2(s.loc[c, 'slope'])}" for c in ("All", "Duke", "I-SPY1", "NACT")])
    parts.append(table("External Calibration by Collection: Expected-to-Observed Ratio; Calibration Intercept; Slope", "tab:s_calcoll",
                       ["Model", "All external", "Duke", "I-SPY1", "NACT"], rows, "lcccc",
                       "Observed pCR rate: 0.243 overall; 0.229 Duke; 0.293 I-SPY1; 0.172 NACT. A ratio above 1 and a negative intercept indicate overprediction."))
    # size of the correction
    G = pd.read_csv(os.path.join(REV, "rev_gdelta.csv"))
    rows = [[sh(r.model), r.set, str(r.n), f3(r.mean_abs), f3(r.p99_abs), f3(r.max_abs), f2(r.max_abs_prob_change)] for r in G.itertuples()]
    parts.append(table("Size of the Imaging Correction $|g\\delta|$ of the Anchored Models (Logits)", "tab:s_gd",
                       ["Model", "Set", "$n$", "Mean", "99th percentile", "Maximum", "Largest change in probability"], rows, "llccccc",
                       "Computed as the difference between the model's logit and the clinical logit. External values are for the average of the five fold models. CARF-mix is bounded at 1 logit by construction."))
    # control arm, by collection
    S = pd.read_csv(os.path.join(REV, "rev_sens_control_arm.csv"))
    rows = []
    for r in S.itertuples():
        rows.append([r.development.replace("All I-SPY2 arms", "All arms").replace("I-SPY2 control arm only", "Control arm"), r.model, r.collection,
                     ci(r.auc, r.auc_lo, r.auc_hi, dash=True), f"{r.e_over_o:.2f}", f2(r.intercept), f2(r.slope), "--" if pd.isna(r.diff) else ci(r.diff, r.ci_lo, r.ci_hi)])
    parts.append(table("Tabular Models Developed on All I-SPY2 Arms or on the Control Arm Only: External Performance by Collection", "tab:s_ctl",
                       ["Development", "Model", "Collection", "AUC (95\\% CI)", "E/O", "Intercept", "Slope", "$\\Delta$ (95\\% CI)"], rows, "lllccccc",
                       "Control arm: paclitaxel, or paclitaxel plus trastuzumab for HER2-positive disease (209 patients, 39 with pCR). $\\Delta$: difference from the clinical model developed on the same patients.", size="\\scriptsize"))
    # per-collection, all models
    rows = []
    for m in PC.model.unique():
        s = PC[PC.model == m].set_index("collection")
        rows.append([sh(m)] + [cic(s.loc[c, "diff"], s.loc[c, "ci_lo"], s.loc[c, "ci_hi"]) for c in ("Duke", "I-SPY1", "NACT", "Within-collection pairs")])
    parts.append(table("AUC Difference from the Clinical Model Within Each External Collection, All Models with Per-Patient Predictions", "tab:s_percoll",
                       ["Model", "Duke", "I-SPY1", "NACT", "Within pairs"], rows, "lcccc", size="\\scriptsize"))
    # revision run: input dependency, positive controls, learning curves
    f = os.path.join(REV, "rev_checks_dependency.csv")
    if os.path.exists(f):
        Dp = pd.read_csv(f)
        rows = [[sh(r.model), "I-SPY2 test folds" if r.set == "test fold" else "External", f3(r.tab), f3(r.img), f3(r.mean_gd), f3(r.max_gd), f2(r.gate)] for r in Dp.itertuples()]
        parts.append(table("Input Dependency of CARF-mix: Mean Absolute Change of the Output Logit When Inputs Are Permuted Between Patients", "tab:s_dep",
                           ["Model", "Patients", "Tabular permuted", "Images permuted", "Mean $|g\\delta|$", "Max. $|g\\delta|$", "Mean gate"], rows, "llccccc",
                           "Means over the five fold models; maximum over fold models and patients. In CARF (round 1 and round 2), permuting the tabular inputs changes the output by exactly zero by construction."))
    f = os.path.join(REV, "rev_checks_positive_controls.csv")
    if os.path.exists(f):
        Pc = pd.read_csv(f)
        tl = {"pe_mean_high": "Mean enhancement $>$ median", "frac_washout_high": "Washout fraction $>$ median", "vendor_ge": "Scanner vendor (GE)"}
        nl = {"Image only, kinetic 3D ResNet": "Kinetic 3D ResNet", "Image only, pretrained 2.5D": "Pretrained 2.5D"}
        rows = []
        for (m, t), gq in Pc.groupby(["model", "target"], sort=False):
            gq = gq.set_index("cohort")
            a, b = gq.loc["I-SPY2 cross-validation"], gq.loc["External"]
            rows.append([tl[t], nl[m], ci(a.auc, a.lo, a.hi, dash=True), ci(b.auc, b.lo, b.hi, dash=True)]
                        + [f3(gq.loc[c, "auc"]) if c in gq.index else "--" for c in ("DUKE", "ISPY1", "NACT")])
        parts.append(table("Positive Controls: AUC of Networks Trained on Targets with Known Answers, Using the Same Crops and Protocol as for pCR", "tab:s_pos",
                           ["Target", "Network", "I-SPY2 AUC (95\\% CI)", "External AUC (95\\% CI)", "Duke", "I-SPY1", "NACT"], rows, "llccccc",
                           "Median: the I-SPY2 median of the feature. I-SPY2: fivefold cross-validation ($n=980$; 970 for washout fraction, which is undefined in 10 tumors). External: $n=526$, including patients without a recorded pCR. Per-collection AUC is not defined (--) where a collection contains one class only.", size="\\scriptsize"))
    f = os.path.join(REV, "rev_checks_learning_curves.csv")
    if os.path.exists(f):
        Lc = pd.read_csv(f)
        rows = [[sh(r.model), f"{100 * r.fraction:.0f}", "--" if pd.isna(r.n_train) else f"{r.n_train:.0f}", "--" if pd.isna(r.train_auc_mean) else f"{r.train_auc_mean:.2f} ({r.train_auc_min:.2f}--{r.train_auc_max:.2f})",
                 ci(r.dev_auc, r.dev_lo, r.dev_hi, dash=True), ci(r.ext_auc, r.ext_lo, r.ext_hi, dash=True)] for r in Lc.itertuples()]
        parts.append(table("Learning Curves for pCR: AUC by Share of Each Training Fold Used", "tab:s_lc",
                           ["Network", "Share, \\%", "Training $n$", "Training AUC (range)", "Held-out AUC (95\\% CI)", "External AUC (95\\% CI)"], rows, "lccccc",
                           "Training AUC: mean and range over the five folds, computed on the patients used for fitting, at the weights kept by early stopping. The 100\\% rows are the round-2 models; their training AUC was not recorded."))
    # second review: nesting audit and clinical specifications
    f = os.path.join(REV, "rev_audit_nesting.csv")
    if os.path.exists(f):
        A = pd.read_csv(f)
        rows = [[str(int(r.fold)), str(int(r.n_train)), f3(r.mean_abs_diff), f3(r.p99_abs_diff), f2(r.max_abs_diff), f3(r.corr), f2(r.sd_anchor)] for r in A.itertuples()]
        parts.append(table("Anchors as Used Versus Fully Nested Anchors for the Network-Training Patients of Each Outer Fold (Logits)", "tab:s_nest",
                           ["Outer fold", "Training patients", "Mean $|$difference$|$", "99th percentile", "Maximum", "Correlation", "SD of anchor"], rows, "ccccccc",
                           "As used: anchors cross-fitted once over all of I-SPY2 with the outer folds. Nested: cross-fitted within the network-training patients of the outer fold (excluding the test fold and the early-stopping patients). The largest differences occur in the three I-SPY2 patients with missing age, whose missing-value indicator is estimated from very few patients."))
    f = os.path.join(REV, "rev_clinical_candidates.csv")
    if os.path.exists(f):
        Cc = pd.read_csv(f)
        rows = [[r.description.replace(" x ", " $\\times$ "), str(int(r.n_terms)), f"{r.cv_auc_mean:.3f} ({r.cv_auc_sd_over_repeats:.3f})", f3(r.ext_auc),
                 ci(r.diff_vs_main, r.ci_lo, r.ci_hi), f"{r.e_over_o:.2f}; {f2(r.intercept)}; {f2(r.slope)}", "yes" if r.selected else ""] for r in Cc.itertuples()]
        parts.append(table("Clinical Specifications Compared on I-SPY2 Data (Post Hoc)", "tab:s_clin",
                           ["Specification", "Terms", "I-SPY2 CV AUC (SD)", "External AUC", "$\\Delta$ vs main (95\\% CI)", "E/O; intercept; slope", "Selected"], rows, "lcccccc",
                           "CV AUC: mean over 20 repeats of fivefold cross-validation; SD over repeats. Spline: natural cubic spline of age with knots at the 10th, 50th and 90th I-SPY2 percentiles. Selection was by CV AUC only.", size="\\scriptsize"))
    for fn, cap, lab in (("rev_opt_summary.tex", None, None), ("rev_nested_summary.tex", None, None)):
        f = os.path.join(HERE, fn)
        if os.path.exists(f):
            parts.append(open(f).read())
    return "\n".join(parts)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    for name, fn in [("rev_tab_all.tex", tab_all), ("rev_tab_percoll.tex", tab_percoll), ("rev_tab_sens.tex", tab_sens), ("rev_supp_tables.tex", supp_tables)]:
        open(os.path.join(HERE, name), "w").write(fn())
        print("wrote", name)
    fig_percoll()
    json.dump(NUM, open(os.path.join(HERE, "rev_numbers.json"), "w"), indent=1, default=float)
