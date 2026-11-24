"""Tables and figures of the restructured manuscript (fourth review), from results/revision4 (rev4_analyses.py first).

    python paper/jbhi/make_rev4_tex.py
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
R4 = os.path.join(ROOT, "results", "revision4")
FIG = os.path.join(HERE, "figures")
BLUE, ORANGE, GREY, INK, GREEN = "#2a78d6", "#eb6834", "#6b7280", "#111827", "#1b9e77"
EXT = ("DUKE", "ISPY1", "NACT")


def f3(x, d=3):
    if pd.isna(x):
        return "--"
    s = f"{x + 0.0:.{d}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return s.replace("-", "$-$")


def ci(a, lo, hi, d=3):
    return f"{f3(a, d)} ({f3(lo, d)} to {f3(hi, d)})"


def C(*w):
    return "".join(f">{{\\centering\\arraybackslash}}p{{{x}cm}}" for x in w)


def table(caption, label, cols, rows, spec, note, size="\\scriptsize"):
    out = ["\\begin{table}[!t]", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\centering", size, "\\setlength{\\tabcolsep}{3pt}",
           f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(cols) + "\\\\", "\\midrule"]
    for r in rows:
        out.append("\\midrule" if r == "MID" else (" & ".join(r) + "\\\\"))
    out += ["\\bottomrule", f"\\multicolumn{{{len(cols)}}}{{p{{0.95\\linewidth}}}}{{\\scriptsize {note}}}\\\\", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(out)


def load():
    M = pd.read_csv(os.path.join(R4, "rev4_locked_matrix.csv"))
    S = pd.read_csv(os.path.join(R4, "rev4_seeded.csv"))
    nl = os.path.join(R4, "rev4_lfnest.csv")
    L = pd.read_csv(nl) if os.path.exists(nl) else None
    return M, S, L


def rep_rows(M, S, L, endpoint):
    """Representative comparisons on one endpoint: list of dicts with the table fields."""
    m = M[M.endpoint == endpoint].set_index("model")
    s = S[S.endpoint == endpoint]
    out = []

    def single(name, key, stage):
        r = m.loc[key]
        out.append({"name": name, "stage": stage, "seeds": "1", "auc": r.auc, "auc_lo": r.auc_lo, "auc_hi": r.auc_hi, "diff": r["diff"], "lo": r.lo, "hi": r.hi,
                    "dmin": np.nan, "dmax": np.nan, "wlo": r.lo, "whi": r.hi, **{f"d_{c}": r[f"diff_{c}"] for c in EXT}, "cal_int": r.cal_int, "cal_slope": r.cal_slope})

    def seeded(name, key, stage, src=None):
        x = (src if src is not None else s)
        x = x[x.model == key].sort_values("seed")
        if not len(x):
            return
        x0 = x[x.seed == 0].iloc[0]
        out.append({"name": name, "stage": stage, "seeds": str(len(x)), "auc": x0.auc, "auc_lo": x0.get("auc_lo", np.nan), "auc_hi": x0.get("auc_hi", np.nan),
                    "diff": x0["diff"], "lo": x0.lo, "hi": x0.hi, "dmin": x["diff"].min(), "dmax": x["diff"].max(), "wlo": x.lo.min(), "whi": x.hi.max(),
                    **{f"d_{c}": x[f"diff_{c}"].mean() for c in EXT}, "cal_int": x.cal_int.mean(), "cal_slope": x.cal_slope.mean()})

    r = m.loc["Clinical"]
    out.append({"name": "Clinical model (age, HR, HER2)", "stage": "ref", "seeds": "--", "auc": r.auc, "auc_lo": r.auc_lo, "auc_hi": r.auc_hi, "diff": np.nan,
                "lo": np.nan, "hi": np.nan, "dmin": np.nan, "dmax": np.nan, "wlo": np.nan, "whi": np.nan, **{f"d_{c}": np.nan for c in EXT},
                **{f"a_{c}": r[f"auc_{c}"] for c in EXT}, "cal_int": r.cal_int, "cal_slope": r.cal_slope})
    single("CARF, 3D (prespecified primary model)", "R1: CARF (proposed)", "c")
    single("Clinical + PETV", "Clinical + FTV", "2")
    seeded("Clinical + kinetic radiomics", "ckin", "3")
    seeded("Image only, pretrained 2.5D", "s25d", "3")
    if L is not None:
        seeded("Late fusion, clinical + 2.5D (nested stacking)", "lfnest", "3", src=L[L.endpoint == endpoint])
    else:
        seeded("Late fusion, clinical + 2.5D", "lf", "3")
    seeded("CARF-mix, pretrained 2.5D", "carfm_25d", "3")
    seeded("CARF-mix, kinetic 3D", "carfm_k", "3")
    return out


def main_table(M, S, L):
    rows = []
    mark = {"ref": "", "c": "$^{\\dagger}$", "2": "$^{\\ddagger}$", "3": "\\textsuperscript{\\S}"}
    for r in rep_rows(M, S, L, "harmonized"):
        if r["stage"] == "ref":
            rows.append([r["name"], r["seeds"], ci(r["auc"], r["auc_lo"], r["auc_hi"]), "reference", "--",
                         *[f"\\emph{{{f3(r[f'a_{c}'])}}}" for c in EXT], f3(r['cal_int'], 2), f3(r['cal_slope'], 2)])
            continue
        rng = "--" if pd.isna(r["dmin"]) else ("identical" if abs(r["dmax"] - r["dmin"]) < 1e-9 else f"{f3(r['dmin'])} to {f3(r['dmax'])}")
        wid = f"{f3(r['wlo'])} to {f3(r['whi'])}"
        rows.append([r["name"] + mark[r["stage"]], r["seeds"], ci(r["auc"], r["auc_lo"], r["auc_hi"]), ci(r["diff"], r["lo"], r["hi"]),
                     rng + (f"; {wid}" if r["seeds"] not in ("1", "--") else ""), f3(r["d_DUKE"]), f3(r["d_ISPY1"]), f3(r["d_NACT"]), f3(r['cal_int'], 2), f3(r['cal_slope'], 2)])
    t = table("External Discrimination and Calibration on the Harmonized Pathologic End Point: Clinical Model and Representative Imaging-Based Models",
              "tab:main", ["Model", "Seeds", "External AUC (95\\% CI)", "$\\Delta$ vs clinical, seed 0 (95\\% CI)", "$\\Delta$ range over seeds; widest 95\\% CI",
                           "Duke $\\Delta$", "I-SPY1 $\\Delta$", "NACT $\\Delta$", "Cal. intercept", "Cal. slope"],
              rows, ">{\\raggedright\\arraybackslash}p{3.3cm}c" + C(2.0, 2.3, 2.6, 0.95, 0.95, 0.95, 0.9, 0.8),
              "509 external patients, 131 with pCR (harmonized end point: no invasive cancer in breast and nodes, in situ disease allowed). "
              "Clinical row: AUC within each collection in italics. $\\Delta$: paired DeLong difference in AUC from the clinical model (95\\% CI, unadjusted). For models with several seeds (outer folds, "
              "inner split and initialization changed), the AUC and $\\Delta$ are for seed 0, the range is over seeds, and the per-collection $\\Delta$ "
              "and calibration are means over seeds. Calibration intercept (calibration-in-the-large) below 0: predicted probabilities too high. "
              "$^{\\dagger}$Prespecified confirmatory comparison (round~1, original inputs); historical analysis on the distributed labels in "
              "Supplementary Table~\\ref{S-tab:s_matrix}. $^{\\ddagger}$Round~2 (exploratory). \\textsuperscript{\\S}Round~3 and later (post hoc), reoriented inputs and nested anchors. "
              "All rows are rescorings of locked predictions; the external patients had been examined in earlier rounds.")
    open(os.path.join(HERE, "rev4_tab_main.tex"), "w").write(t)


def recovery_table():
    R = pd.read_csv(os.path.join(R4, "rev4_recovery_two_draws.csv"))
    Dt = pd.read_csv(os.path.join(R4, "rev4_scalar_detectability.csv")).set_index("level")
    rows = []
    for lev in ["0", "0.02", "0.035", "0.05", "0.10"]:
        for d in (0, 1):
            x = R[(np.isclose(R.level.astype(float), float(lev))) & (R.draw == d)]
            if not len(x):
                continue
            x = x.iloc[0]
            det = Dt[np.isclose(Dt.index.astype(float), float(lev))].iloc[0]
            nom = {"0": "None ($\\beta=0$)"}.get(lev, lev)
            rows.append([nom if d == 0 else "", str(d + 1), ci(x.oracle_same, x.oracle_same_lo, x.oracle_same_hi), ci(x.scalar, x.scalar_lo, x.scalar_hi),
                         ci(x.carfm, x.carfm_lo, x.carfm_hi), ci(x.lf, x.lf_lo, x.lf_hi),
                         (f"{100 * det.detected:.0f}\\%" if d == 0 else "")])
        rows.append("MID")
    rows = rows[:-1]
    t = table("Recovery of a Planted Conditional Imaging Signal: Same-Comparator Oracle, a Model Given the Signal Directly, and the Imaging Pipelines",
              "tab:recovery", ["Planted gain (nominal)", "Draw", "Oracle gain$^{a}$", "Clinical + planted scalar$^{b}$", "CARF-mix 2.5D", "Late fusion$^{c}$",
                               "Scalar model detects gain$^{d}$"],
              rows, ">{\\raggedright\\arraybackslash}p{1.8cm}c" + C(2.6, 2.6, 2.6, 2.6, 1.6),
              "External gain in AUC over the clinical model refitted on the semi-synthetic outcome $y^*\\sim\\mathrm{Bernoulli}\\{\\sigma(z+\\beta s)\\}$ "
              "(95\\% CI), 509--511 external patients. $s$: mean early percentage enhancement in the tumor, linearly residualized on age, HR and HER2 and "
              "standardized. $^{a}$AUC of the true risk $z+\\beta s$ minus that of the refitted clinical model. $^{b}$Logistic regression on age, HR, HER2 "
              "and $s$ itself, fitted on I-SPY2: what is statistically detectable when the signal is given directly. $^{c}$Round-3 stacking (external "
              "predictions use I-SPY2 data only). $^{d}$Proportion of 500 independent outcome draws in which the scalar model's lower 95\\% bound exceeded 0. "
              "The imaging pipelines were run on two draws per level only, so no detection threshold is estimated for them.")
    open(os.path.join(HERE, "rev4_tab_recovery.tex"), "w").write(t)


def supp_matrix(M):
    h = M[M.endpoint == "harmonized"].set_index("model")
    d = M[M.endpoint == "distributed"].set_index("model")
    rows = []
    for m in d.index:
        if m in ("Clinical", "R1: Clinical") or m not in h.index:
            continue
        rows.append([m.replace("Clinical + FTV", "Clinical + PETV").replace("Whole-breast", "Larger-context").replace("R1: ", "Round 1: "),
                     f3(d.loc[m].auc), ci(d.loc[m]["diff"], d.loc[m].lo, d.loc[m].hi), f3(h.loc[m].auc), ci(h.loc[m]["diff"], h.loc[m].lo, h.loc[m].hi),
                     f"{h.loc[m].cal_int:.2f} / {h.loc[m].cal_slope:.2f}"])
    t = table("All Locked Models of Rounds 1--3 Rescored: Distributed Labels (Historical Analysis) and Harmonized Pathologic End Point",
              "tab:s_matrix", ["Model", "AUC, distributed", "$\\Delta$, distributed (95\\% CI)", "AUC, harmonized", "$\\Delta$, harmonized (95\\% CI)",
                               "Calibration intercept / slope, harmonized"],
              rows, ">{\\raggedright\\arraybackslash}p{5.0cm}c" + C(2.8, 1.4, 2.8, 2.2),
              f"Distributed: 511 patients, 124 with pCR; clinical AUC {f3(d.loc['Clinical'].auc)}. Harmonized: 509 patients, 131 with pCR; clinical AUC "
              f"{f3(h.loc['Clinical'].auc)}. Round-1 models are compared with the round-1 clinical model. Same predictions in both columns; only the labels differ. "
              "Models trained on external collections (LOCO, multi-source) cannot be rescored without retraining and are reported in Section~S7 with the "
              "distributed labels.")
    open(os.path.join(HERE, "rev4_supp_matrix.tex"), "w").write(t)


def fig_flow():
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    ax.axis("off")

    def box(x, y, w, h, txt, fc="#f3f4f6", ec=INK, fs=7):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=0.8))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, wrap=True)

    def arr(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=INK))
    box(0.30, 0.86, 0.40, 0.12, "MAMA-MIA: 1,506 patients, four TCIA collections")
    box(0.73, 0.86, 0.26, 0.12, "Excluded: 15 without recorded\noutcome (Duke 11, I-SPY1 4)", fc="white")
    arr(0.70, 0.92, 0.73, 0.92)
    box(0.02, 0.56, 0.36, 0.20, "Development: I-SPY2\n980 patients, 316 pCR\nlabels verified: 980/980 agree", fc="#dbe8f8")
    box(0.44, 0.56, 0.54, 0.20, "External (held out): Duke 280, I-SPY1 167, NACT 64\nDistributed labels: 511 patients, 124 pCR\n(historical prespecified analysis)", fc="#fde8dc")
    arr(0.42, 0.86, 0.20, 0.76); arr(0.58, 0.86, 0.70, 0.76)
    box(0.44, 0.22, 0.54, 0.24, "Source-record audit (post hoc)\nI-SPY1: 167/167 agree\nDuke: 12 with only DCIS/LCIS remaining relabeled pCR\nNACT: label was clinical response; replaced by pathology\n(2 without pathology excluded)", fc="white")
    arr(0.71, 0.56, 0.71, 0.46)
    box(0.44, 0.02, 0.54, 0.13, "Harmonized pathologic end point (headline)\n509 patients, 131 pCR (Duke 76, I-SPY1 49, NACT 6)", fc="#fde8dc")
    arr(0.71, 0.22, 0.71, 0.15)
    box(0.02, 0.10, 0.36, 0.30, "All models fitted on I-SPY2 only\n(fivefold cross-validation;\nseeds 0-4 for post hoc models);\nlocked predictions rescored\non both external end points", fc="white")
    arr(0.20, 0.56, 0.20, 0.40); arr(0.38, 0.25, 0.44, 0.10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_flow4.pdf")); fig.savefig(os.path.join(FIG, "fig_flow4.png"), dpi=300)
    plt.close(fig)


def fig_forest(M, S, L):
    reps = rep_rows(M, S, L, "harmonized")[1:]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5})
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 3.1), sharey=True, gridspec_kw={"width_ratios": [1.25, 1, 1, 1]})
    names = [r["name"].replace(" (prespecified primary model)", "\n(prespecified)").replace(", clinical + 2.5D (nested stacking)", "\n(nested stacking)") for r in reps]
    y = np.arange(len(reps))[::-1]
    for j, (ax, key, title) in enumerate(zip(axes, ["pool", "DUKE", "ISPY1", "NACT"], ["Pooled external", "Duke", "I-SPY1", "NACT"])):
        ax.axvspan(-0.03, 0.03, color="#dbe8f8", alpha=0.7, lw=0)
        ax.axvline(0, color=GREY, lw=0.7)
        for k, r in enumerate(reps):
            if key == "pool":
                ax.errorbar(r["diff"], y[k], xerr=[[r["diff"] - r["lo"]], [r["hi"] - r["diff"]]], fmt="s" if r["stage"] == "c" else "o", ms=3.5,
                            mfc="white" if r["stage"] == "3" else INK, mec=INK, color=INK, elinewidth=0.9)
                if not pd.isna(r["dmin"]):
                    ax.plot([r["dmin"], r["dmax"]], [y[k] - 0.25] * 2, color=BLUE, lw=2.2, solid_capstyle="butt")
            else:
                ax.plot(r[f"d_{key}"], y[k], "o", ms=3.5, mfc="white" if r["stage"] == "3" else INK, mec=INK)
        ax.set_title(title, fontsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_yticks(y); axes[0].set_yticklabels(names, fontsize=7)
    axes[0].set_xlim(-0.30, 0.08)
    for ax in axes[1:]:
        ax.set_xlim(-0.30, 0.15)
    fig.supxlabel("Difference in external AUC from the clinical model (harmonized end point)", fontsize=7.5, y=0.02)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], marker="s", ms=4, color=INK, lw=0, label="Prespecified (round 1)"), Line2D([], [], marker="o", ms=4, color=INK, lw=0, label="Round 2"),
          Line2D([], [], marker="o", ms=4, mfc="white", mec=INK, lw=0, label="Post hoc, seed 0 (95% CI)"), Line2D([], [], color=BLUE, lw=2.2, label="Range over 5 seeds")]
    fig.legend(handles=hs, loc="upper center", ncol=4, fontsize=6.8, frameon=False, bbox_to_anchor=(0.6, 1.0))
    fig.tight_layout(rect=(0, 0.04, 1, 0.93), w_pad=0.5)
    fig.savefig(os.path.join(FIG, "fig_forest4.pdf")); fig.savefig(os.path.join(FIG, "fig_forest4.png"), dpi=300)
    plt.close(fig)


def fig_recovery():
    R = pd.read_csv(os.path.join(R4, "rev4_recovery_two_draws.csv"))
    Dt = pd.read_csv(os.path.join(R4, "rev4_scalar_detectability.csv"))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})
    a = ax[0]
    a.plot([-0.01, 0.11], [-0.01, 0.11], color=GREY, lw=0.8, ls="--")
    a.axhline(0, color=GREY, lw=0.6)
    for col, c, off, lab in (("scalar", GREEN, -0.0015, "Clinical + planted scalar"), ("carfm", BLUE, 0.0, "CARF-mix 2.5D"), ("lf", ORANGE, 0.0015, "Late fusion")):
        a.errorbar(R.oracle_same + off, R[col], yerr=[R[col] - R[f"{col}_lo"], R[f"{col}_hi"] - R[col]], fmt="o", ms=3.8, color=c, ecolor=c, elinewidth=0.9, label=lab)
    a.text(0.075, 0.058, "full recovery", rotation=40, fontsize=6.5, color=GREY)
    a.set_xlabel("Oracle gain (same refitted clinical comparator)")
    a.set_ylabel("Recovered external gain (95% CI)")
    a.set_title("a  Two outcome draws per level", loc="left", fontsize=8.5)
    a.legend(fontsize=6.8, frameon=False, loc="upper left")
    b = ax[1]
    b.plot(Dt.oracle_same_mean, Dt.detected, "-o", color=GREEN, ms=4)
    for x, yv in zip(Dt.oracle_same_mean, Dt.detected):
        b.text(x, yv + 0.05, f"{100 * yv:.0f}%", ha="center", fontsize=6.5, color=GREEN)
    b.set_ylim(-0.03, 1.12); b.set_xlim(-0.005, 0.11)
    b.set_xlabel("Mean oracle gain (500 outcome draws per level)")
    b.set_ylabel("Proportion of draws with gain detected")
    b.set_title("b  Detectability with the signal given directly", loc="left", fontsize=8.5)
    for x in ax:
        for sp in ("top", "right"):
            x.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_recovery4.pdf")); fig.savefig(os.path.join(FIG, "fig_recovery4.png"), dpi=300)
    plt.close(fig)


def cohort_table():
    t = r"""\begin{table}[!t]
\caption{Patients, Outcome Definitions, Treatment and Acquisition by Collection}
\label{tab:cohort}
\centering
\setlength{\tabcolsep}{3.5pt}
\scriptsize
\begin{tabular}{p{3.0cm}p{3.4cm}p{3.4cm}p{3.2cm}p{3.4cm}}
\toprule
 & I-SPY2 (development) & Duke & I-SPY1 & NACT\\
\midrule
Patients & 980 & 280 & 167 & 64 (62 with pathology)\\
pCR, harmonized, $n$ (\%) & 316 (32.2) & 76 (27.1) & 49 (29.3) & 6 (9.7)\\
pCR, distributed label, $n$ (\%) & 316 (32.2) & 64 (22.9) & 49 (29.3) & 11 (17.2)\\
Distributed label, verified against source records & No invasive cancer in breast and nodes (agreed in all) & Complete response in the medical record; only DCIS or LCIS remaining coded as no pCR & No invasive cancer in breast and nodes; in situ disease allowed (agreed in all) & Clinical response ``no evidence of disease'', not pathologic\\
Harmonized end point & As distributed & Complete response, or only DCIS or LCIS remaining (12 relabeled) & As distributed & Pathologic size 0 and no positive node (9 of 62 changed; 2 without pathology excluded)\\
Age, mean (SD), y & 48.8 (10.5) & 48.3 (10.9) & 47.6 (9.0) & 47.7 (9.6)\\
HR-positive / HER2-positive, \% & 54.7 / 24.6 & 60.7 / 28.2 & 55.2 / 32.7 & 62.0 / 31.9\\
Receptors missing, $n$ & 0 & 0 & 5 & 17\\
Study period \cite{garrucho2025data} & 2010--2016 & 2000--2014 & 2002--2006 & 1995--2002\\
Setting & Platform trial, 22 sites & Routine care, one center & Trial, multi-site & Pilot study, one center\\
Neoadjuvant regimen & Paclitaxel $\pm$ investigational agent (79\%) or control (21\%), then anthracycline & Heterogeneous; not recorded in MAMA-MIA & Anthracycline $\pm$ taxane & Anthracycline $\pm$ taxane\\
Field strength 3\,T, \%; plane & 27.0; axial & 53.9; axial & 0; sagittal & 0; sagittal\\
\bottomrule
\multicolumn{5}{p{0.95\linewidth}}{\scriptsize Percentages for HR and HER2 are of patients with recorded status. Source-record extracts and the patient-level comparison are released with the code (Supplementary Table~\ref{S-tab:s_prov}).}\\
\end{tabular}
\end{table}
"""
    open(os.path.join(HERE, "rev4_tab_cohort.tex"), "w").write(t)


def internal_table(S, L):
    R3 = os.path.join(ROOT, "results", "revision3")
    rep = pd.read_csv(os.path.join(R3, "rev3_repeats_lps.csv"))
    tr = pd.read_csv(os.path.join(R3, "rev3_treatment.csv"))
    rng = lambda v: "--" if not len(v) else (f3(min(v)) if abs(max(v) - min(v)) < 5e-4 else f"{f3(min(v))} to {f3(max(v))}")

    def ext(src, key, ep):
        x = src[(src.model == key) & (src.endpoint == ep)]
        return rng(list(x["diff"]))

    rows = []
    for name, key in [("Clinical + kinetic radiomics", "ckin"), ("CARF-mix, kinetic 3D", "carfm_k"), ("CARF-mix, pretrained 2.5D", "carfm_25d"),
                      ("Image only, pretrained 2.5D", "s25d")]:
        o = rep[rep.model == key]
        rows.append([name, str(len(o)), rng(list(o.oof_diff)), ext(S, key, "harmonized"), ext(S, key, "distributed")])
    if L is not None and "oof_diff" in L.columns:
        o = L[(L.endpoint == "harmonized") & (L.model == "lfnest")]
        rows.insert(3, ["Late fusion, clinical + 2.5D (nested stacking)", str(len(o)), rng(list(o.oof_diff.dropna())), ext(L, "lfnest", "harmonized"),
                        ext(L, "lfnest", "distributed")])
    else:
        o = rep[rep.model == "lf"]
        rows.insert(3, ["Late fusion, clinical + 2.5D (stacking not nested)$^{a}$", str(len(o)), rng(list(o.oof_diff)), ext(S, "lf", "harmonized"),
                        ext(S, "lf", "distributed")])
    rows.append("MID")
    g = tr[(tr.analysis == "regimen group held out") & (tr.group == "all")]
    for key, name in [("carfm_25d", "CARF-mix 2.5D"), ("lf", "Late fusion")]:
        if key == "lf" and L is not None and os.path.exists(os.path.join(R4, "rev4_lfnest_regimen.csv")):
            G = pd.read_csv(os.path.join(R4, "rev4_lfnest_regimen.csv"))
            x = G[G.group == "all"].iloc[0]
            name = "Late fusion (nested stacking)"
        else:
            x = g[g.model == key].iloc[0]
            if key == "lf":
                name = "Late fusion (stacking not nested)$^{a}$"
        rows.append([f"{name}: each regimen group held out", "1", ci(x["diff"], x.lo, x.hi), "--", "--"])
    m = tr[tr.analysis.str.startswith("matched training size")]
    c = m[m.group == "ctl"].iloc[0]
    rr = m[m.group.str.startswith("rnd")]
    rows.append(["Late fusion trained on the I-SPY2 control arm ($n=209$)", "1", "--", "--", ci(c["diff"], c.lo, c.hi)])
    rows.append(["Late fusion trained on random all-arm subsets ($n=209$)", str(len(rr)), "--", "--", rng(list(rr["diff"]))])
    t = table("Gains Over the Clinical Model Within the Development Trial and in the External Collections",
              "tab:internal", ["Model or analysis", "Seeds", "Internal $\\Delta$, I-SPY2 out-of-fold", "External $\\Delta$, harmonized", "External $\\Delta$, distributed"],
              rows, ">{\\raggedright\\arraybackslash}p{6.2cm}c" + C(2.9, 2.6, 2.6),
              "$\\Delta$: difference in AUC from the clinical model (range over seeds, or estimate with 95\\% CI). Internal $\\Delta$ uses fully nested "
              "clinical anchors and, for late fusion, nested stacking. Held-out regimen groups: control, HER2-targeted, immunotherapy and other "
              "investigational arms of I-SPY2, each predicted by models trained on the other groups; pooled out-of-group $\\Delta$. Training-size analyses: "
              "external $\\Delta$ on the distributed labels (not rescored). "
              + ("" if (L is not None and "oof_diff" in L.columns) else "$^{a}$Stacker cross-fitted on out-of-fold predictions generated over all of I-SPY2. ")
              + "Group-level results in Supplementary Table~\\ref{S-tab:s_regimen4}.")
    open(os.path.join(HERE, "rev4_tab_internal.tex"), "w").write(t)


NAMES = {"Clinical": "Clinical", "ckin": "Clinical + kinetic radiomics", "s25d": "Image only, 2.5D", "lf": "Late fusion (stacking not nested)",
         "lfnest": "Late fusion (nested stacking)", "carfm_25d": "CARF-mix, 2.5D", "carfm_k": "CARF-mix, kinetic 3D"}


def supp_seeds(S, L):
    A = pd.concat([S, L]) if L is not None else S
    rows = []
    for key in ["ckin", "s25d", "lf", "lfnest", "carfm_25d", "carfm_k"]:
        for sd in sorted(A[A.model == key].seed.unique()):
            h = A[(A.model == key) & (A.seed == sd) & (A.endpoint == "harmonized")]
            d = A[(A.model == key) & (A.seed == sd) & (A.endpoint == "distributed")]
            if not len(h):
                continue
            h, d = h.iloc[0], d.iloc[0]
            rows.append([NAMES[key] if sd == min(A[A.model == key].seed) else "", str(int(sd)), f3(h.auc), ci(h["diff"], h.lo, h.hi),
                         f"{f3(h.cal_int, 2)} / {f3(h.cal_slope, 2)}", f3(d.auc), ci(d["diff"], d.lo, d.hi)])
        rows.append("MID")
    t = table("Seed-Level External Results of the Retrained Models on Both End Points", "tab:s_seeds4",
              ["Model", "Seed", "AUC, harm.", "$\\Delta$, harmonized (95\\% CI)", "Cal. int. / slope, harm.", "AUC, distr.", "$\\Delta$, distributed (95\\% CI)"],
              rows[:-1], ">{\\raggedright\\arraybackslash}p{3.6cm}c" + C(1.2, 3.0, 1.9, 1.2, 3.0),
              "Each seed changed the outer folds, the inner split and the initialization. Clinical model: AUC 0.733 (harmonized; 509 patients, 131 events) and "
              "0.701 (distributed; 511, 124). Clinical + kinetic radiomics is deterministic given the folds; its external predictions do not change with the seed. "
              "Late fusion without nested stacking is shown for comparison with the earlier version.")
    open(os.path.join(HERE, "rev4_supp_seeds.tex"), "w").write(t)


def supp_regimen():
    tr = pd.read_csv(os.path.join(ROOT, "results", "revision3", "rev3_treatment.csv"))
    G = None
    p = os.path.join(R4, "rev4_lfnest_regimen.csv")
    if os.path.exists(p):
        G = pd.read_csv(p)
    lab = {"all": "All", "control": "Control", "her2_targeted": "HER2-targeted", "immunotherapy": "Immunotherapy", "other_experimental": "Other investigational"}
    rows = []
    for an, title in [("within group (5-fold OOF, seed-averaged)", "Within each group"), ("regimen group held out", "Group held out")]:
        x = tr[tr.analysis == an]
        for gk, gl in lab.items():
            c = x[(x.group == gk) & (x.model == "carfm_25d")]
            if not len(c):
                continue
            c = c.iloc[0]
            if G is not None:
                lv = G[(G.analysis == ("within" if an.startswith("within") else "held_out")) & (G.group == gk)]
                lcell = ci(lv.iloc[0]["diff"], lv.iloc[0].lo, lv.iloc[0].hi) if len(lv) else "--"
            else:
                l = x[(x.group == gk) & (x.model == "lf")].iloc[0]
                lcell = ci(l["diff"], l.lo, l.hi) + "$^{a}$"
            rows.append([title if gk == "all" else "", gl, f"{int(c.n)} ({int(c.events)})", ci(c["diff"], c.lo, c.hi), lcell])
        rows.append("MID")
    t = table("Imaging Increment Within I-SPY2 by Regimen Group", "tab:s_regimen4",
              ["Analysis", "Regimen group", "Patients (pCR)", "CARF-mix 2.5D $\\Delta$ (95\\% CI)", ("Late fusion, nested stacking, " if G is not None else "Late fusion, ") + "$\\Delta$ (95\\% CI)"],
              rows[:-1], "llc" + C(3.4, 3.4),
              "Out-of-fold AUC difference from the clinical model fitted under the same split. Within each group: fivefold cross-validation over all of "
              "I-SPY2, evaluated within the group (CARF-mix: predictions averaged over five seeds; late fusion: seeds averaged likewise). Group held out: "
              "models trained on the other groups, pooled over held-out groups in the first row. "
              + ("" if G is not None else "$^{a}$Stacking not nested (earlier version). "))
    open(os.path.join(HERE, "rev4_supp_regimen.tex"), "w").write(t)


def supp_detect():
    Dt = pd.read_csv(os.path.join(R4, "rev4_scalar_detectability.csv"))
    rows = [[{"0": "None"}.get(str(r.level), f"{float(r.level):.3g}"), f"{r.beta:.2f}", f3(r.oracle_same_mean), f3(r.scalar_gain_mean),
             f"{f3(r.scalar_gain_p10)} to {f3(r.scalar_gain_p90)}", f"{100 * r.detected:.1f}\\%"] for r in Dt.itertuples()]
    t = table("Detectability of a Planted Signal Given Directly: 500 Independent Outcome Draws per Level", "tab:s_detect",
              ["Nominal gain", "$\\beta$", "Mean oracle gain", "Mean observed gain", "10th to 90th percentile", "Lower 95\\% bound above 0"],
              rows, "lc" + C(2.2, 2.2, 2.8, 2.6),
              "Model: logistic regression on age, HR, HER2 and the planted scalar $s$, fitted on I-SPY2 for each draw; comparator: clinical model refitted on the "
              "same draw. Oracle gain: AUC of $z+\\beta s$ minus that of the refitted clinical model. Outcome draws used seeds 50000 to 50499.")
    open(os.path.join(HERE, "rev4_supp_detect.tex"), "w").write(t)


if __name__ == "__main__":
    M, S, L = load()
    supp_seeds(S, L)
    supp_regimen()
    supp_detect()
    cohort_table()
    internal_table(S, L)
    main_table(M, S, L)
    recovery_table()
    supp_matrix(M)
    fig_flow()
    fig_forest(M, S, L)
    fig_recovery()
    print("ok")
