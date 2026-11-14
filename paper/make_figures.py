"""Build the manuscript figures from the exported results (results/final).

    python paper/make_figures.py
"""
import io
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "final")
OUT = os.path.join(ROOT, "paper", "figures")
os.makedirs(OUT, exist_ok=True)

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C = {"CARF (proposed)": "#2a78d6", "Clinical": "#eb6834", "Clinical + radiomics": "#1baf7a",
     "Concatenation fusion": "#eda100", "Image only (ResNet)": "#e87ba4"}
LS = {"CARF (proposed)": "-", "Clinical": (0, (4, 2)), "Clinical + radiomics": (0, (1, 1.5)),
      "Concatenation fusion": "-.", "Image only (ResNet)": (0, (6, 2, 1, 2))}
SHOWN = ["CARF (proposed)", "Clinical", "Clinical + radiomics", "Concatenation fusion", "Image only (ResNet)"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.bbox": "tight", "savefig.dpi": 300, "legend.frameon": False})


def tables():
    txt = open(os.path.join(RES, "_export_tables.txt")).read()
    out = {}
    for block in txt.split("### ")[1:]:
        name, body = block.split("\n", 1)
        out[name.strip().replace(".csv", "")] = pd.read_csv(io.StringIO(body.strip()))
    for k, v in out.items():
        v.to_csv(os.path.join(RES, k + ".csv"), index=False)
    return out


def curves():
    return [json.loads(l) for l in open(os.path.join(RES, "_export_curves_external.jsonl")) if l.strip()]


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"))
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)


def fig_workflow():
    steps = [("MAMA-MIA", "1,506 patients\n4 US collections"), ("Preprocess", "1 mm resampling\ntumour crop 64³\n20 radiomic\nfeatures"),
             ("Develop", "I-SPY2, n = 980\n5-fold CV\n9 models +\n5 CARF variants"),
             ("Freeze", "weights and\nthresholds\ncommitted"), ("External test", "DUKE, I-SPY1,\nNACT; n = 511\nscored once"),
             ("Evaluate", "AUC, calibration\ndecision curves\nsites, subgroups")]
    fig, ax = plt.subplots(figsize=(7.2, 1.7))
    ax.axis("off")
    w = 1 / len(steps)
    for i, (t, s) in enumerate(steps):
        x = i * w + 0.01
        ax.add_patch(plt.Rectangle((x, 0.1), w - 0.03, 0.8, transform=ax.transAxes, fc="#eef4fc" if i != 4 else "#fdeee7",
                                   ec="#2a78d6" if i != 4 else "#eb6834", lw=0.8))
        ax.text(x + (w - 0.03) / 2, 0.78, t, transform=ax.transAxes, ha="center", va="center", fontsize=7, weight="bold", color=INK)
        ax.text(x + (w - 0.03) / 2, 0.42, s, transform=ax.transAxes, ha="center", va="center", fontsize=5.6, color=INK2, linespacing=1.3)
        if i:
            ax.annotate("", xy=(x, 0.5), xytext=(x - 0.02, 0.5), xycoords="axes fraction", arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8))
    save(fig, "fig1_workflow")


def fig_auc_dot(T):
    """Internal vs external AUC per model: the generalisation gap at a glance."""
    t2 = T["table2_internal"].set_index("model")
    t3 = T["table3_external"]
    t3 = t3[t3.cohort == "Pooled external"].set_index("model")
    order = ["Clinical", "Clinical + radiomics", "Radiomics", "Concatenation fusion", "CARF (proposed)",
             "Image only (ResNet)", "Image only (DenseNet-121)", "Image only (EfficientNet-B0)", "Image only (ViT)"]
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    y = np.arange(len(order))[::-1]
    for yi, m in zip(y, order):
        a, b = t2.loc[m], t3.loc[m]
        ax.plot([a.auc, b.auc], [yi + 0.12, yi - 0.12], color=GRID, lw=1.2, zorder=1)
        ax.errorbar(a.auc, yi + 0.12, xerr=[[a.auc - a.auc_lo], [a.auc_hi - a.auc]], fmt="o", ms=4, color="#86b6ef",
                    mec="white", mew=0.6, elinewidth=1, capsize=0, zorder=2, label="Internal (I-SPY2, out-of-fold)" if yi == y[0] else None)
        ax.errorbar(b.auc, yi - 0.12, xerr=[[b.auc - b.auc_lo], [b.auc_hi - b.auc]], fmt="s", ms=4, color="#1c5cab",
                    mec="white", mew=0.6, elinewidth=1, capsize=0, zorder=3, label="External (pooled, n = 511)" if yi == y[0] else None)
    ax.axvline(0.5, color=INK2, lw=0.7, ls=":")
    ax.axvline(t3.loc["Clinical"].auc, color="#eb6834", lw=0.8, ls=(0, (4, 2)))
    ax.text(t3.loc["Clinical"].auc + 0.004, y[-1] - 0.75, "clinical model,\nexternal", fontsize=6, color=INK2, va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels([m.replace(" (proposed)", "") for m in order], fontsize=7)
    ax.set_xlim(0.42, 0.82)
    ax.set_ylim(-1, len(order))
    ax.set_xlabel("AUC (95% CI)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, fontsize=6.5, handletextpad=0.3)
    save(fig, "fig2_auc_internal_external")


def fig_roc_cal_dca(CV):
    cv = {c["model"]: c for c in CV if c.get("kind") == "curves"}
    allc = next(c for c in CV if c.get("kind") == "dca_all")
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for m in SHOWN:
        c = cv[m]
        lab = m.replace(" (proposed)", "")
        ax[0].plot(*c["roc"], color=C[m], ls=LS[m], lw=1.4, label=lab)
        ax[1].plot(*c["cal"], color=C[m], ls=LS[m], lw=1.2, marker="o", ms=2.5, label=lab)
        ax[2].plot(*c["dca"], color=C[m], ls=LS[m], lw=1.3, label=lab)
    ax[0].plot([0, 1], [0, 1], color=INK2, lw=0.6, ls=":")
    ax[0].set(xlabel="1 − specificity", ylabel="Sensitivity", title="a  ROC, pooled external", xlim=(0, 1), ylim=(0, 1))
    ax[1].plot([0, 0.8], [0, 0.8], color=INK2, lw=0.6, ls=":")
    ax[1].set(xlabel="Predicted probability of pCR", ylabel="Observed pCR rate", title="b  Calibration", xlim=(0, 0.8), ylim=(0, 0.8))
    ax[2].plot(allc["t"], allc["nb"], color=INK, lw=0.9, label="Treat all")
    ax[2].axhline(0, color=INK2, lw=0.7, label="Treat none")
    ax[2].set(xlabel="Threshold probability", ylabel="Net benefit", title="c  Decision curve", ylim=(-0.12, 0.18), xlim=(0.1, 0.6))
    for a in ax:
        a.title.set_fontsize(8)
        a.title.set_ha("left")
        a.title.set_x(0)
    h, l = ax[2].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, fontsize=6.5, bbox_to_anchor=(0.5, -0.1), handlelength=2.6)
    fig.tight_layout()
    save(fig, "fig3_roc_calibration_dca")


def fig_ablation(T):
    t5 = T["table5_vs_clinical"].set_index("model")
    rows = ["CARF (proposed)", "CARF (proposed) seed 1", "CARF (proposed) seed 2", "CARF without gate",
            "CARF without timing embedding", "CARF without auxiliary loss", "Concatenation fusion", "Clinical + radiomics",
            "Image only (EfficientNet-B0)"]
    lab = ["CARF (seed 0)", "CARF (seed 1)", "CARF (seed 2)", "− gate", "− timing embedding", "− auxiliary loss",
           "Concatenation fusion", "Clinical + radiomics", "Best image-only (EfficientNet-B0)"]
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    y = np.arange(len(rows))[::-1]
    ax.axvspan(-0.03, 0.03, color="#eef4fc", zorder=0)
    ax.text(0.0, len(rows) - 0.25, "equivalence margin ±0.03", ha="center", va="center", fontsize=6, color=INK2, bbox=dict(fc="white", ec="none", pad=1))
    for yi, r in zip(y, rows):
        d = t5.loc[r]
        col = "#2a78d6" if r.startswith("CARF") else "#52514e"
        ax.errorbar(d["diff"], yi, xerr=[[d["diff"] - d.ci_lo], [d.ci_hi - d["diff"]]], fmt="o", ms=4, color=col,
                    mec="white", mew=0.6, elinewidth=1.1, capsize=0)
        ax.text(0.075, yi, (f"{d['diff']:+.3f}" if abs(d['diff']) >= 0.0005 else "0.000"), va="center", fontsize=6.3, color=INK2)
    ax.axvline(0, color=INK, lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(lab, fontsize=7)
    ax.set_xlim(-0.24, 0.1)
    ax.set_ylim(-0.8, len(rows) + 0.1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Δ AUC vs clinical model, pooled external (95% CI, DeLong)")
    save(fig, "fig4_delta_auc_ablation")


def fig_subgroups(T):
    s = T["tableS_subgroups"]
    keys = list(dict.fromkeys(zip(s.variable, s.level)))
    nice = {"age_band": "Age", "ethnicity": "Ethnicity", "vendor": "Vendor", "field_strength": "Field strength",
            "tumor_subtype": "Subtype", "view": "Plane"}
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    for j, (m, col, mk) in enumerate([("Clinical", "#eb6834", "s"), ("CARF (proposed)", "#2a78d6", "o")]):
        ss = s[s.model == m].set_index(["variable", "level"])
        for i, k in enumerate(keys):
            if k in ss.index:
                r = ss.loc[k]
                yy = len(keys) - 1 - i + (0.17 if j == 0 else -0.17)
                ax.errorbar(r.auc, yy, xerr=[[r.auc - r.lo], [r.hi - r.auc]], fmt=mk, ms=3.5, color=col, mec="white",
                            mew=0.5, elinewidth=0.9, label=m.replace(" (proposed)", "") if i == 0 else None)
    ax.axvline(0.5, color=INK2, lw=0.7, ls=":")
    ax.set_yticks(range(len(keys)))
    n = s.drop_duplicates(["variable", "level"]).set_index(["variable", "level"]).n
    ax.set_yticklabels([f"{nice[v]}: {l}  (n={n.loc[(v, l)]})" for v, l in keys][::-1], fontsize=6.5)
    ax.set_xlabel("External AUC (95% CI)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7)
    save(fig, "fig5_subgroups")


def fig_sites(T):
    s = T.get("tableS_site_auc")
    if s is None:
        return
    p = T["tableS_site_pooled"].set_index("model")
    c = s[s.model == "CARF (proposed)"].sort_values("auc")
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    y = np.arange(len(c))
    ax.errorbar(c.auc, y, xerr=1.96 * c.se, fmt="o", ms=3, color="#2a78d6", elinewidth=0.8, mec="white", mew=0.5)
    r = p.loc["CARF (proposed)"]
    ax.axvspan(r.lo, r.hi, color="#cde2fb", zorder=0)
    ax.axvline(r.pooled_auc, color="#1c5cab", lw=1)
    ax.axvline(0.5, color=INK2, lw=0.6, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{a} (n={n})" for a, n in zip(c.site, c.n)], fontsize=6)
    ax.set_xlabel("I-SPY2 site AUC, out-of-fold (95% CI)")
    ax.set_title(f"Pooled {r.pooled_auc:.2f} ({r.lo:.2f}–{r.hi:.2f}); I² = {100*r.i2:.0f}%", fontsize=7, loc="left")
    ax.grid(axis="y", visible=False)
    save(fig, "figS1_sites")


def fig_gate(CV):
    g = next(c for c in CV if c.get("kind") == "gate_hist")
    edges = np.linspace(0, 0.4, 21)
    fig, ax = plt.subplots(figsize=(3.6, 2.2))
    cols = {"ISPY2": "#2a78d6", "DUKE": "#eb6834", "ISPY1": "#1baf7a", "NACT": "#eda100"}
    w = (edges[1] - edges[0]) / 4.6
    for j, (k, col) in enumerate(cols.items()):
        h = np.array(g[k], float)
        ax.bar(edges[:-1] + j * w + w / 2, h / h.sum(), width=w * 0.9, color=col, label=k, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("CARF imaging gate g (0 = clinical model only)")
    ax.set_ylabel("Share of patients")
    ax.legend(fontsize=6.5, ncol=4, loc="upper right")
    ax.grid(axis="x", visible=False)
    save(fig, "figS2_gate")


if __name__ == "__main__":
    T = tables()
    CV = curves()
    fig_workflow()
    fig_auc_dot(T)
    fig_roc_cal_dca(CV)
    fig_ablation(T)
    fig_subgroups(T)
    fig_gate(CV)
    print(sorted(os.listdir(OUT)))
