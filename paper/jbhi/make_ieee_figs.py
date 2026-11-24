"""Figures drawn at IEEE print size (column 3.5 in, page 7.16 in) for the JBHI version.

    python paper/jbhi/make_ieee_figs.py
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "figures")
sys.path.insert(0, os.path.join(ROOT, "paper"))
import make_round2_figures as m  # noqa: E402

BLUE, ORANGE, GREY, INK, INK2 = "#2a78d6", "#eb6834", "#6b7280", "#111827", "#374151"
COL = 3.5

# ------------------------------------------------------------------ round-2 figures at print size
SIZES = {"fig_forest_all": (COL, 4.45), "fig_controls": (COL, 2.35), "fig_loco": (COL, 2.5), "fig_samples": (7.16, 4.05)}
_orig_subplots = plt.subplots
_current = {"name": None}


def _subplots(*a, **k):
    if _current["name"] in SIZES:
        k["figsize"] = SIZES[_current["name"]]
    return _orig_subplots(*a, **k)


def _save(fig, name):
    ax = fig.axes[0]
    if name == "fig_forest_all":
        ax.set_yticklabels([t.get_text().replace("Clinical + FTV", "Clinical + PETV") for t in ax.get_yticklabels()])
        ax.set_xlabel("External AUC difference vs clinical model (95% CI)", fontsize=7)
        ax.tick_params(axis="y", labelsize=6.6)
        ax.tick_params(axis="x", labelsize=6.5)
        for t in ax.texts:
            t.set_fontsize(min(t.get_fontsize(), 6.6))
    if name == "fig_controls":
        ax.set_xlabel("AUC (95% CI), kinetic 3D network, same tumor crops", fontsize=7)
        short = {"Hormone receptor status": "HR status", "HER2 status": "HER2 status",
                 "Field strength (3 T vs 1.5 T)": "Field strength", "pCR (the study target)": "pCR (study target)",
                 "Shuffled pCR labels (negative control)": "Shuffled labels"}
        ax.set_yticklabels([short.get(t.get_text(), t.get_text()) for t in ax.get_yticklabels()], fontsize=7)
        ax.tick_params(axis="x", labelsize=6.5)
        ax.set_xlim(0.38, 0.72)
        for t in ax.texts:
            t.set_fontsize(6.2)
        leg = ax.get_legend()
        if leg:
            h, l = leg.legend_handles, [t.get_text() for t in leg.get_texts()]
            leg.remove()
            ax.legend(h, l, fontsize=6.3, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    if name == "fig_loco":
        labs = []
        for t in ax.get_xticklabels():
            s = t.get_text()
            if "held out" in s:
                parts = s.split("\n")
                labs.append(parts[0].replace(" held out", "") + "\n" + parts[1].replace("n = ", "n=") + "\nAUC " + parts[2].split()[-1])
            else:
                labs.append("Pooled")
        ax.set_xticklabels(labs, fontsize=6.3)
        ax.set_ylabel("AUC difference vs clinical\n(held-out collection, 95% CI)", fontsize=6.8)
        ax.tick_params(axis="y", labelsize=6.5)
        leg = ax.get_legend()
        if leg:
            leg.remove()
        h, l = ax.get_legend_handles_labels()
        ax.legend(h, ["Clin. + kinetic radiomics" if "radiomics" in x else ("Kinetic 3D, image only" if "Image" in x else x) for x in l],
                  fontsize=6.2, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handletextpad=0.2, columnspacing=0.8)
    if name == "fig_samples":
        for a in fig.axes:
            a.title.set_fontsize(6.6)
            a.yaxis.label.set_fontsize(7)
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)
    plt.close(fig)


m.plt.subplots = _subplots
m.save = _save


def run(name, fn, *a):
    _current["name"] = name
    fn(*a)
    _current["name"] = None


# ------------------------------------------------------------------ patient flow (column)
def fig_flow():
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, ax = _orig_subplots(figsize=(COL, 2.85))
    ax.set_xlim(0, 100); ax.set_ylim(0, 82); ax.axis("off")

    def box(x, y, w, h, title, body, c=BLUE, fs=6.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5", lw=0.9, ec=c, fc=c + "12"))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", weight="bold", fontsize=fs + 0.4, color=INK)
        ax.text(x + w / 2, y + h - 8.6, body, ha="center", va="top", fontsize=fs, color=INK2, linespacing=1.3)

    def arrow(x0, y0, x1, y1, c=GREY):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.8, color=c))

    box(4, 63, 54, 18, "MAMA-MIA dataset", "Synapse syn60868042\n1,506 patients, 4 collections")
    box(64, 63, 35, 18, "Excluded: 15", "no pCR status\n(Duke 11, I-SPY1 4)", GREY, fs=6.2)
    arrow(58.6, 72, 63.4, 72)
    arrow(31, 62.4, 33, 59.6)
    box(8, 45, 54, 14, "Included: 1,491 women", "440 with pCR (29.5%)")
    arrow(24, 44.4, 22, 38.6); arrow(48, 44.4, 76, 38.6)
    box(1, 1, 45, 37, "Development: I-SPY2", "980 patients (316 pCR)\n22 sites\nfivefold cross-validation\nmodels and thresholds\nfrozen before scoring")
    box(54, 1, 45, 37, "External test", "Duke 280 (64 pCR)\nI-SPY1 167 (49 pCR)\nNACT 64 (11 pCR)\n511 patients, 124 pCR", ORANGE)
    arrow(46.6, 19, 53.4, 19, ORANGE)
    fig.savefig(os.path.join(OUT, "fig1_workflow.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig1_workflow.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import pandas as pd

    os.makedirs(OUT, exist_ok=True)
    r2dir = os.path.join(ROOT, "results", "round2")
    run("fig_samples", m.fig_samples, os.path.join(r2dir, "raw", "samples.npz"))
    r1 = pd.read_csv(os.path.join(ROOT, "results", "final", "table5_vs_clinical.csv"))
    r1["model"] = r1.model.replace({"CARF (proposed)": "CARF", "Image only (ResNet)": "Image only, ResNet",
                                    "Image only (DenseNet-121)": "Image only, DenseNet-121",
                                    "Image only (EfficientNet-B0)": "Image only, EfficientNet-B0",
                                    "Image only (ViT)": "Image only, vision transformer"})
    r2 = {"perf": pd.read_csv(os.path.join(r2dir, "r2_performance.csv")), "cmp": pd.read_csv(os.path.join(r2dir, "r2_vs_clinical.csv"))}
    run("fig_forest_all", m.fig_forest, r1, r2)
    run("fig_controls", m.fig_controls, pd.read_csv(os.path.join(r2dir, "r2_controls.csv")), r2["perf"].set_index("model"))
    run("fig_loco", m.fig_loco, pd.read_csv(os.path.join(r2dir, "r2_loco.csv")), pd.read_csv(os.path.join(r2dir, "r2_loco_pooled.csv")))
    fig_flow()
