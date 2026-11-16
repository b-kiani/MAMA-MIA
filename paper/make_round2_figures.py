"""Figures for the round-2 analyses.

    python paper/make_round2_figures.py [--r2 results/round2] [--samples results/round2/raw/samples.npz]

Writes to paper/figures:
  fig_samples          real example cases (one pCR and one non-pCR per collection)
  fig_forest_all       external AUC difference from the clinical model, all models of both rounds
  fig_controls         what the same image network can and cannot learn from the same crops
  fig_loco             leave-one-collection-out AUC difference from the clinical model
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "paper", "figures")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREEN, PINK, YELLOW, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#eda100", "#8a8984"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "savefig.bbox": "tight", "savefig.dpi": 300, "legend.frameon": False})
COLL = {"DUKE": "Duke", "ISPY1": "I-SPY1", "ISPY2": "I-SPY2", "NACT": "NACT"}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"))
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)


def kind(m):
    m = m.lower()
    if "carf" in m:
        return BLUE
    if "image only" in m:
        return PINK
    if "concat" in m:
        return YELLOW
    if "radiomic" in m or "ftv" in m or "size" in m:
        return GREEN
    return GREY


# ----------------------------------------------------------------------------- real example cases
def fig_samples(path):
    z = np.load(path, allow_pickle=False)
    img, msk, meta = z["img"].astype(np.float32), z["msk"], [m.split(",") for m in z["meta"]]
    cases = {}
    for k, m in enumerate(meta):
        cases[(m[1], int(m[2]))] = k
    cols = [c for c in ("ISPY2", "DUKE", "ISPY1", "NACT") if (c, 1) in cases and (c, 0) in cases]
    panels = [("Pre-contrast", lambda x: x[0], "gray", (0, 2)), ("Early post-contrast", lambda x: x[1], "gray", (0, 4)),
              ("Late post-contrast", lambda x: x[2], "gray", (0, 4)), ("Enhancement (early / pre − 1)", lambda x: x[1] - x[0], "magma", (0, 2.5))]
    order = [(c, y) for c in cols for y in (1, 0)]
    fig, axs = plt.subplots(4, len(order), figsize=(7.4, 4.35), gridspec_kw={"wspace": 0.04, "hspace": 0.05})
    for col, (c, y) in enumerate(order):
        k = cases[(c, y)]
        x, m = img[k], msk[k] > 0
        for row, (title, f, cmap, (lo, hi)) in enumerate(panels):
            ax = axs[row, col]
            ax.imshow(f(x), cmap=cmap, vmin=lo, vmax=hi, interpolation="bilinear")
            ax.contour(m, levels=[0.5], colors=["#22d3ee"], linewidths=0.6)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if col == 0:
                ax.set_ylabel(title.replace(" (", "\n("), fontsize=7, rotation=0, ha="right", va="center", labelpad=4, color=INK)
        sub = ("HR+" if meta[k][3] == "1" else "HR−") + ("/HER2+" if meta[k][4] == "1" else "/HER2−")
        axs[0, col].set_title(f"{'pCR' if y else 'No pCR'}\n{sub}\n{meta[k][5]} y", fontsize=6.4, color=INK, pad=3)
    for j2, c in enumerate(cols):
        a, b = axs[0, 2 * j2].get_position(), axs[0, 2 * j2 + 1].get_position()
        fig.text((a.x0 + b.x1) / 2, a.y1 + 0.105, COLL[c], ha="center", fontsize=8, weight="bold", color=INK)
        fig.add_artist(plt.Line2D([a.x0 + 0.004, b.x1 - 0.004], [a.y1 + 0.098] * 2, color=INK2, lw=0.6))
    fig.text(0.5, 0.06, "Cyan outline: expert tumour segmentation. One slice through the tumour in the acquisition plane per case, chosen as the pCR and the non-pCR "
             "case of median tumour volume in each collection.\nSignal is relative to the mean pre-contrast signal of the crop. "
             "Data: MAMA-MIA (Synapse syn60868042), CC BY-NC 4.0.", ha="center", va="top", fontsize=6.2, color=INK2)
    save(fig, "fig_samples")


# ----------------------------------------------------------------------------- forest of all models
def fig_forest(r1, r2):
    rows = []
    for t in r1.itertuples():
        rows.append(("Round 1", t.model, t.auc, t.diff, t.ci_lo, t.ci_hi))
    perf = r2["perf"].set_index("model")
    for t in r2["cmp"].itertuples():
        rows.append(("Round 2", t.model, perf.loc[t.model, "external_auc"], t.diff, t.ci_lo, t.ci_hi))
    d = pd.DataFrame(rows, columns=["round", "model", "auc", "diff", "lo", "hi"])
    d["model"] = d.model.str.replace("CARF (proposed) ", "CARF, ", regex=False)
    groups = [("Round 1 (prespecified)", d[d["round"] == "Round 1"].sort_values("diff", ascending=False)),
              ("Round 2 (secondary, post hoc)", d[d["round"] == "Round 2"].sort_values("diff", ascending=False))]
    ys, labels, heads, yy = [], [], [], 0.0
    rows_ = []
    for name, g in groups:
        heads.append((name, yy))
        yy -= 1.0
        for t in g.itertuples():
            rows_.append((yy, t)); labels.append(t.model); ys.append(yy)
            yy -= 1.0
        yy -= 0.6
    fig, ax = plt.subplots(figsize=(7.0, 0.26 * (len(rows_) + 4) + 0.8))
    ax.axvspan(-0.03, 0.03, color="#e8f0fb", zorder=0)
    ax.axvline(0, color=INK2, lw=0.8, zorder=1)
    lo_lim, hi_lim = -0.33, 0.16
    for yi, t in rows_:
        a, b = max(t.lo, lo_lim), min(t.hi, hi_lim)
        ax.plot([a, b], [yi, yi], color=kind(t.model), lw=1.6, solid_capstyle="round", zorder=2)
        if t.lo < lo_lim:
            ax.plot(lo_lim, yi, "<", ms=4, color=kind(t.model), zorder=2)
        ax.plot(t.diff, yi, "o", ms=5, color=kind(t.model), mec="white", mew=1, zorder=3)
        ax.text(0.205, yi, f"{t.auc:.2f}", va="center", ha="right", fontsize=7, color=INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=7.2)
    for name, yh in heads:
        ax.text(lo_lim, yh, name, fontsize=7.5, weight="bold", color=INK, va="center")
    ax.text(0.205, heads[0][1], "Ext. AUC", ha="right", va="center", fontsize=7, color=INK2, weight="bold")
    ax.text(0, heads[0][1], "±0.03 margin", ha="center", va="center", fontsize=6.8, color=BLUE)
    ax.set_ylim(min(ys) - 0.8, heads[0][1] + 0.8)
    ax.set_xlim(lo_lim, 0.21)
    ax.set_xticks([-0.3, -0.2, -0.1, 0, 0.1])
    ax.set_xlabel("Difference in pooled external AUC vs clinical model (95% CI, paired DeLong)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "fig_forest_all")


# ----------------------------------------------------------------------------- control tasks
def fig_controls(ctl, perf):
    lab = {"hr": "Hormone receptor status", "her2": "HER2 status", "field3t": "Field strength (3 T vs 1.5 T)",
           "perm": "Shuffled pCR labels (negative control)"}
    rows = []
    for t, name in lab.items():
        for cohort in ("I-SPY2 cross-validation", "External"):
            s = ctl[(ctl.target == t) & (ctl.cohort == cohort)]
            if len(s):
                rows.append((name, cohort, s.auc.iloc[0], s.lo.iloc[0], s.hi.iloc[0]))
    m = "Image only, kinetic 3D ResNet"
    if m in perf.index:
        p = perf.loc[m]
        rows.append(("pCR (the study target)", "I-SPY2 cross-validation", p.internal_auc, p.internal_lo, p.internal_hi))
        rows.append(("pCR (the study target)", "External", p.external_auc, p.external_lo, p.external_hi))
    d = pd.DataFrame(rows, columns=["target", "cohort", "auc", "lo", "hi"])
    order = [n for n in list(lab.values())[:3] + ["pCR (the study target)", lab["perm"]] if n in set(d.target)]
    fig, ax = plt.subplots(figsize=(6.6, 0.55 * len(order) + 0.9))
    ax.axvline(0.5, color=INK2, lw=0.8, ls=(0, (3, 2)))
    for i, n in enumerate(order[::-1]):
        for dy, cohort, col in ((0.13, "I-SPY2 cross-validation", BLUE), (-0.13, "External", ORANGE)):
            s = d[(d.target == n) & (d.cohort == cohort)]
            if len(s):
                t = s.iloc[0]
                ax.plot([t.lo, t.hi], [i + dy] * 2, color=col, lw=1.8, solid_capstyle="round")
                ax.plot(t.auc, i + dy, "o", ms=5.5, color=col, mec="white", mew=1)
                ax.text(t.hi + 0.012, i + dy, f"{t.auc:.2f}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order[::-1])
    ax.set_xlim(0.35, 1.02)
    ax.set_xlabel("AUC (95% CI) of the kinetic 3D ResNet trained on each target from the same tumour crops")
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], color=BLUE, marker="o", lw=1.8), Line2D([], [], color=ORANGE, marker="o", lw=1.8)],
              ["I-SPY2 cross-validation", "External collections"], loc="lower right", fontsize=7)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "fig_controls")


# ----------------------------------------------------------------------------- leave one collection out
def fig_loco(loco, pooled):
    ms = [m for m in loco.model.unique() if m != "Clinical"]
    hs = ["ISPY2", "DUKE", "ISPY1", "NACT"]
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    ax.axhspan(-0.03, 0.03, color="#e8f0fb", zorder=0)
    ax.axhline(0, color=INK2, lw=0.8)
    w = 0.8 / len(ms)
    for j, m in enumerate(ms):
        for i, h in enumerate(hs + ["Pooled"]):
            if h == "Pooled":
                s = pooled[pooled.model == m]
                if not len(s):
                    continue
                v, lo, hi = s.pooled_diff.iloc[0], s.ci_lo.iloc[0], s.ci_hi.iloc[0]
            else:
                s = loco[(loco.model == m) & (loco.held_out == h)]
                if not len(s):
                    continue
                v, lo, hi = s["diff"].iloc[0], s.ci_lo.iloc[0], s.ci_hi.iloc[0]
            x = i + (j - (len(ms) - 1) / 2) * w
            ax.plot([x, x], [lo, hi], color=kind(m), lw=1.6)
            ax.plot(x, v, "o" if h != "Pooled" else "D", ms=5, color=kind(m), mec="white", mew=1, label=m if i == 0 else None)
    n = loco[loco.model == "Clinical"].set_index("held_out")
    ax.set_xticks(range(len(hs) + 1))
    ax.set_xticklabels([f"{COLL[h]} held out\nn = {int(n.loc[h, 'n'])}\nclinical AUC {n.loc[h, 'auc']:.2f}" if h in n.index else COLL[h] for h in hs]
                       + ["Pooled\n(inverse\nvariance)"], fontsize=7)
    ax.set_ylabel("AUC difference vs clinical model\n(held-out collection, 95% CI)")
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    ax.grid(axis="y", color=GRID, lw=0.6)
    save(fig, "fig_loco")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r2", default=os.path.join(ROOT, "results", "round2"))
    ap.add_argument("--samples")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    smp = a.samples or os.path.join(a.r2, "raw", "samples.npz")
    if os.path.exists(smp):
        fig_samples(smp)
    r1 = pd.read_csv(os.path.join(ROOT, "results", "final", "table5_vs_clinical.csv"))
    r1 = r1.rename(columns={"model": "model"})
    r1["model"] = r1.model.replace({"CARF (proposed)": "CARF", "Image only (ResNet)": "Image only, ResNet",
                                    "Image only (DenseNet-121)": "Image only, DenseNet-121",
                                    "Image only (EfficientNet-B0)": "Image only, EfficientNet-B0",
                                    "Image only (ViT)": "Image only, vision transformer"})
    r2 = {"perf": pd.read_csv(os.path.join(a.r2, "r2_performance.csv")), "cmp": pd.read_csv(os.path.join(a.r2, "r2_vs_clinical.csv"))}
    fig_forest(r1, r2)
    perf = r2["perf"].set_index("model")
    if os.path.exists(os.path.join(a.r2, "r2_controls.csv")):
        fig_controls(pd.read_csv(os.path.join(a.r2, "r2_controls.csv")), perf)
    if os.path.exists(os.path.join(a.r2, "r2_loco.csv")):
        fig_loco(pd.read_csv(os.path.join(a.r2, "r2_loco.csv")), pd.read_csv(os.path.join(a.r2, "r2_loco_pooled.csv")))


if __name__ == "__main__":
    main()
