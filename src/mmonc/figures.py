"""Publication figures. Every figure made from synthetic data carries a watermark."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from . import SYNTHETIC_WATERMARK
from .metrics import net_benefit, net_benefit_all, reliability_points

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


def _save(fig, path, synthetic):
    if synthetic:
        fig.text(0.5, 0.5, SYNTHETIC_WATERMARK, ha="center", va="center", rotation=30,
                 fontsize=16, color="red", alpha=0.25, zorder=100)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path + ".png")
    fig.savefig(path + ".pdf")
    plt.close(fig)


def roc_pr(y, preds: dict, path, synthetic=False, title=""):
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for (name, p), c in zip(preds.items(), COLORS):
        fpr, tpr, _ = roc_curve(y, p)
        ax[0].plot(fpr, tpr, color=c, label=f"{name} (AUC {roc_auc_score(y, p):.2f})")
        pr, rc, _ = precision_recall_curve(y, p)
        ax[1].plot(rc, pr, color=c, label=name)
    ax[0].plot([0, 1], [0, 1], "--", color="grey", lw=0.8)
    ax[0].set(xlabel="1 - specificity", ylabel="Sensitivity", title=f"ROC {title}")
    ax[1].axhline(np.mean(y), ls="--", color="grey", lw=0.8)
    ax[1].set(xlabel="Recall", ylabel="Precision", title=f"Precision-recall {title}", ylim=(0, 1))
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.2))
    _save(fig, path, synthetic)


def calibration_plot(y, preds: dict, path, synthetic=False):
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    for (name, p), c in zip(preds.items(), COLORS):
        x, o = reliability_points(y, p)
        ax.plot(x, o, "o-", color=c, ms=3, label=name)
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=0.8)
    ax.set(xlabel="Predicted probability", ylabel="Observed pCR rate", xlim=(0, 1), ylim=(0, 1))
    ax.legend(frameon=False, fontsize=7)
    _save(fig, path, synthetic)


def decision_curve(y, preds: dict, path, synthetic=False, lo=0.10, hi=0.60):
    t = np.linspace(lo, hi, 51)
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    for (name, p), c in zip(preds.items(), COLORS):
        ax.plot(t, net_benefit(y, p, t), color=c, label=name)
    ax.plot(t, net_benefit_all(y, t), color="black", lw=0.8, label="Treat all")
    ax.axhline(0, color="grey", lw=0.8, label="Treat none")
    ax.set(xlabel="Threshold probability", ylabel="Net benefit", ylim=(-0.05, max(0.05, np.mean(y) + 0.02)))
    ax.legend(frameon=False, fontsize=7)
    _save(fig, path, synthetic)


def confusion(cm_by_model: dict, path, synthetic=False):
    k = len(cm_by_model)
    fig, axes = plt.subplots(1, k, figsize=(2.3 * k, 2.4), squeeze=False)
    for ax, (name, m) in zip(axes[0], cm_by_model.items()):
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["no pCR", "pCR"], yticklabels=["no pCR", "pCR"],
               xlabel="Predicted", ylabel="Observed", title=name)
    _save(fig, path, synthetic)


def kaplan_meier(time, event, risk, path, synthetic=False):
    from lifelines import KaplanMeierFitter

    q = np.quantile(risk, [1 / 3, 2 / 3])
    grp = np.digitize(risk, q)
    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    for g, lab, c in zip(range(3), ["Low risk", "Intermediate", "High risk"], COLORS):
        if (grp == g).sum() > 1:
            KaplanMeierFitter().fit(time[grp == g], event[grp == g], label=lab).plot_survival_function(ax=ax, color=c, ci_show=False)
    ax.set(xlabel="Months", ylabel="Recurrence-free survival", ylim=(0, 1.02))
    _save(fig, path, synthetic)


def gradcam_panel(image, mask, cam, path, synthetic=False):
    z = int(np.argmax(mask.sum(axis=(1, 2)))) if mask.sum() else image.shape[1] // 2
    fig, ax = plt.subplots(1, 3, figsize=(6.6, 2.4))
    ax[0].imshow(image[1, z], cmap="gray"); ax[0].set_title("First post-contrast")
    ax[1].imshow(image[2, z], cmap="gray"); ax[1].contour(mask[z], levels=[0.5], colors="cyan", linewidths=0.8)
    ax[1].set_title("Subtraction + tumour mask")
    ax[2].imshow(image[1, z], cmap="gray"); ax[2].imshow(cam[z], cmap="inferno", alpha=0.5)
    ax[2].set_title("Grad-CAM (imaging contribution)")
    for a in ax:
        a.axis("off")
    _save(fig, path, synthetic)


def workflow(path, synthetic=False):
    steps = ["MAMA-MIA\n4 collections", "Crop + radiomics\nfrozen splits", "ISPY2: 5-fold CV\nanchor, baselines, CARF",
             "Freeze + hash\npre-register", "External test\nDUKE, ISPY1, NACT", "Stats, explainability\nbias analysis"]
    fig, ax = plt.subplots(figsize=(9, 1.5))
    for i, s in enumerate(steps):
        ax.text(i, 0, s, ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.5", fc="#EEF2F8", ec="#4C72B0"))
        if i:
            ax.annotate("", xy=(i - 0.38, 0), xytext=(i - 0.62, 0), arrowprops=dict(arrowstyle="->", color="#4C72B0"))
    ax.set(xlim=(-0.6, len(steps) - 0.4), ylim=(-1, 1)); ax.axis("off")
    _save(fig, path, synthetic)


def forest(sub, path, synthetic=False):
    """Subgroup AUCs with 95% CIs, one row per (variable, level), one marker per model."""
    if sub is None or len(sub) == 0:
        return
    keys = list(dict.fromkeys(zip(sub.variable, sub.level)))
    models = list(dict.fromkeys(sub.model))
    fig, ax = plt.subplots(figsize=(5.2, 0.28 * len(keys) + 0.8))
    for j, (m, c) in enumerate(zip(models, COLORS)):
        s = sub[sub.model == m].set_index(["variable", "level"])
        ys, xs, lo, hi = [], [], [], []
        for i, k in enumerate(keys):
            if k in s.index:
                r = s.loc[k]
                ys.append(i + (j - (len(models) - 1) / 2) * 0.25); xs.append(r.auc); lo.append(r.auc - r.lo); hi.append(r.hi - r.auc)
        ax.errorbar(xs, ys, xerr=[lo, hi], fmt="o", ms=3, color=c, label=m, lw=0.8)
    ax.axvline(0.5, color="grey", ls="--", lw=0.8)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f"{v}: {l}" for v, l in keys], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("AUC (95% CI)")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    _save(fig, path, synthetic)
