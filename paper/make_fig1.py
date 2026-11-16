"""Figure 1: study design and patient flow (both rounds)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE, ORANGE, GREY, INK, INK2 = "#2a78d6", "#eb6834", "#6b7280", "#111827", "#374151"
plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(10, 6.2), dpi=220)
ax.set_xlim(0, 100); ax.set_ylim(-3, 62); ax.axis("off")


def box(x, y, w, h, title, body="", c=BLUE, fill=None, fs=8.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1", lw=1.2, ec=c, fc=fill or c + "12"))
    ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", weight="bold", fontsize=fs + 0.6, color=INK)
    if body:
        ax.text(x + w / 2, y + h - 4.6, body, ha="center", va="top", fontsize=fs, color=INK2, linespacing=1.35)


def arrow(x0, y0, x1, y1, c=GREY):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10, lw=1.1, color=c, shrinkA=0, shrinkB=0))


box(30, 53, 40, 8, "MAMA-MIA (Synapse syn60868042)", "1,506 patients, 4 U.S. collections, 1995–2016")
box(74, 53.6, 24, 6.8, "Excluded: 15", "no pCR status (Duke 11, I-SPY1 4)", GREY, fs=7.8)
arrow(70, 57, 73.6, 57)
arrow(50, 52.6, 50, 49.4)
box(30, 41.5, 40, 7.6, "Included: 1,491 women", "440 with pCR (29.5%)")
arrow(40, 41.1, 22, 37.4); arrow(60, 41.1, 78, 37.4)
box(2, 26, 40, 11, "Development: I-SPY2", "980 patients (316 pCR), 22 sites\nfivefold cross-validation\nmodels and thresholds frozen")
box(58, 26, 40, 11, "External test, scored once", "Duke 280 (64 pCR)\nI-SPY1 167 (49)    NACT 64 (11)\n511 patients, 124 pCR", ORANGE)
box(2, -2.5, 46, 24.5, "Models compared with the clinical model",
    "Clinical: age, HR, HER2 (reference)\n\nRound 1 (prespecified): radiomics ×2,\n4 image-only 3D networks, concatenation\nfusion, CARF (+3 ablations, 2 seeds)\n\nRound 2 (secondary): kinetic radiomics ×2,\nclinical + FTV, kinetic 3D network (alone,\nconcatenation, CARF), pretrained 2.5D (alone, CARF)", fs=7.6)
box(52, -2.5, 46, 24.5, "Analyses",
    "Primary: external AUC, CARF vs clinical,\n±0.03 equivalence margin (paired DeLong)\n\nAll models: discrimination, calibration,\ndecision curves, subgroups, sites\n\nRound 2: leave-one-collection-out (4 splits),\ncontrol tasks (HR, HER2, field strength,\nshuffled labels), added-value tests", fs=7.6)
arrow(22, 25.6, 22, 22.4); arrow(78, 25.6, 78, 22.4)
arrow(42.4, 31.5, 57.6, 31.5, ORANGE)
ax.text(50, 32.4, "frozen models", ha="center", fontsize=7.4, color=ORANGE)
fig.savefig("paper/figures/fig1_workflow.png", bbox_inches="tight", facecolor="white")
fig.savefig("paper/figures/fig1_workflow.pdf", bbox_inches="tight")
