"""Architecture schematic of clinically anchored residual fusion (CARF)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE, ORANGE, GREEN, GREY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#6b7280", "#111827"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig, ax = plt.subplots(figsize=(11, 5.6), dpi=200)
ax.set_xlim(0, 110); ax.set_ylim(0, 56); ax.axis("off")


def box(x, y, w, h, title, sub="", c=BLUE, fill=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", lw=1.3, ec=c,
                                fc=fill or (c + "14")))
    ax.text(x + w / 2, y + h / 2 + (1.5 if sub else 0), title, ha="center", va="center", weight="bold", color=INK, fontsize=9)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 1.9, sub, ha="center", va="center", color="#374151", fontsize=7.6, linespacing=1.25)


def arrow(x0, y0, x1, y1, c=GREY, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, lw=1.2, color=c, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


# column 1: inputs
box(1, 42, 21, 10, "DCE MRI tumour crop", "pre, early, late phases\nenhancement, washout, mask", BLUE)
box(1, 26, 21, 9, "Acquisition times", "of the three phases", BLUE)
box(1, 12, 21, 9, "Tabular features", "age, HR, HER2\n22 kinetic features", GREEN)
box(1, 0.8, 21, 7, "Clinical variables", "age, HR, HER2", ORANGE)
# column 2: encoders
box(29, 42, 21, 10, "Image encoder", "3D ResNet (kinetic) or\npretrained 2.5D ResNet-18", BLUE)
box(29, 26, 21, 9, "Timing embedding", "MLP of log times", BLUE)
box(29, 12, 21, 9, "Feature tokens + [CLS]", "one token per variable;\nlearned token if missing", GREEN)
box(29, 0.8, 21, 7, "Clinical model (frozen)", "logistic regression", ORANGE)
# column 3: fusion
box(57, 44, 21, 8.5, "Shift score", "distance of image embedding\nfrom training distribution", GREY)
box(57, 22, 21, 12, "Cross-attention", "feature tokens query\nimage tokens", BLUE)
# column 4: heads
box(84, 44.7, 11, 7, "Gate g", "0 to 1", GREY)
box(84, 26, 11, 6, "Residual δ", "", BLUE)
ax.add_patch(plt.Circle((103, 38.5), 2.1, ec=INK, fc="white", lw=1.3))
ax.text(103, 38.4, "×", ha="center", va="center", fontsize=13, color=INK)
ax.add_patch(FancyBboxPatch((80, 1), 28, 14, boxstyle="round,pad=0.25,rounding_size=1.2", lw=1.3, ec=ORANGE, fc="#fff7ed"))
ax.text(94, 12, "logit = z_clin + g · δ", ha="center", va="center", weight="bold", fontsize=10, color=INK)
ax.text(94, 5.6, "predicted probability of pCR\nno image or unfamiliar image: g → 0,\nso the output is the clinical model", ha="center",
        va="center", fontsize=7.6, color="#374151", linespacing=1.3)

for y in (47, 30.5, 16.5, 4.3):
    arrow(22.6, y, 28.4, y)
arrow(50.6, 49, 56.4, 49, GREY, ls="--")     # image embedding -> shift score
arrow(50.6, 44.5, 56.6, 34.4, BLUE)          # image tokens -> attention
ax.text(54.6, 41.2, "image tokens", fontsize=7.4, color=BLUE, va="center", ha="left")
arrow(39.5, 35.6, 39.5, 41.4, BLUE)          # timing added to image tokens
ax.text(40.3, 38.2, "added", fontsize=7.4, color=BLUE)
arrow(50.6, 17.5, 56.6, 23.2, GREEN)         # tabular tokens -> attention
ax.text(54.6, 18.6, "queries", fontsize=7.4, color=GREEN, ha="left")
arrow(78.6, 49, 83.4, 49, GREY)              # shift -> gate
arrow(78.6, 31.5, 83.6, 44.6, GREY)          # CLS -> gate
arrow(78.6, 28.5, 83.4, 28.8, BLUE)          # CLS -> residual
ax.text(79.0, 25.2, "[CLS]", fontsize=7.4, color="#374151")
arrow(95.6, 47, 101.6, 40.4, GREY)           # gate -> multiply
arrow(95.6, 29.6, 101.6, 36.6, BLUE)         # residual -> multiply
arrow(103, 36.2, 100, 15.8, INK)             # g*delta -> output
ax.text(103.2, 25.5, "g · δ", fontsize=8.5, color=INK, weight="bold")
arrow(50.6, 4.3, 79.4, 6.2, ORANGE)          # z_clin -> output
ax.text(64.5, 7.0, "z_clin  (clinical logit)", fontsize=8, color=ORANGE, weight="bold", ha="center")
fig.savefig("paper/figures/fig_architecture.png", bbox_inches="tight", facecolor="white")
fig.savefig("paper/figures/fig_architecture.pdf", bbox_inches="tight")
