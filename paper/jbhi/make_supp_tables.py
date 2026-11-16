"""Generate the LaTeX tables of the supplementary material directly from the result files.

    python paper/jbhi/make_supp_tables.py   ->  paper/jbhi/supp_tables.tex
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
R1, R2 = os.path.join(ROOT, "results", "final"), os.path.join(ROOT, "results", "round2")
COLL = {"DUKE": "Duke", "ISPY1": "I-SPY1", "ISPY2": "I-SPY2", "NACT": "NACT"}
NAME = {"CARF (proposed)": "CARF", "Image only (ResNet)": "ResNet, image only", "Image only (DenseNet-121)": "DenseNet-121, image only",
        "Image only (EfficientNet-B0)": "EfficientNet-B0, image only", "Image only (ViT)": "Vision transformer, image only",
        "CARF (proposed) seed 1": "CARF, seed 1", "CARF (proposed) seed 2": "CARF, seed 2"}


def f2(x):
    return "--" if pd.isna(x) else f"{x:.2f}"


def f3(x):
    return "--" if pd.isna(x) else f"{x:.3f}".replace("-", "$-$")


def ci(a, lo, hi, nd=2):
    f = f2 if nd == 2 else f3
    return f"{f(a)} ({f(lo)}--{f(hi)})".replace("--$-$", " to $-$")


def table(caption, label, cols, rows, spec, note=None, size="\\scriptsize", star=False):
    env = "table*" if star else "table"
    out = [f"\\begin{{{env}}}[!htbp]", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\centering", size,
           "\\setlength{\\tabcolsep}{3pt}", f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(cols) + "\\\\", "\\midrule"]
    out += [" & ".join(r) + "\\\\" for r in rows]
    out.append("\\bottomrule")
    if note:
        out.append(f"\\multicolumn{{{len(cols)}}}{{p{{{'0.95\\textwidth' if star else '0.95\\columnwidth'}}}}}{{\\tiny {note}}}\\\\")
    out += ["\\end{tabular}", f"\\end{{{env}}}", ""]
    return "\n".join(out)


parts = []

# S4: secondary analyses summary (moved from the main text)
lp = pd.read_csv(os.path.join(R2, "r2_loco_pooled.csv")).set_index("model")
av = pd.read_csv(os.path.join(R2, "r2_added_value.csv"))
ct = pd.read_csv(os.path.join(R2, "r2_controls.csv"))
site = pd.read_csv(os.path.join(R1, "tableS_site_pooled.csv")).set_index("model")
rows = []
for m in ["Clinical + kinetic radiomics", "CARF, kinetic", "Image only, kinetic 3D ResNet"]:
    r = lp.loc[m]
    rows.append([f"LOCO pooled $\\Delta$: {m}", f"{f3(r.pooled_diff)} ({f3(r.ci_lo)}, {f3(r.ci_hi)}); $I^2$ {r.i2*100:.0f}\\%"])
for m in ["Image only, pretrained 2.5D", "Image only, kinetic 3D ResNet", "Kinetic radiomics"]:
    r = av[(av.group == "All external") & (av.image_score == m)].iloc[0]
    rows.append([f"Added to clinical logit: {m}", f"LR $P$={r.p:.2f}; stacking {f3(r.stack_auc_gain)} ({f3(r.stack_ci_lo)}, {f3(r.stack_ci_hi)})"])
lab = {"hr": "HR status", "her2": "HER2 status", "field3t": "Field strength", "perm": "Shuffled pCR labels"}
for t in ["hr", "her2", "field3t", "perm"]:
    for coh in ["I-SPY2 cross-validation", "External"]:
        s = ct[(ct.target == t) & (ct.cohort == coh)]
        if len(s):
            r = s.iloc[0]
            rows.append([f"Control task, {lab[t]} ({'internal' if coh.startswith('I-SPY2') else 'external'})", ci(r.auc, r.lo, r.hi)])
for m, nm in [("Clinical", "clinical"), ("CARF (proposed)", "CARF")]:
    r = site.loc[m]
    rows.append([f"I-SPY2 sites ($k$={int(r.k)}), pooled AUC, {nm}", f"{ci(r.pooled_auc, r.lo, r.hi)}; $I^2$ {r.i2*100:.0f}\\%"])
parts.append(table("Summary of Secondary Analyses", "tab:s_secondary", ["Analysis", "Result (95\\% CI)"], rows, "p{4.2cm}p{4.2cm}",
                   "LOCO: leave-one-collection-out; $\\Delta$: AUC difference from the clinical model; LR: likelihood-ratio test; stacking: fivefold cross-validated logistic model of clinical and image logits fitted within the external data (optimistic bound)."))

# S5: per-collection external AUC, round 1 and round 2
r1 = pd.read_csv(os.path.join(R1, "table3_external.csv"))
r1 = r1[r1.cohort != "Pooled external"]
p1 = r1.pivot_table(index="model", columns="cohort", values="auc")
r2 = pd.read_csv(os.path.join(R2, "r2_per_collection.csv")).pivot_table(index="model", columns="collection", values="auc")
rows = [[NAME.get(m, m), "1"] + [f2(p1.loc[m, c]) for c in ["DUKE", "ISPY1", "NACT"]] for m in p1.index]
rows += [[m, "2"] + [f2(r2.loc[m, c]) for c in ["DUKE", "ISPY1", "NACT"]] for m in r2.index if m != "Clinical"]
parts.append(table("External AUC by Collection (Duke $n$=280, I-SPY1 $n$=167, NACT $n$=64)", "tab:s_coll", ["Model", "Round", "Duke", "I-SPY1", "NACT"], rows, "lcccc"))

# S6: external calibration and threshold metrics
e1 = pd.read_csv(os.path.join(R1, "table3_external.csv"))
e1 = e1[e1.cohort == "Pooled external"]
rows = [[NAME.get(r.model, r.model), "1", f2(r.slope), f2(r.intercept), f3(r.brier), f2(r.sensitivity), f2(r.specificity), f2(r.balanced_accuracy)] for r in e1.itertuples()]
e2 = pd.read_csv(os.path.join(R2, "r2_performance.csv"))
rows += [[r.model, "2", f2(r.ext_slope), f2(r.ext_intercept), f3(r.ext_brier), f2(r.ext_sensitivity), f2(r.ext_specificity), f2(r.ext_balanced_accuracy)]
         for r in e2.itertuples() if r.model != "Clinical"]
parts.append(table("External Calibration and Performance at the Frozen I-SPY2 Threshold", "tab:s_cal",
                   ["Model", "Rd", "Slope", "Intercept", "Brier", "Sens.", "Spec.", "Bal. acc."], rows, "lccccccc",
                   "Threshold: Youden index on I-SPY2 out-of-fold predictions, applied unchanged externally ($n$=511; 124 with pCR).", star=True))

# S7: LOCO per held-out collection
lo = pd.read_csv(os.path.join(R2, "r2_loco.csv"))
rows = []
for r in lo.itertuples():
    rows.append([COLL[r.held_out], r.model, f"{int(r.n)} ({int(r.events)})", ci(r.auc, r.lo, r.hi),
                 "reference" if pd.isna(r.diff) else ci(r.diff, r.ci_lo, r.ci_hi, 3)])
parts.append(table("Leave-One-Collection-Out Validation", "tab:s_loco", ["Held out", "Model", "$n$ (pCR)", "AUC (95\\% CI)", "$\\Delta$ vs clinical (95\\% CI)"],
                   rows, "llccc", star=True))

# S8: within-subtype AUC (external) and added-value tests
st = pd.read_csv(os.path.join(R2, "r2_subtype_auc.csv"))
keep = ["Clinical", "Clinical + FTV", "CARF, kinetic", "CARF, pretrained 2.5D", "Image only, pretrained 2.5D", "Image only, kinetic 3D ResNet", "Kinetic radiomics"]
rows = []
for g in ["HR+/HER2-", "HER2+", "Triple negative"]:
    s = st[st.group == g]
    n, e = int(s.n.iloc[0]), int(s.events.iloc[0])
    for m in keep:
        r = s[s.model == m].iloc[0]
        rows.append([g.replace("-", "$-$") + f" ({n}; {e})" if m == "Clinical" else "", m, ci(r.auc, r.lo, r.hi)])
parts.append(table("External AUC Within Receptor Subtypes", "tab:s_subtype", ["Subtype ($n$; pCR)", "Model", "AUC (95\\% CI)"], rows, "llc",
                   "Within a subtype, the clinical model reduces essentially to age (and HR within HER2+). Likelihood-ratio tests of adding each image score within subtypes: Holm-adjusted $P\\ge.44$."))

# S9: recalibration
rc = pd.read_csv(os.path.join(R2, "r2_recalibration.csv"))
rows = [[r.model, COLL[r.collection], f2(r.intercept_before), f2(r.intercept_after), f3(r.brier_before), f3(r.brier_after), f3(r.ece_before), f3(r.ece_after)]
        for r in rc.itertuples()]
parts.append(table("Intercept-Only Recalibration per External Collection (Fivefold Cross-Validated)", "tab:s_recal",
                   ["Model", "Coll.", "Int. before", "Int. after", "Brier before", "Brier after", "ECE before", "ECE after"], rows, "llcccccc", star=True))

# S10: clinical variants, survival, gate
cv = pd.read_csv(os.path.join(R1, "tableS_clinical_variants.csv"))
rows = [[r.model.replace("tumour", "tumor"), r.internal_auc.replace("-", "--"), r.external_auc.replace("-", "--")] for r in cv.itertuples()]
parts.append(table("Clinical Model Variants", "tab:s_clinvar", ["Model", "Internal AUC", "External AUC"], rows, "lcc"))
sv = pd.read_csv(os.path.join(R1, "tableS_survival.csv"))
rows = [[NAME.get(r.model, r.model), str(int(r.n)), str(int(r.events)), f3(r.c_index), f"{r.hr_per_sd_adj:.2f} ({r.hr_lo:.2f}--{r.hr_hi:.2f})", f"{r.p:.2f}"] for r in sv.itertuples()]
parts.append(table("Exploratory Recurrence-Free Survival (External Collections)", "tab:s_surv",
                   ["Model", "$n$", "Events", "C index", "HR per SD (adj.)", "$P$"], rows, "lccccc",
                   "C index of predicted non-response for recurrence; Cox model adjusted for HR and HER2 status. Values below 0.5 indicate an inverse relation (subtypes with higher pCR rates also recur more)."))
g = pd.read_csv(os.path.join(R1, "tableS_gate.csv"))
rows = [[COLL[r.group.split("=")[1]], str(int(r.n)), f3(r.gate_mean), f3(r.gate_sd), f3(r.mean_abs_imaging_shift_logit)] for r in g.itertuples()]
g2 = pd.read_csv(os.path.join(R2, "r2_gate.csv"), header=[0, 1])
parts.append(table("CARF (Round 1, 3D Encoder) Gate and Imaging Contribution", "tab:s_gate",
                   ["Collection", "$n$", "Gate mean", "Gate SD", "Mean $|g\\delta|$ (logit)"], rows, "lcccc",
                   "Round 2: mean gate 0.17 (kinetic 3D encoder) in every collection; 0.71 (I-SPY2) and 0.71--0.73 (external) with the pretrained 2.5D encoder."))

open(os.path.join(HERE, "supp_tables.tex"), "w").write("\n".join(parts))
print("written", len(parts), "tables")
