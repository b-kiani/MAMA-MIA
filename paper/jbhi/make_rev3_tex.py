"""Tables and figure for the third-review analyses, from results/revision3 (revision3_analyses.py first).

    python paper/jbhi/make_rev3_tex.py  ->  rev3_tab_repeats.tex, rev3_tab_recovery.tex (main),
                                            rev3_supp_tables.tex (supplement), figures/fig_rev3.*
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
R3 = os.path.join(ROOT, "results", "revision3")
FIG = os.path.join(HERE, "figures")
NM = {"Clinical": "Clinical (age, HR, HER2)", "lf": "Late fusion, clinical + 2.5D image", "carfm_25d": "CARF-mix, pretrained 2.5D",
      "ckin": "Clinical + kinetic radiomics", "s25d": "Image only, pretrained 2.5D", "carfm_k": "CARF-mix, kinetic 3D",
      "image_only_k": "Image only, kinetic 3D"}
COLL = {"DUKE": "Duke", "ISPY1": "I-SPY1", "NACT": "NACT"}
GRP = {"all": "All I-SPY2", "control": "Control (paclitaxel $\\pm$ trastuzumab)", "her2_targeted": "HER2-targeted experimental",
       "immunotherapy": "Pembrolizumab", "other_experimental": "Other experimental"}
BLUE, ORANGE, GREY, INK = "#2a78d6", "#eb6834", "#6b7280", "#111827"


def C(*w):
    return "".join(f">{{\\centering\\arraybackslash}}p{{{x}cm}}" for x in w)


def f3(x, d=3):
    if pd.isna(x):
        return "--"
    s = f"{x + 0.0:.{d}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return s.replace("-", "$-$")


def ci(a, lo, hi):
    return f"{f3(a)} ({f3(lo)} to {f3(hi)})"


def rng(a, lo, hi):
    return f"{f3(a)} ({f3(lo)} to {f3(hi)})"


def table(caption, label, cols, rows, spec, note, env="table", size="\\scriptsize"):
    if spec.startswith("p{"):
        spec = ">{\\raggedright\\arraybackslash}" + spec
    out = [f"\\begin{{{env}}}[!htbp]", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\centering", size, "\\setlength{\\tabcolsep}{3pt}",
           f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(cols) + "\\\\", "\\midrule"]
    for r in rows:
        out.append("\\midrule" if r == "MID" else (" & ".join(r) + "\\\\"))
    out += ["\\bottomrule", f"\\multicolumn{{{len(cols)}}}{{p{{0.95\\linewidth}}}}{{\\scriptsize {note}}}\\\\", "\\end{tabular}", f"\\end{{{env}}}", ""]
    return "\n".join(out)


def main_tables():
    R = pd.read_csv(os.path.join(R3, "rev3_repeats_lps.csv"))
    S = pd.read_csv(os.path.join(R3, "rev3_repeats_lps_summary.csv")).set_index("model")
    O = pd.read_csv(os.path.join(R3, "rev3_orientation.csv"))
    rows = []
    for m in ["Clinical", "lf", "carfm_25d", "ckin", "s25d", "carfm_k"]:
        if m not in S.index:
            continue
        s = S.loc[m]
        r = R[R.model == m]
        o = O[O.model == m]
        widest = f"{f3(r.ext_lo.min())} to {f3(r.ext_hi.max())}" if m != "Clinical" else "--"
        fold = f"{f3(s.fold_ext_min)}--{f3(s.fold_ext_max)}" if pd.notna(s.get("fold_ext_min")) else "--"
        ori = (f"{f3(o.ext_diff.mean())} ({f3(o.ext_diff.min())} to {f3(o.ext_diff.max())})" if len(o) > 1 else (f3(o.ext_diff.iloc[0]) if len(o) else "--"))
        oof = f"{f3(s.oof_mean)} ({f3(s.oof_sd)})" if s.n_seeds > 1 else f3(s.oof_mean)
        ext = f"{f3(s.ext_mean)} ({f3(s.ext_min)}--{f3(s.ext_max)})" if s.n_seeds > 1 and s.ext_max > s.ext_min else f3(s.ext_mean)
        dif = "--" if m == "Clinical" else (f"{f3(s.diff_mean)} ({f3(s.diff_min)} to {f3(s.diff_max)})" if s.n_seeds > 1 and s.diff_max > s.diff_min else f3(s.diff_mean))
        rows.append([NM[m], str(int(s.n_seeds)), oof, ext, dif, widest, fold, ori])
    t1 = table("Repeated Training After Reorientation (Round 3, Post Hoc): Variation Across Seeds and Fold Splits",
               "tab:repeats", ["Model", "Seeds", "I-SPY2 OOF AUC, mean (SD)", "External AUC, mean (range)", "External $\\Delta$, mean (range)",
                               "Widest 95\\% CI of $\\Delta$", "Single fold models, external AUC", "Reorientation effect on external AUC, mean (range)"],
               rows, "p{3.0cm}c" + C(1.6, 1.8, 2.2, 1.8, 1.6, 2.4),
               "All inputs reoriented to a common anatomical frame (LPS). Each seed changes the outer folds, the inner early-stopping split and the "
               "initialization; seed 0 reproduces earlier rounds. $\\Delta$: paired DeLong difference from the clinical model of the same seed (external: "
               "fitted on all of I-SPY2, so identical across seeds). External predictions are the mean of five fold models; single fold models are "
               "reported separately. Reorientation effect: external AUC with reoriented minus original inputs, same code, folds and seed "
               "(kinetic 3D: compared with the round-3 nested refit). CARF-mix 3D seeds 1--4 ran in a second Kaggle run (version 8). OOF, out-of-fold; CARF, clinically anchored residual fusion.")
    # recovery
    V = pd.read_csv(os.path.join(R3, "rev3_recovery.csv"))
    rows = []
    for lev in ["0", "0.02", "0.035", "0.05", "0.10", "permuted"]:
        for d in (0, 1):
            x = V[(V.level.astype(str) == lev) & (V.draw == d)].set_index("model")
            if not len(x):
                continue
            nom = {"0": "None ($\\beta=0$)", "permuted": "Images permuted"}.get(lev, lev)
            orc = "--" if lev == "permuted" else f3(x.oracle_ext.iloc[0])
            c, l = x.loc["carfm_25d"], x.loc["lf"]
            rows.append([nom if d == 0 else "", str(d + 1), orc, ci(c.ext_diff, c.ext_lo, c.ext_hi), ci(l.ext_diff, l.ext_lo, l.ext_hi),
                         ci(c.oof_diff, c.oof_lo, c.oof_hi), ci(l.oof_diff, l.oof_lo, l.oof_hi)])
        rows.append("MID")
    rows = rows[:-1]
    t2 = table("Conditional Signal-Recovery Control (Round 3, Post Hoc): Recovery of a Planted Imaging Signal Beyond the Clinical Model",
               "tab:recovery", ["Planted gain (nominal)", "Draw", "Realized external oracle gain", "CARF-mix 2.5D, external $\\Delta$",
                                "Late fusion, external $\\Delta$", "CARF-mix 2.5D, I-SPY2 OOF $\\Delta$", "Late fusion, I-SPY2 OOF $\\Delta$"],
               rows, "p{1.7cm}cc" + C(2.7, 2.7, 2.7, 2.7),
               "Semi-synthetic outcomes $y^*\\sim\\mathrm{Bernoulli}(\\sigma(z+\\beta s))$: $z$, clinical logit; $s$, mean early percentage enhancement in the "
               "tumor, residualized on age, HR and HER2 and standardized. $\\beta$ was set before the run for expected external oracle gains of 0, 0.02, "
               "0.035, 0.05 and 0.10; the realized gain $\\mathrm{AUC}(z+\\beta s)-\\mathrm{AUC}(z)$ differs between draws. All models were retrained on $y^*$ "
               "with clinical-only tabular inputs, so the signal could enter only through the image; $\\Delta$ is the paired DeLong difference from the "
               "clinical model refitted on $y^*$ on the same folds (95\\% CI). Images permuted: real pCR labels, images shuffled within collection and "
               "clinical-risk quintile. Reoriented inputs, seed 0.")
    open(os.path.join(HERE, "rev3_tab_repeats.tex"), "w").write(t1)
    open(os.path.join(HERE, "rev3_tab_recovery.tex"), "w").write(t2)


def supp_tables():
    out = []
    # audit
    A = pd.read_csv(os.path.join(R3, "rev3_audit_summary.csv")).set_index("collection")
    order = ["ISPY2", "DUKE", "ISPY1", "NACT"]
    lab = {"ISPY2": "I-SPY2", "DUKE": "Duke", "ISPY1": "I-SPY1", "NACT": "NACT"}
    rows = [["Patients"] + [str(int(A.loc[c, "n"])) for c in order],
            ["Acquisition plane (table)"] + [A.loc[c, "view"].split()[0] for c in order],
            ["Stored orientation code, phases"] + [A.loc[c, "stored_code_pre"].replace(" ", ": ") for c in order],
            ["Stored orientation code, mask"] + [A.loc[c, "stored_code_mask"].replace(" ", ": ") for c in order],
            ["Phases and mask share size, spacing, origin, direction, \\%"] + [f"{100 * A.loc[c, 'phase_mask_geometry_match']:.0f}" for c in order],
            ["Orientation after reorientation"] + [A.loc[c, "code_after"].replace(" ", ": ") for c in order],
            ["Crop partly outside the image, \\%"] + [f"{100 * A.loc[c, 'crop_inside_image_lt1']:.1f}" for c in order],
            ["Smallest fraction of a crop inside the image"] + [f3(A.loc[c, "crop_coverage_min"], 2) for c in order],
            ["Tumor cut by the tumor-scaled crop, \\%"] + [f"{100 * A.loc[c, 'tumour_truncated']:.0f}" for c in order],
            ["Tumor extent, median, mm"] + [f"{A.loc[c, 'tumour_extent_mm_median']:.0f}" for c in order],
            ["Tumor cut by the 96-mm field of view, \\%"] + [f"{100 * A.loc[c, 'fov96_tumour_truncated']:.1f}" for c in order],
            ["Larger-context box: tissue coverage, median (10th pct.)"] + [f"{f3(A.loc[c, 'ctx_coverage_median'], 2)} ({f3(A.loc[c, 'ctx_coverage_p10'], 2)})" for c in order],
            ["Kinetic features, original vs reoriented: max. abs. change, median (max.)"] + [f"{f3(A.loc[c, 'kinetic_orig_vs_lps_maxabs_median'], 3)} ({A.loc[c, 'kinetic_orig_vs_lps_maxabs_max']:.2g})" for c in order]]
    out.append(table("Orientation and Preprocessing Audit (Round 3)", "tab:s_audit", ["", *[lab[c] for c in order]], rows, "p{6.2cm}cccc",
                     "Orientation codes follow the ITK/DICOM convention (letters give the direction toward which the first, second and third array axes "
                     "increase; LAI: left, anterior, inferior; PSL: posterior, superior, left). Rounds 1 and 2 kept these stored orders, so the anatomical "
                     "meaning of the array axes differed between axial (I-SPY2, Duke) and sagittal (I-SPY1, NACT) acquisitions. Larger-context box: "
                     "$48\\times160\\times160$ voxels at 1.5\\,mm around the tumor center; tissue, pre-contrast signal above the Otsu threshold on the "
                     "side of the tumor, no more than 30\\,mm posterior to its center (approximate). Kinetic changes come from the different resampling grid "
                     "after reorientation; the largest occur in ratio features of very small tumor cores (6\\% of cases change by more than 0.5 in any feature).",
                     size="\\footnotesize"))
    # orientation per collection + calibration
    O = pd.read_csv(os.path.join(R3, "rev3_orientation.csv"))
    rows = []
    for m in ["s25d", "lf", "carfm_25d", "carfm_k", "image_only_k"]:
        o = O[O.model == m]
        if not len(o):
            continue
        def mr(col):
            return f3(o[col].mean()) if len(o) == 1 else f"{f3(o[col].mean())} ({f3(o[col].min())} to {f3(o[col].max())})"
        cal = "; ".join(f"{COLL[c]} {o[f'orig_slope_{c}'].mean():.2f}$\\to${o[f'lps_slope_{c}'].mean():.2f}" for c in COLL)
        rows.append([NM[m], str(len(o)), mr("ext_diff"), f"{f3(o.ext_lo.min())} to {f3(o.ext_hi.max())}", mr("oof_diff"), mr("diff_DUKE"), mr("diff_ISPY1"), mr("diff_NACT"), cal])
    out.append(table("Original Versus Reoriented Inputs: Paired AUC Differences and Calibration Slopes (Round 3)", "tab:s_orient",
                     ["Model", "Seeds", "External $\\Delta$ (reoriented $-$ original)", "Widest 95\\% CI", "I-SPY2 OOF $\\Delta$", "Duke", "I-SPY1", "NACT",
                      "Calibration slope, original$\\to$reoriented"],
                     rows, "p{2.6cm}c" + C(1.9, 1.5, 1.6, 1.85, 1.85, 1.85) + "p{2.4cm}",
                     "Mean (range over seeds) of paired DeLong differences between the same model trained on reoriented and on original inputs, with the same "
                     "code, folds and initialization. Calibration slopes are means over seeds."))
    # ablation
    B = pd.read_csv(os.path.join(R3, "rev3_ablation.csv"))
    rows = []
    for st in B.setting.unique():
        rows.append([f"\\multicolumn{{9}}{{l}}{{\\emph{{{st.replace('planted 0.05', 'Planted signal, nominal gain 0.05, clinical-only tabular inputs').replace('real', 'Real pCR labels')}}}}}"])
        for r in B[B.setting == st].itertuples():
            if r.mix == "late fusion":
                name = "Late fusion"
            else:
                name = f"mix {'on' if int(r.mix) else 'off'}, bound {'on' if r.bound == 1 else 'off'}, gate reg. {'on' if r.gate_reg == 1 else 'off'}" + (" (CARF-mix)" if (int(r.mix), r.bound, r.gate_reg) == (1, 1, 1) else (" (CARF)" if (int(r.mix), r.bound, r.gate_reg) == (0, 0, 1) else ""))
            rows.append([name, f3(r.ext_auc), ci(r.ext_diff, r.ext_lo, r.ext_hi), f3(r.oof_diff), f"{r.slope_ext:.2f}", f"{r.int_ext:.2f}",
                         f"{r.corr_abs_ext:.2f}" if pd.notna(r.corr_abs_ext) else "--", f"{r.gate_ext:.2f}" if pd.notna(r.gate_ext) else "--",
                         f"{r.recovery_ext:.2f}" if pd.notna(getattr(r, "recovery_ext", np.nan)) else "--"])
    out.append(table("Factorial Ablation of the Corrected Fusion on the Pretrained 2.5D Encoder (Round 3)", "tab:s_ablation",
                     ["Configuration", "Ext. AUC", "Ext. $\\Delta$ (95\\% CI)", "OOF $\\Delta$", "Slope", "Intercept", "Mean $|g\\delta|$", "Mean gate", "Recovery"],
                     rows, "p{5.2cm}cccccccc",
                     "Token mixing, bounded residual ($B\\tanh(r/B)$, $B=1$) and gate regularization ($0.01\\times$ mean gate) switched on or off; nested "
                     "anchors throughout; development-only selection. Slope and intercept: external calibration. Recovery: external $\\Delta$ divided by the "
                     "realized oracle gain (0.030 in this draw). Reoriented inputs."))
    # FOV
    F = pd.read_csv(os.path.join(R3, "rev3_fov.csv"))
    rows = []
    for m in ["s25d", "lf", "carfm_25d"]:
        f = F[F.model == m]
        rows.append([NM[m], str(len(f)), f"{f3(f.ext_diff.mean())} ({f3(f.ext_diff.min())} to {f3(f.ext_diff.max())})", f"{f3(f.ext_lo.min())} to {f3(f.ext_hi.max())}",
                     f"{f3(f.oof_diff.mean())} ({f3(f.oof_diff.min())} to {f3(f.oof_diff.max())})", f"{f3(f.fixed_vs_clin.mean())} ({f3(f.fixed_vs_clin.min())} to {f3(f.fixed_vs_clin.max())})"])
    out.append(table("Fixed Physical Field of View Versus Tumor-Scaled Crops (Round 3)", "tab:s_fov",
                     ["Model", "Seeds", "External $\\Delta$ (fixed $-$ tumor-scaled)", "Widest 95\\% CI", "I-SPY2 OOF $\\Delta$", "Fixed field of view: external $\\Delta$ vs clinical"],
                     rows, "p{3.6cm}c" + C(2.6, 1.9, 2.4, 2.8),
                     "Fixed field of view: a 96-mm cube centered on the tumor, resized to $64^3$ voxels and $112\\times112$ slices, so image-only models keep "
                     "absolute tumor size; 5.2\\% of tumors extend beyond it. Mean (range) over seeds 0--2. Reoriented inputs."))
    # treatment
    T = pd.read_csv(os.path.join(R3, "rev3_treatment.csv"))
    rows = []
    for an, title in [("within group (5-fold OOF, seed-averaged)", "Imaging increment within regimen groups (I-SPY2 out-of-fold, predictions averaged over five seeds)"),
                      ("regimen group held out", "Regimen group held out (each group predicted by models trained on the other groups)")]:
        rows.append([f"\\multicolumn{{6}}{{l}}{{\\emph{{{title}}}}}"])
        x = T[T.analysis == an]
        for g in GRP:
            for m in ("lf", "carfm_25d"):
                r = x[(x.group == g) & (x.model == m)]
                if len(r):
                    r = r.iloc[0]
                    rows.append([f"{GRP[g]}, {'late fusion' if m == 'lf' else 'CARF-mix 2.5D'}", f"{int(r.n)} ({int(r.events)})", f3(r.auc_clin), f3(r.auc_model), ci(r["diff"], r.lo, r.hi), ""])
    rows.append(["\\multicolumn{6}{l}{\\emph{Matched training size: 209 patients (external evaluation of late fusion vs a clinical model trained on the same patients)}}"])
    for r in T[T.analysis.str.startswith("matched")].itertuples():
        rows.append(["Control arm" if r.group == "ctl" else f"Random all-arm subset {int(r.group[3:]) + 1}", "209", f3(r.auc_clin), f3(r.auc_model), ci(r.diff, r.lo, r.hi),
                     f"{f3(r.oof_clin)} / {f3(r.oof_model)}"])
    out.append(table("Treatment-Aware Analyses of the Imaging Increment (Round 3)", "tab:s_treat",
                     ["Analysis", "$n$ (pCR)", "AUC, clinical", "AUC, with imaging", "$\\Delta$ (95\\% CI)", "OOF AUC, clinical / with imaging"],
                     rows, "p{5.6cm}" + C(1.3, 1.3, 1.4, 2.6, 2.2),
                     "Regimen groups of I-SPY2: control, paclitaxel $\\pm$ trastuzumab; HER2-targeted experimental (T-DM1 + pertuzumab, paclitaxel + "
                     "pertuzumab + trastuzumab, + MK-2206 + trastuzumab, + AMG 386 + trastuzumab, + neratinib); pembrolizumab; other experimental. "
                     "Internal increments are out-of-fold and were not confirmed externally. Reoriented inputs."))
    # richer clinical + kinetic variants + decision
    CR = pd.read_csv(os.path.join(R3, "rev3_richer_clinical.csv"))
    K = pd.read_csv(os.path.join(R3, "rev3_kinetic_variants.csv"))
    rows = [["\\multicolumn{5}{l}{\\emph{Richer clinical comparator (additional pretreatment variables)}}"]]
    for r in CR.itertuples():
        rows.append([r.comparison.replace("richer clinical", "richer clinical model").capitalize(), f3(r.oof_auc), f3(r.ext_auc), ci(r.ext_diff, r.ext_lo, r.ext_hi), "--"])
    kv = {"default": "As implemented (reference)", "no_smoothing": "No smoothing of the denominator", "floor_0.05": "Denominator floor 5\\% of mean",
          "floor_0.5": "Denominator floor 50\\% of mean", "ser_unclipped": "SER floor 0.01, no clipping", "mask_eroded": "Mask eroded by 1 voxel",
          "mask_dilated": "Mask dilated by 1 voxel"}
    rows.append(["\\multicolumn{5}{l}{\\emph{Clinical + kinetic radiomics under kinetic-feature variants (external $\\Delta$ vs clinical)}}"])
    for v in kv:
        r = K[K.variant == v]
        if len(r):
            r = r.iloc[0]
            rows.append([kv[v], f3(r.oof_auc), f3(r.ext_auc), ci(r.ext_diff, r.ext_lo, r.ext_hi), str(int(r.features_changed))])
    out.append(table("Richer Clinical Comparator and Sensitivity of the Kinetic Features to Implementation Choices (Round 3)", "tab:s_kinrich",
                     ["Model or variant", "I-SPY2 OOF AUC", "External AUC", "External $\\Delta$ (95\\% CI)", "Features changed"],
                     rows, "p{6.6cm}cccc",
                     "Richer clinical model: age, ER, PR, HER2, menopausal status, multifocality, BMI group and ethnicity (L2 logistic regression, missing "
                     "indicators). These variables are missing for whole collections (PR in I-SPY2; menopausal status in I-SPY1 and NACT; multifocality in "
                     "I-SPY1), which limits this comparator. Its $\\Delta$ is against the age/HR/HER2 model; the late-fusion row is against the richer model. "
                     "Features changed: kinetic features whose median absolute change exceeds $10^{-6}$."))
    D = pd.read_csv(os.path.join(R3, "rev3_decision.csv"))
    rows = [[("Clinical" if r.model == "Clinical" else NM.get(r.model, r.model)), f"{r.threshold:.1f}", f3(r.sensitivity, 2), f3(r.specificity, 2), f3(r.ppv, 2), f3(r.npv, 2),
             f3(r.net_benefit), f3(max(r.net_benefit_all, 0))] for r in D.itertuples()]
    out.append(table("Decision-Specific Evaluation at Prespecified Thresholds, External Cohort (Round 3)", "tab:s_decision",
                     ["Model", "Threshold", "Sensitivity", "Specificity", "PPV", "NPV", "Net benefit", "Default net benefit"],
                     rows, "p{4.2cm}ccccccc",
                     "Intended use stated before the run: counselling about the likelihood of pCR before standard neoadjuvant therapy, not treatment "
                     "escalation or de-escalation. The fusion model compared is the one with the highest I-SPY2 out-of-fold AUC (seed 0), fixed by the "
                     "analysis plan. Default net benefit: the better of treating all and treating none."))
    open(os.path.join(HERE, "rev3_supp_tables.tex"), "w").write("\n".join(out))


def figure():
    V = pd.read_csv(os.path.join(R3, "rev3_recovery.csv"))
    R = pd.read_csv(os.path.join(R3, "rev3_repeats_lps.csv"))
    Ro = pd.read_csv(os.path.join(R3, "rev3_repeats_orig.csv"))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 3.0), gridspec_kw={"width_ratios": [1.05, 1]})
    a = ax[0]
    a.plot([-0.01, 0.105], [-0.01, 0.105], color=GREY, lw=0.8, ls="--")
    a.axhline(0, color=GREY, lw=0.6)
    for m, col, off, lab in (("carfm_25d", BLUE, -0.0012, "CARF-mix 2.5D"), ("lf", ORANGE, 0.0012, "Late fusion")):
        v = V[(V.model == m) & (V.level != "permuted")]
        a.errorbar(v.oracle_ext + off, v.ext_diff, yerr=[v.ext_diff - v.ext_lo, v.ext_hi - v.ext_diff], fmt="o", ms=4, color=col, ecolor=col, elinewidth=1, capsize=0, label=lab)
        p = V[(V.model == m) & (V.level == "permuted")]
        a.errorbar(np.full(len(p), -0.006) + off, p.ext_diff, yerr=[p.ext_diff - p.ext_lo, p.ext_hi - p.ext_diff], fmt="s", ms=4, mfc="white", color=col, ecolor=col, elinewidth=1)
    a.text(-0.003, 0.1, "permuted\nimages", ha="center", va="top", fontsize=6.5, color=GREY)
    a.text(0.066, 0.052, "full recovery", rotation=38, fontsize=6.5, color=GREY)
    a.set_xlim(-0.012, 0.105); a.set_ylim(-0.065, 0.105)
    a.set_xlabel("Planted signal: realized external oracle gain in AUC")
    a.set_ylabel("Recovered external gain over clinical (95% CI)")
    a.set_title("a  Conditional signal recovery (post hoc)", loc="left", fontsize=8.5)
    a.legend(fontsize=7, frameon=False, loc="lower right")
    b = ax[1]
    order = ["lf", "carfm_25d", "ckin", "carfm_k", "s25d"]
    b.axvspan(-0.03, 0.03, color="#dbe8f8", alpha=0.6, lw=0)
    b.axvline(0, color=GREY, lw=0.7)
    for i, m in enumerate(order):
        y = len(order) - 1 - i
        r = R[R.model == m]
        b.scatter(r.ext_diff, np.full(len(r), y + 0.12), s=16, color=BLUE, zorder=3, label="Reoriented (seeds)" if i == 0 else None)
        for rr in r.itertuples():
            b.plot([rr.ext_lo, rr.ext_hi], [y + 0.12] * 2, color=BLUE, lw=0.5, alpha=0.35)
        o = Ro[Ro.model == m]
        if len(o):
            b.scatter(o.ext_diff, np.full(len(o), y - 0.12), s=16, facecolor="white", edgecolor=INK, lw=0.8, zorder=3, label="Original orientation (seeds)" if i == 0 else None)
    b.set_yticks(range(len(order))); b.set_yticklabels([NM[m].replace(", ", ",\n") for m in order][::-1], fontsize=7)
    b.set_xlabel("External AUC difference from clinical model")
    b.set_title("b  Repeated training (post hoc)", loc="left", fontsize=8.5)
    b.text(0.0, len(order) - 0.45, "$\\pm$0.03 margin", ha="center", fontsize=6.5, color=BLUE)
    b.legend(fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    b.set_ylim(-0.6, len(order) - 0.3)
    for x in ax:
        for sp in ("top", "right"):
            x.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_rev3.pdf")); fig.savefig(os.path.join(FIG, "fig_rev3.png"), dpi=300)
    plt.close(fig)


RELABEL = [("Clinical + FTV", "Clinical + PETV"), ("clinical + FTV", "clinical + PETV"), ("Whole-breast 3D", "Larger-context 3D"),
           ("Whole-breast", "Larger-context"), ("important loss", "loss $\\ge m$"), ("important gain", "gain $\\ge m$"),
           ("Statistical and Clinical-Importance Questions", "Statistical and Margin-Based Questions"), ("Clinical importance: equivalent", "Prespecified margin: equivalent")]


def relabel():
    """Third-review terminology applied to the tables generated in earlier rounds (idempotent)."""
    import glob
    for f in glob.glob(os.path.join(HERE, "rev_*.tex")) + [os.path.join(HERE, "supp_tables.tex")]:
        if f.endswith("rev_tab_prior.tex"):
            continue
        t = open(f).read()
        for a, b in RELABEL:
            t = t.replace(a, b)
        open(f, "w").write(t)


if __name__ == "__main__":
    relabel()
    main_tables()
    supp_tables()
    figure()
    print("ok")
