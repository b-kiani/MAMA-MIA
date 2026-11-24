"""Tables of the Scientific Reports version, from the same results as the JBHI version (results/revision4).

Reuses paper/jbhi/make_rev4_tex.py and adapts the output to the Springer Nature template: sentence-case
captions, table legends below the table, wide tables set sideways, exact two-sided P values in Table 2.

    python paper/srep/make_srep_tables.py
"""
import importlib.util
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
JB = os.path.join(os.path.dirname(HERE), "jbhi")
spec = importlib.util.spec_from_file_location("m4", os.path.join(JB, "make_rev4_tex.py"))
m4 = importlib.util.module_from_spec(spec)
sys.modules["m4"] = m4
spec.loader.exec_module(m4)
f3, ci, C = m4.f3, m4.ci, m4.C

CAPTIONS = {
    "tab:cohort": "Patients, outcome definitions, treatment and acquisition by collection",
    "tab:main": "External discrimination and calibration of the clinical model and representative imaging-based models on the harmonized pathologic end point",
    "tab:internal": "Gains over the clinical model within the development trial and in the external collections",
    "tab:recovery": "Recovery of a planted conditional imaging signal by a model given the signal directly and by the imaging pipelines",
}
TEXTWIDTH_CM = 12.9


def pval(d, lo, hi):
    se = (hi - lo) / (2 * norm.ppf(0.975))
    return 2 * norm.sf(abs(d) / se) if se > 0 else np.nan


def fp(p):
    if pd.isna(p):
        return "--"
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def env(t, label, sideways):
    t = re.sub(r"\\caption\{.*?\}\n\\label", lambda _: "\\caption{" + CAPTIONS[label] + "}\n\\label", t, count=1, flags=re.S)
    if sideways:
        t = t.replace("\\begin{table}[!t]", "\\begin{sidewaystable}").replace("\\end{table}", "\\end{sidewaystable}")
    else:
        t = t.replace("\\begin{table}[!t]", "\\begin{table}[!htbp]")
    return t


def scale_widths(t, total):
    """Scale all p{..cm} column widths of the tabular so that they fit the text width."""
    m = re.search(r"\\begin\{tabular\}\{(.*)\}\n\\toprule", t)
    spec = m.group(1)
    ws = [float(x) for x in re.findall(r"p\{([\d.]+)cm\}", spec)]
    ncol = spec.count("p{") + len(re.findall(r"(?<![{a-z])c(?![a-z])", re.sub(r"\{[^{}]*\}", "", spec)))
    k = (total - 0.6 - 0.21 * ncol) / sum(ws)
    new = re.sub(r"p\{([\d.]+)cm\}", lambda x: f"p{{{float(x.group(1)) * k:.2f}cm}}", spec)
    return t.replace(spec, new)


def main_table(M, S, L):
    rows = []
    mark = {"ref": "", "c": "$^{a}$", "2": "$^{b}$", "3": "$^{c}$"}
    for r in m4.rep_rows(M, S, L, "harmonized"):
        if r["stage"] == "ref":
            rows.append([r["name"], r["seeds"], ci(r["auc"], r["auc_lo"], r["auc_hi"]), "reference", "--", "--",
                         *[f"\\emph{{{f3(r[f'a_{c}'])}}}" for c in m4.EXT], f3(r['cal_int'], 2), f3(r['cal_slope'], 2)])
            continue
        rng = "--" if pd.isna(r["dmin"]) else ("identical" if abs(r["dmax"] - r["dmin"]) < 1e-9 else f"{f3(r['dmin'])} to {f3(r['dmax'])}")
        wid = f"{f3(r['wlo'])} to {f3(r['whi'])}"
        rows.append([r["name"] + mark[r["stage"]], r["seeds"], ci(r["auc"], r["auc_lo"], r["auc_hi"]), ci(r["diff"], r["lo"], r["hi"]),
                     fp(pval(r["diff"], r["lo"], r["hi"])), rng + (f"; {wid}" if r["seeds"] not in ("1", "--") else ""),
                     f3(r["d_DUKE"]), f3(r["d_ISPY1"]), f3(r["d_NACT"]), f3(r['cal_int'], 2), f3(r['cal_slope'], 2)])
    t = m4.table("x", "tab:main", ["Model", "Seeds", "External AUC (95\\% CI)", "$\\Delta$ vs clinical, seed 0 (95\\% CI)", "$P$",
                                   "$\\Delta$ range over seeds; widest 95\\% CI", "Duke $\\Delta$", "I-SPY1 $\\Delta$", "NACT $\\Delta$",
                                   "Cal. intercept", "Cal. slope"],
                 rows, ">{\\raggedright\\arraybackslash}p{2.8cm}c" + C(2.4, 2.4, 0.9, 2.8, 1.0, 1.0, 1.0, 1.1, 1.0),
                 "509 external patients, 131 with pCR (harmonized end point: no invasive cancer in breast and nodes, in situ disease allowed). "
                 "Clinical row: AUC within each collection in italics. $\\Delta$: paired difference in AUC from the clinical model, with 95\\% CI and "
                 "two-sided $P$ value from DeLong's test (unadjusted). For models with several seeds (outer folds, inner split and initialization "
                 "changed), the AUC, $\\Delta$ and $P$ are for seed 0, the range is over seeds, and the per-collection $\\Delta$ and calibration are "
                 "means over seeds. A calibration intercept below 0 means predicted probabilities that are too high; a slope above 1, predictions "
                 "that are too moderate. $^{a}$Prespecified confirmatory comparison (round~1, original inputs); historical analysis on the distributed "
                 "labels in Supplementary Table~\\ref{S-tab:s_matrix}. $^{b}$Round~2 (exploratory). $^{c}$Round~3 and later (post hoc), reoriented "
                 "inputs and nested anchors. All rows are rescorings of locked predictions; the external patients had been examined in earlier rounds.")
    open(os.path.join(HERE, "srep_tab_main.tex"), "w").write(env(t, "tab:main", True))


if __name__ == "__main__":
    M, S, L = m4.load()
    main_table(M, S, L)
    for src, dst, label, side in [("rev4_tab_cohort.tex", "srep_tab_cohort.tex", "tab:cohort", True),
                                  ("rev4_tab_internal.tex", "srep_tab_internal.tex", "tab:internal", False),
                                  ("rev4_tab_recovery.tex", "srep_tab_recovery.tex", "tab:recovery", False)]:
        t = open(os.path.join(JB, src)).read()
        t = env(t, label, side)
        if not side:
            t = scale_widths(t, TEXTWIDTH_CM)
        open(os.path.join(HERE, dst), "w").write(t)
    print("ok")
