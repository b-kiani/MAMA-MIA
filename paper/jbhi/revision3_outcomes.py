"""Patient-level outcome verification against the source collections, and an outcome-harmonized sensitivity analysis.

    python paper/jbhi/revision3_outcomes.py  ->  results/revision3/rev3_outcome_check.csv,
                                                 rev3_outcome_sensitivity.csv, paper/jbhi/rev3_outcome_tables.tex

Source extracts (results/revision3/source_outcomes/) were read from the TCIA clinical files:
  I-SPY2  ISPY2-Imaging-Cohort-1-Clinical-Data.xlsx (column pCR; compared directly in the browser: 980/980 agree,
          recorded in ispy2_check.json)
  I-SPY1  I-SPY-1-All-Patient-Clinical-and-Outcome-Data.xlsx, sheet of outcome variables (PCR, RCBClass)
  Duke    Clinical_and_Other_Features.xlsx (Pathologic Response to Neoadjuvant Therapy; ypT; ypN; near-complete response)
  NACT    SharedClinicalAndRFS.xls (path size, LN, Clinical response); the file has no pCR field
Harmonized definition (no invasive cancer in breast and nodes, in situ disease allowed; ypT0/is ypN0):
  Duke: complete response, or only DCIS or LCIS remaining; NACT: pathologic size 0 and no positive node.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
from mmonc import stats  # noqa: E402

SRC = os.path.join(ROOT, "results", "revision3", "source_outcomes")
OUT = os.path.join(ROOT, "results", "revision3")
EXT = ("DUKE", "ISPY1", "NACT")


def f3(x):
    return "--" if pd.isna(x) else f"{x + 0.0:.3f}".replace("-0.000", "0.000").replace("-", "$-$")


def main():
    man = pd.read_csv(os.path.join(ROOT, "results", "revision", "case_manifest.csv"))
    rec = man.set_index("case_id").pcr_as_recorded
    rows, harm = [], {}
    # I-SPY1
    s = pd.read_csv(os.path.join(SRC, "ispy1_source.csv"))
    s.index = "ISPY1_" + s.subject_id.astype(str)
    for c in man[man.collection == "ISPY1"].case_id:
        rows.append({"case_id": c, "collection": "ISPY1", "recorded": rec[c], "source_field": "PCR", "source_value": s.pcr_source.get(c, np.nan),
                     "harmonized": s.pcr_source.get(c, np.nan)})
    # Duke
    d = pd.read_csv(os.path.join(SRC, "duke_source.csv"), dtype=str)
    d.index = "DUKE_" + d.patient.str.zfill(3)
    for c in man[man.collection == "DUKE"].case_id:
        r = d.loc[c] if c in d.index else None
        pr = r.path_response if r is not None else np.nan
        h = np.nan if pr in ("5", np.nan) or pd.isna(pr) else (1.0 if pr in ("1", "3", "4") else 0.0)
        rows.append({"case_id": c, "collection": "DUKE", "recorded": rec[c], "source_field": "Pathologic response to neoadjuvant therapy",
                     "source_value": pr, "ypT": r.ypT if r is not None else np.nan, "ypN": r.ypN if r is not None else np.nan, "harmonized": h})
    # NACT
    n = pd.read_csv(os.path.join(SRC, "nact_source.csv"))
    n.index = "NACT_" + n.primary_id.astype(int).astype(str).str.zfill(2)
    for c in man[man.collection == "NACT"].case_id:
        r = n.loc[c]
        h = np.nan if pd.isna(r.path_size_cm) or pd.isna(r.ln_positive) else float(r.path_size_cm == 0 and r.ln_positive == 0)
        rows.append({"case_id": c, "collection": "NACT", "recorded": rec[c], "source_field": "Clinical response (1 = no evidence of disease); path size; LN",
                     "source_value": r.clinical_response, "path_size_cm": r.path_size_cm, "ln_positive": r.ln_positive, "harmonized": h})
    C = pd.DataFrame(rows)
    C.to_csv(os.path.join(OUT, "rev3_outcome_check.csv"), index=False)
    ispy2 = json.load(open(os.path.join(SRC, "ispy2_check.json")))
    N = {"ispy2": ispy2}
    for k, g in C.groupby("collection"):
        both = g.dropna(subset=["recorded", "harmonized"])
        N[k] = {"n": len(g), "recorded_events": int(g.recorded.sum()), "harmonized_events": int(g.harmonized.sum()),
                "harmonized_missing": int(g.harmonized.isna().sum()), "agree": int((both.recorded == both.harmonized).sum()), "compared": len(both),
                "recorded1_harm0": int(((both.recorded == 1) & (both.harmonized == 0)).sum()), "recorded0_harm1": int(((both.recorded == 0) & (both.harmonized == 1)).sum())}
    nn = C[C.collection == "NACT"]
    N["NACT"]["recorded_equals_clinical_ned"] = bool(((nn.source_value == 1).astype(float) == nn.recorded).all())
    dd = C[C.collection == "DUKE"]
    N["DUKE"]["recorded_by_category"] = {str(k): v for k, v in dd.groupby(dd.source_value.fillna("NA")).recorded.agg(lambda s: f"{int((s == 1).sum())}/{int((s == 0).sum())}/{int(s.isna().sum())}").items()}
    # sensitivity: external results with harmonized labels
    P = pd.read_csv(os.path.join(ROOT, "results", "revision", "rev_all_predictions.csv"))
    E = P[P.set == "ext"].copy()
    E = E.merge(C[["case_id", "harmonized"]], on="case_id", how="left")
    r3 = os.path.join(ROOT, "results", "revision3", "raw", "preds")
    S = pd.concat([pd.read_csv(f) for f in [os.path.join(ROOT, "results", "revision3", "raw", "runs_g0.csv"), os.path.join(ROOT, "results", "revision3", "raw", "runs_g1.csv")]]).set_index("run_id")
    sys.path.insert(0, HERE)
    from revision3_analyses import late_fusion  # noqa: E402
    for rid, col in (("s25d|lps|s0|real|clinkin|all|cv", "Late fusion, 2.5D (reoriented)"), ("carfm_25d|lps|s0|real|clinkin|all|cv", "CARF-mix, 2.5D (reoriented)")):
        D = pd.read_csv(os.path.join(r3, S.loc[rid, "file"]))
        v = late_fusion(D) if rid.startswith("s25d") else D.p.to_numpy()
        E = E.merge(pd.DataFrame({"case_id": D.case_id, col: v})[D.set == "ext"], on="case_id", how="left")
    models = [("R1: CARF (proposed)", "CARF, 3D (primary)"), ("CARF, kinetic", "CARF, kinetic 3D"), ("CARF-mix, kinetic", "CARF-mix, kinetic 3D"),
              ("CARF-mix, pretrained 2.5D", "CARF-mix, pretrained 2.5D"), ("Clinical + FTV", "Clinical + PETV"), ("Clinical + kinetic radiomics", "Clinical + kinetic radiomics"),
              ("Late fusion, 2.5D (reoriented)", "Late fusion, 2.5D (round 3)"), ("CARF-mix, 2.5D (reoriented)", "CARF-mix, 2.5D (round 3, reoriented)"),
              ("Image only, pretrained 2.5D", "Image only, pretrained 2.5D")]
    out = []
    for lab, ycol, mask in (("As recorded", "y", None), ("Harmonized (ypT0/is ypN0)", "harmonized", None), ("Harmonized, without NACT", "harmonized", "noNACT")):
        X = E.dropna(subset=[ycol]) if mask is None else E[(E.collection != "NACT")].dropna(subset=[ycol])
        y = X[ycol].to_numpy().astype(int)
        base = {"labels": lab, "n": len(X), "events": int(y.sum()), "clinical_auc": roc_auc_score(y, X.Clinical)}
        for m, nm in models:
            if m not in X or X[m].isna().any():
                continue
            dl = stats.delong(y, X.Clinical.to_numpy(), X[m].to_numpy())
            r = {**base, "model": nm, "auc": roc_auc_score(y, X[m]), "diff": dl["diff"], "lo": dl["ci_lo"], "hi": dl["ci_hi"]}
            for c in ("DUKE", "NACT"):
                k = (X.collection == c).to_numpy()
                if k.sum() and len(np.unique(y[k])) > 1:
                    dc = stats.delong(y[k], X.Clinical.to_numpy()[k], X[m].to_numpy()[k])
                    r.update({f"diff_{c}": dc["diff"], f"lo_{c}": dc["ci_lo"], f"hi_{c}": dc["ci_hi"], f"clin_{c}": roc_auc_score(y[k], X.Clinical.to_numpy()[k])})
            out.append(r)
    R = pd.DataFrame(out)
    R.to_csv(os.path.join(OUT, "rev3_outcome_sensitivity.csv"), index=False)
    N["sensitivity_clinical"] = R.groupby("labels").agg(n=("n", "first"), events=("events", "first"), clinical_auc=("clinical_auc", "first")).round(4).reset_index().to_dict(orient="records")
    json.dump(N, open(os.path.join(HERE, "rev3_outcome_numbers.json"), "w"), indent=1, default=str)
    # LaTeX
    rows = []
    for lab in R.labels.unique():
        x = R[R.labels == lab]
        rows.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{lab}}}: {x.n.iloc[0]} patients, {x.events.iloc[0]} with pCR; clinical model AUC {f3(x.clinical_auc.iloc[0])}}}\\\\")
        for r in x.itertuples():
            rows.append(f"{r.model} & {f3(r.auc)} & {f3(r.diff)} ({f3(r.lo)} to {f3(r.hi)}) & {f3(r.diff_DUKE)} ({f3(r.lo_DUKE)} to {f3(r.hi_DUKE)}) & "
                        + (f"{f3(r.diff_NACT)} ({f3(r.lo_NACT)} to {f3(r.hi_NACT)})" if pd.notna(getattr(r, 'diff_NACT', np.nan)) else "--") + "\\\\")
    t = ["\\begin{table}[!htbp]", "\\caption{External Results With Outcome Labels Harmonized to the Source Records (Round 3)}", "\\label{tab:s_outsens}",
         "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
         "\\begin{tabular}{>{\\raggedright\\arraybackslash}p{4.4cm}c>{\\centering\\arraybackslash}p{3.0cm}>{\\centering\\arraybackslash}p{3.0cm}>{\\centering\\arraybackslash}p{3.0cm}}",
         "\\toprule", "Model & External AUC & Pooled $\\Delta$ vs clinical (95\\% CI) & Duke $\\Delta$ & NACT $\\Delta$\\\\", "\\midrule", *rows, "\\bottomrule",
         "\\multicolumn{5}{p{0.95\\linewidth}}{\\scriptsize Harmonized: no invasive cancer in breast and nodes, in situ disease allowed (the I-SPY definition). "
         "Duke: 12 patients with only DCIS or LCIS remaining relabeled as pCR. NACT: the recorded label equals the clinical response category "
         "``no evidence of disease''; the harmonized label is pathologic size 0 with no positive node (2 patients without pathology excluded). "
         "I-SPY1 and I-SPY2 labels agreed with the source in every patient. Predictions are unchanged; only the labels differ.}\\\\",
         "\\end{tabular}", "\\end{table}", ""]
    open(os.path.join(HERE, "rev3_outcome_tables.tex"), "w").write("\n".join(t))
    print(json.dumps(N, indent=1, default=str)[:3000])
    print(R[["labels", "model", "auc", "diff", "lo", "hi", "diff_DUKE", "diff_NACT"]].round(3).to_string())


if __name__ == "__main__":
    main()
