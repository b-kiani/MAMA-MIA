"""Assemble the supplement of the restructured manuscript (fourth review).

Base: supp_r3.tex (S1-S5). Adds S6 (fourth-review analyses) and S7 (Methods and Results of the previous
main text, i.e. the historical analysis with the distributed labels, from main_r3.tex), updates the title
and the checklist locations, and merges the bibliography.

    python paper/jbhi/build_supp4.py
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SCR = os.environ.get("CHK_MAP")
rd = lambda f: open(os.path.join(HERE, f), encoding="utf-8").read()

base = rd("supp_r3.tex")
main3 = rd("main_r3.tex")
main4 = rd("main.tex")

# ---------------------------------------------------------------- preamble and title
base = base.replace("\\usepackage{amsmath,amssymb}", "\\usepackage{cite}\n\\usepackage{amsmath,amssymb}\n\\usepackage{multirow}")
base = re.sub(r"\\title\{Supplementary Material for ``.*?''\}",
              "\\\\title{Supplementary Material for ``Incremental Value of Pretreatment Breast MRI Beyond Receptor Status in a Multi-Collection Benchmark''}",
              base, flags=re.S)

# ---------------------------------------------------------------- checklist locations
exec(open(os.environ.get("CHK_MAP_FILE", os.path.join(HERE, "checklist_map4.py"))).read())


def relocate(block, mapping):
    def f(m):
        item, topic = m.group(1), m.group(2)
        return f"{item} & {topic} & {mapping[item]}\\\\" if item in mapping else m.group(0)
    return re.sub(r"(?m)^(\d+[a-g]?) & ([^&\n]*) & [^\n]*\\\\$", f, block)


i0, i1 = base.index("\\section*{S5. Reporting Checklists}"), base.index("\\begin{thebibliography}")
s5 = base[i0:i1]
j = s5.index("\\label{tab:s_claim}")
s5 = relocate(s5[:j], TRIPOD) + relocate(s5[j:], CLAIM)
s5 = s5.replace("Locations refer to sections (Roman numeral and letter), tables and figures of the main manuscript",
                "Locations refer to sections (Roman numeral and letter), tables (Roman numerals) and figures of the restructured main manuscript")

# ---------------------------------------------------------------- S6
s6 = r"""
\clearpage
\section*{S6. Fourth-Review Analyses (Post Hoc)}
These analyses were planned in \texttt{paper/analysis\_plan\_revision4.md} and committed before they were run (commit \texttt{a6a7389}). They comprise (i)~rescoring every locked external prediction against the harmonized pathologic end point (Section~S4; Table~\ref{tab:s_outsens} gives the earlier sensitivity analysis), (ii)~fully nested stacking for late fusion, retrained over five seeds and with each regimen group of I-SPY2 held out (Kaggle notebook version~9), and (iii)~an extension of the signal-recovery control with a model given the planted signal directly and the same refitted clinical comparator for oracle and recovered gains. No model was retrained for (i); the predictions are those of the earlier runs, and only the labels differ. All results are exploratory and reuse external patients examined in earlier rounds.

\emph{Nested stacking.} For each outer fold $k$ of I-SPY2, the 2.5D image-only network was trained by fivefold cross-validation of the outer training partition $\mathcal{T}_k$ only, giving inner out-of-fold image logits for every patient in $\mathcal{T}_k$; the clinical model was cross-fitted on the same inner folds. The stacker (logistic regression on the two logits) was fitted on these inner predictions and applied to the outer test fold, whose image logit was the mean of the five inner networks and whose clinical logit came from the clinical model fitted on $\mathcal{T}_k$. No base model behind a stacker's training input had seen that stacker's test fold. External predictions average the five outer stackers, each applied to its own inner networks and clinical model. In the earlier version, the stacker was cross-fitted on out-of-fold predictions produced over all of I-SPY2, so the networks that produced a stacker's training inputs had been trained on its test fold; that affected internal, not external, estimates. Nested late fusion was trained over five seeds (internal gain 0.033 to 0.046; external difference 0.000 to 0.011 on the harmonized end point) and with each regimen group held out (Table~\ref{tab:s_regimen4}); trained on the regimen-group splits, its external difference was 0.007 ($-0.012$ to $0.027$). Its statistics were computed from the decoded predictions with the same methods as the other results (DeLong differences, calibration by logistic regression, 2,000 outcome-stratified bootstrap resamples for AUC intervals, with a different random stream); the results are in \texttt{results/revision4/lfnest}.

\input{rev4_supp_matrix.tex}
\input{rev4_supp_seeds.tex}
\input{rev4_supp_regimen.tex}
\input{rev4_supp_detect.tex}
"""

# ---------------------------------------------------------------- S7 from main_r3
a = main3.index("\\section{Materials and Methods}")
b = main3.index("\\section{Discussion}")
body = main3[a:b]
pm = re.search(r"\\begin\{table\}.*?\\end\{table\}", rd("rev_tab_prior.tex"), re.S)
labels, out, sec_i, sub_i = {}, [], 0, 0
TITLES = {"Materials and Methods": "Methods (Previous Version)", "Results": "Results (Previous Version, Distributed Labels)"}
for line in body.split("\n"):
    m = re.match(r"\\section\{(.*)\}", line)
    if m:
        sec_i += 1; sub_i = 0; cur = f"S7.{sec_i}"
        out.append(f"\\subsection*{{{cur}. {TITLES.get(m.group(1), m.group(1))}}}")
        continue
    m = re.match(r"\\subsection\{(.*)\}", line)
    if m:
        sub_i += 1; cur = f"S7.{sec_i}.{sub_i}"
        out.append(f"\\subsubsection*{{{cur}. {m.group(1)}}}")
        continue
    m = re.match(r"\\label\{(sec:[^}]*)\}\s*$", line)
    if m:
        labels[m.group(1)] = cur
        continue
    out.append(line)
s7 = "\n".join(out)
s7 = re.sub(r"\\ref\{(sec:[^}]*)\}", lambda m: labels[m.group(1)], s7)
for x, y in [("Supplementary Tables~", "Tables~"), ("Supplementary Table~", "Table~"), ("Supplementary Section~", "Section~"),
             ("Supplementary Fig.~", "Fig.~"), ("Supplementary Material", "this supplement")]:
    s7 = s7.replace(x, y)
# two statements superseded by the fourth review (narrowed wording)
fix = [("so that it is an MRI-derived signal unrelated to the clinical variables", "that is, linearly residualized on age, HR and HER2"),
       ("The pipeline is therefore not sensitive enough to detect conditional imaging gains of the size of the margin; it detects gains of about 0.06 or more.",
        "The largest tested planted signal was recovered in both outcome draws, whereas smaller signals were not reliably recovered."),
       ("The pipeline recovered a planted conditional signal only when it was large", "The pipeline recovered the planted conditional signal reliably only at the largest level")]
for x, y in fix:
    assert x in s7, x
    s7 = s7.replace(x, y)
s7 = (r"""
\clearpage
\section*{S7. Historical Analysis: Methods and Results of the Previous Version}
This section reproduces the Methods and Results of the previous version of the manuscript, which analyzed the outcome labels as distributed by MAMA-MIA (511 external patients, 124 with pCR). It is the record of the prespecified analysis (round~1) and of the exploratory rounds~2 and 3, including the analysis timeline (Table~\ref{tab:timeline}), the full model matrix (Table~\ref{tab:all}), the per-collection results, the checks of the imaging pipelines, the ablations, calibration and decision curves, and the multi-source validation. The text is unchanged except for cross-references and three statements, narrowed as in the main manuscript: the planted signal is described as linearly residualized on age, HR and HER2, and the signal-recovery results no longer state a detection threshold. Internal late-fusion estimates in this section used stacking that was not fully nested; nested results are in Section~S6. References to the main manuscript in this section are to the previous version.

""" + s7 + "\n\\subsubsection*{Comparison with Previous Work}\n" + pm.group(0) + "\n")

# ---------------------------------------------------------------- bibliography: merge main items
bm = re.search(r"\\begin\{thebibliography\}\{99\}\n\\footnotesize\n(.*?)\n\\end\{thebibliography\}", main3 + "\n" + main4, re.S)
items = {}
for src in (main3, main4):
    m = re.search(r"\\begin\{thebibliography\}\{99\}\n\\footnotesize\n(.*?)\n\\end\{thebibliography\}", src, re.S)
    for it in re.split(r"(?=\\bibitem\{)", m.group(1).strip()):
        if it.strip():
            items.setdefault(re.match(r"\\bibitem\{([^}]*)\}", it).group(1), it.strip())
bib = base[i1:]
own = re.search(r"\\footnotesize\n(.*?)\n\\end\{thebibliography\}", bib, re.S).group(1)
text = base[:i0] + s5 + s6 + s7
cited = []
for c in re.findall(r"\\cite\{([^}]*)\}", text + "".join(rd(f) for f in re.findall(r"\\input\{([^}]*)\}", text) if os.path.exists(os.path.join(HERE, f)))):
    for k in c.split(","):
        k = k.strip()
        if k not in cited:
            cited.append(k)
ownkeys = re.findall(r"\\bibitem\{([^}]*)\}", own)
extra = [items[k] for k in cited if k not in ownkeys and k in items]
missing = [k for k in cited if k not in ownkeys and k not in items]
assert not missing, missing
bib = bib.replace("\\begin{thebibliography}{9}", "\\begin{thebibliography}{99}").replace(own, own + "\n" + "\n".join(extra))
open(os.path.join(HERE, "supp.tex"), "w", encoding="utf-8").write(text + bib)
print("supp.tex written;", len(extra), "main references added; S7 labels:", labels)
