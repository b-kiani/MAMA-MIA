"""Supplementary Information for the Scientific Reports version, from the JBHI supplement (paper/jbhi/supp.tex).

Changes: article class on A4 with title and authors, cross-references to the main text by section name,
checklist locations for the Scientific Reports structure. Run paper/jbhi/build_supp4.py first.

    python paper/srep/build_supp_srep.py
"""
import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
JB = os.path.join(os.path.dirname(HERE), "jbhi")
s = open(os.path.join(JB, "supp.tex"), encoding="utf-8").read()
exec(open(os.path.join(HERE, "checklist_map_srep.py")).read())

TITLE = ("Pretreatment breast MRI models did not consistently add to receptor status for predicting pathologic complete response "
         "in held-out collections")
pre_old = s[:s.index("\\maketitle") + len("\\maketitle")]
pre_new = r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=1.6cm]{geometry}
\usepackage{adjustbox}
\usepackage{cite}
\usepackage{amsmath,amssymb}
\usepackage{multirow}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{url}
\usepackage[hidelinks]{hyperref}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thefigure}{S\arabic{figure}}
\renewcommand{\thealgorithm}{S\arabic{algorithm}}
\renewcommand{\theequation}{S\arabic{equation}}
\renewcommand{\algorithmicrequire}{\textbf{Input:}}
\renewcommand{\algorithmicensure}{\textbf{Output:}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}
\sloppy

\begin{document}
\begin{center}
{\Large\bfseries Supplementary Information}\\[8pt]
{\large """ + TITLE + r"""}\\[8pt]
Behnam Kiani Kalejahi$^{1,2,*}$ and Mohammad Javad Rajabi$^{2}$\\[4pt]
{\small $^{1}$Department of Computer Science, School of Engineering, Central Asian University, Tashkent, Uzbekistan.
$^{2}$Faculty of Data Science and Information Technology, INTI International University, Nilai, Negeri Sembilan, Malaysia.
$^{*}$Corresponding author: b.kiani@centralasian.uz}
\end{center}
\vspace{6pt}
"""
s = s.replace(pre_old, pre_new)

# references to the main text
for a, b in [("(main text, Section III-B)", "(main text, Methods, Analysis stages and reuse of the external collections)"),
             ("given in main Table~I", "given in Table~1 of the main text"),
             ("for the example cases in main Fig.~2", "for the example cases in Fig.~\\ref{fig:samples}"),
             ("Locations refer to sections (Roman numeral and letter), tables (Roman numerals) and figures of the restructured main manuscript",
              "Locations refer to sections of the main text (R, Results; M, Methods, with subsection names), its tables and figures"),
             ]:
    assert a in s, a
    s = s.replace(a, b)


def relocate(block, mapping):
    def f(m):
        item, topic = m.group(1), m.group(2)
        return f"{item} & {topic} & {mapping[item]}\\\\" if item in mapping else m.group(0)
    return re.sub(r"(?m)^(\d+[a-g]?) & ([^&\n]*) & [^\n]*\\\\$", f, block)


i0, i1 = s.index("\\section*{S5. Reporting Checklists}"), s.index("\\section*{S6.")
s5 = s[i0:i1]
j = s5.index("\\label{tab:s_claim}")
s = s[:i0] + relocate(s5[:j], TRIPOD) + relocate(s5[j:], CLAIM) + s[i1:]
s = s.replace("this supplement", "this Supplementary Information")
open(os.path.join(HERE, "supp.tex"), "w", encoding="utf-8").write(s)

# inputs and figures used by the supplement
def fit(t):
    """Scale a tabular down to the text width when it is wider (adjustbox max width)."""
    t = re.sub(r"\\begin\{tabular\}", r"\\begin{adjustbox}{max width=\\textwidth}\n\\begin{tabular}", t)
    return re.sub(r"\\end\{tabular\}", r"\\end{tabular}\n\\end{adjustbox}", t)


s = fit(s)
open(os.path.join(HERE, "supp.tex"), "w", encoding="utf-8").write(s)
for f in set(re.findall(r"\\input\{([^}]*)\}", s)):
    open(os.path.join(HERE, f), "w", encoding="utf-8").write(fit(open(os.path.join(JB, f), encoding="utf-8").read()))
for f in set(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]*)\}", s)):
    src = os.path.join(JB, f)
    if not os.path.exists(src):
        src = os.path.join(JB, f + ".pdf"); f = f + ".pdf"
    os.makedirs(os.path.dirname(os.path.join(HERE, f)), exist_ok=True)
    shutil.copy(src, os.path.join(HERE, f))
print("supp.tex written")
