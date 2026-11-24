"""Insert the abstract and the bibliography (ordered by first citation, uncited entries dropped) into main.tex.

Edit main.tex itself for text; keep %ABSTRACT% / %BIB% markers in main_src.tex. Run: python paper/srep/assemble.py
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "main_src.tex"), encoding="utf-8").read()
abstract = open(os.path.join(HERE, "abstract.txt"), encoding="utf-8").read().strip()
bib = open(os.path.join(HERE, "bib_srep.tex"), encoding="utf-8").read()
items = {re.match(r"\\bibitem\{([^}]*)\}", it).group(1): it.strip()
         for it in re.split(r"(?=\\bibitem\{)", re.search(r"\\begin\{thebibliography\}\{99\}(.*)\\end\{thebibliography\}", bib, re.S).group(1)) if it.strip()}
body = src.replace("%ABSTRACT%", abstract)
order = []
for c in re.findall(r"\\cite\{([^}]*)\}", body):
    for k in c.split(","):
        k = k.strip()
        if k not in order:
            order.append(k)
missing = [k for k in order if k not in items]
assert not missing, missing
unused = [k for k in items if k not in order]
out = "\\begin{thebibliography}{99}\n" + "\n".join(items[k] for k in order) + "\n\\end{thebibliography}"
body = body.replace("%BIB%", out)
open(os.path.join(HERE, "main.tex"), "w", encoding="utf-8").write(body)
print(f"{len(order)} references; unused: {unused}")
