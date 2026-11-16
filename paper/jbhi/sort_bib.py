"""Reorder the bibliography of main.tex by first citation (IEEE style), expanding \\input files."""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, "main.tex")
s = open(p, encoding="utf-8").read()
m = re.search(r"(\\begin\{thebibliography\}\{99\}\n\\footnotesize\n)(.*?)(\n\\end\{thebibliography\})", s, re.S)
items = re.split(r"(?=\\bibitem\{)", m.group(2).strip())
items = {re.match(r"\\bibitem\{([^}]*)\}", it).group(1): it.strip() for it in items if it.strip()}
body = s[:m.start()]
body = re.sub(r"\\input\{([^}]*)\}", lambda k: open(os.path.join(HERE, k.group(1)), encoding="utf-8").read(), body)
order = []
for c in re.findall(r"\\cite\{([^}]*)\}", body):
    for k in c.split(","):
        k = k.strip()
        if k not in order:
            order.append(k)
missing = [k for k in order if k not in items]
unused = [k for k in items if k not in order]
assert not missing, missing
s = s[:m.start()] + m.group(1) + "\n".join(items[k] for k in order) + m.group(3) + s[m.end():]
open(p, "w", encoding="utf-8").write(s)
print(len(order), "references; uncited and dropped:", unused)
