"""Convert the JBHI LaTeX manuscript (main.tex, supp.tex) to Word with IEEE-like layout.

    python paper/jbhi/to_word.py

Steps: resolve citations, cross-references and figure paths in a copy of the LaTeX; convert with
pandoc (equations become native Word equations); then lay the document out in Word XML: US Letter,
IEEE margins, Times New Roman, title block in one column, body in two columns, full-width figures and
tables in one-column sections.
"""
import copy
import os
import re
import shutil
import subprocess
import tempfile
import zipfile

from lxml import etree

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_SRC = {"fig3_roc_calibration_dca": "../figures", "fig2_auc_internal_external": "../figures",
           "fig4_delta_auc_ablation": "../figures", "fig5_subgroups": "../figures", "figS2_gate": "../figures"}
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W, "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
      "a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
EMU_IN = 914400
ONECOL = True      # single-column manuscript: 1-inch margins, 11-pt type, figures and tables at text width
TEXT_IN = 6.5


def q(tag):
    p, t = tag.split(":")
    return f"{{{NS[p]}}}{t}"


# ------------------------------------------------------------------ LaTeX pre-processing
def preprocess(tex, prefix=""):
    tex = re.sub(r"(?m)^%.*$", "", tex)
    # numbering of floats, equations and algorithms, in order of appearance
    labels, n = {}, {"figure": 0, "table": 0, "eq": 0, "alg": 0}
    roman = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]
    for m in re.finditer(r"\\begin\{(figure\*?|table\*?|equation|align|algorithm)\}(.*?)\\end\{\1\}", tex, re.S):
        env, body = m.group(1).rstrip("*"), m.group(2)
        if env in ("equation", "align"):
            rows = [r for r in re.split(r"\\\\", body) if r.strip()] if env == "align" else [body]
            for r in rows:
                n["eq"] += 1
                lm = re.search(r"\\label\{([^}]*)\}", r)
                if lm:
                    labels[lm.group(1)] = str(n["eq"])
            continue
        key = {"figure": "figure", "table": "table", "algorithm": "alg"}[env]
        n[key] += 1
        lm = re.search(r"\\label\{([^}]*)\}", body)
        if lm:
            labels[lm.group(1)] = (prefix + roman[n[key] - 1]) if key == "table" and not prefix else f"{prefix}{n[key]}"
    if not prefix:
        sec = sub = 0
        out, pos = [], 0
        for m in re.finditer(r"\\(section|subsection)\{([^}]*)\}(\s*\\label\{([^}]*)\})?", tex):
            if m.group(1) == "section":
                sec, sub = sec + 1, 0
                num, title = roman[sec - 1], f"{roman[sec - 1]}. {m.group(2)}"
            else:
                sub += 1
                num, title = f"{roman[sec - 1]}-{chr(64 + sub)}", f"{chr(64 + sub)}. {m.group(2)}"
            if m.group(4):
                labels[m.group(4)] = num
            out.append(tex[pos:m.start()] + f"\\{m.group(1)}{{{title}}}")
            pos = m.end()
        tex = "".join(out) + tex[pos:]
    aux = os.path.join(HERE, "supp.aux")   # references into the supplement (xr-hyper, prefix S-)
    if os.path.exists(aux):
        for k, v in re.findall(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", open(aux, encoding="utf-8").read()):
            labels.setdefault("S-" + k, v)
    tex = re.sub(r"\\usepackage\{xr-hyper\}|\\externaldocument(\[[^]]*\])?\{[^}]*\}", "", tex)
    tex = re.sub(r"\\ref\{([^}]*)\}", lambda m: labels.get(m.group(1), "?"), tex)
    tex = tex.replace("~", " ")
    tex = tex.replace("$<$", "\\textless{}").replace("$>$", "\\textgreater{}")
    tex = tex.replace("epoch $=1,\\dots,60$", "epoch $e=1,\\dots,60$")
    # citations -> [n]
    keys = re.findall(r"\\bibitem\{([^}]*)\}", tex)
    num = {k: i + 1 for i, k in enumerate(keys)}

    def cite(m):
        ks = [num[k.strip()] for k in m.group(1).split(",")]
        ks.sort()
        out, i = [], 0
        while i < len(ks):
            j = i
            while j + 1 < len(ks) and ks[j + 1] == ks[j] + 1:
                j += 1
            out.append(f"{ks[i]}–{ks[j]}" if j - i >= 2 else ", ".join(str(x) for x in ks[i:j + 1]))
            i = j + 1
        return "[" + ", ".join(out) + "]"
    tex = re.sub(r"\\cite\{([^}]*)\}", cite, tex)
    # bibliography -> numbered paragraphs
    def bib(m):
        items = re.split(r"\\bibitem\{[^}]*\}", m.group(1))[1:]
        return "\\section*{References}\n" + "\n\n".join(f"[{i + 1}] {it.strip()}" for i, it in enumerate(items))
    tex = re.sub(r"\\begin\{thebibliography\}\{\d+\}(.*?)\\end\{thebibliography\}", bib, tex, flags=re.S)
    tex = tex.replace("\\footnotesize\n\\section*{References}", "\\section*{References}")
    # IEEE-specific commands
    tex = re.sub(r"\\IEEEPARstart\{(.)\}\{([^}]*)\}", r"\1\2", tex)
    tex = re.sub(r"\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}", r"\\noindent\\textit{Index Terms}---\1", tex, flags=re.S)
    tex = re.sub(r"\\orcidlink\{[^}]*\}", "", tex)
    tex = re.sub(r"\\markboth\{[^}]*\}\{[^}]*\}", "", tex)
    tex = re.sub(r"\\hyphenation\{[^}]*\}", "", tex)
    # algorithm -> plain numbered steps
    def alg(m):
        body = m.group(1)
        cap = re.search(r"\\caption\{([^}]*)\}", body).group(1)
        lines = []
        for ln in body.splitlines():
            ln = ln.strip()
            if not ln or ln.startswith(("\\caption", "\\label", "\\footnotesize", "\\begin{algorithmic}", "\\end{algorithmic}")):
                continue
            ln = re.sub(r"^\\Require\s*", "Input: ", ln)
            ln = re.sub(r"^\\Ensure\s*", "Output: ", ln)
            ln = re.sub(r"^\\State\s*", "", ln)
            ln = re.sub(r"^\\For\{(.*)\}$", r"for \1 do", ln)
            ln = re.sub(r"^\\EndFor$", "end for", ln)
            lines.append(ln)
        k = labels.get(re.search(r"\\label\{([^}]*)\}", body).group(1), "")
        steps = "\n".join(f"\\item {x}" for x in lines if not x.startswith(("Input:", "Output:")))
        head = [x for x in lines if x.startswith("Input:")]
        tail = [x for x in lines if x.startswith("Output:")]
        return (f"\\noindent\\textbf{{Algorithm {k}: {cap}}}\n\n" + ("\\noindent " + head[0] + "\n\n" if head else "")
                + "\\begin{enumerate}\n" + steps + "\n\\end{enumerate}\n\n" + ("\\noindent " + tail[0] + "\n\n" if tail else ""))
    tex = re.sub(r"\\begin\{algorithm\}\[[^\]]*\](.*?)\\end\{algorithm\}", alg, tex, flags=re.S)
    # figures: png paths, explicit widths, and markers for full-width floats
    def fig(m):
        star, body = m.group(1), m.group(2)
        g = re.search(r"\\includegraphics\[[^\]]*\]\{figures/([^}.]*)\.pdf\}", body)
        name = g.group(1)
        src = os.path.join(FIG_SRC.get(name, "figures"), name + ".png")
        w = "6.9in" if star else "3.4in"
        if ONECOL:
            fr = re.search(r"width=([\d.]*)\\(?:linewidth|textwidth|columnwidth)", g.group(0))
            w = f"{TEXT_IN * float(fr.group(1) or 1) if fr else TEXT_IN:.2f}in"
            star = ""
        body = body.replace(g.group(0), f"\\includegraphics[width={w}]{{{src}}}")
        cap = re.search(r"\\caption\{(.*)\}\s*\\label", body, re.S)
        k = labels.get(re.search(r"\\label\{([^}]*)\}", body).group(1), "")
        body = re.sub(r"\\caption\{.*\}\s*(\\label\{[^}]*\})", "", body, flags=re.S)
        mark = "FULLWIDTHSTART\n\n" if star else ""
        endm = "\n\nFULLWIDTHEND" if star else ""
        return (f"{mark}\\begin{{center}}\n{body.replace(chr(92) + 'centering', '').strip()}\n\\end{{center}}\n\n"
                f"CAPFIG Fig. {k}. {cap.group(1).strip()}{endm}\n\n")
    tex = re.sub(r"\\begin\{figure(\*?)\}\[[^\]]*\](.*?)\\end\{figure\*?\}", fig, tex, flags=re.S)
    # tables: caption above, markers for full-width
    def tab(m):
        star, body = ("" if ONECOL else m.group(1)), m.group(2)
        cap = re.search(r"\\caption\{([^}]*)\}", body).group(1)
        k = labels.get(re.search(r"\\label\{([^}]*)\}", body).group(1), "")
        body = re.sub(r"\\caption\{[^}]*\}", "", body)
        body = re.sub(r"\\label\{[^}]*\}", "", body)
        for cmd in ("\\centering", "\\footnotesize", "\\scriptsize", "\\small"):
            body = body.replace(cmd, "")
        body = re.sub(r"\\setlength\{\\tabcolsep\}\{[^}]*\}", "", body)
        # footnote rows spanning all columns -> paragraph below the table
        notes = re.findall(r"\\multicolumn\{\d+\}\{p\{[^}]*\}\}\{\s*(?:\\(?:tiny|scriptsize))?\s*(.*?)\}\\\\", body, re.S)
        body = re.sub(r"\\multicolumn\{\d+\}\{p\{[^}]*\}\}\{\s*(?:\\(?:tiny|scriptsize))?\s*(.*?)\}\\\\", "", body, flags=re.S)
        body = re.sub(r"\\multicolumn\{\d+\}\{l\}\{\\scriptsize\s*(.*?)\}\\\\", "", body, flags=re.S)
        note = "\n\n" + "\n\n".join(f"CAPNOTE {x.strip()}" for x in notes) if notes else ""
        mark = "FULLWIDTHSTART\n\n" if star else ""
        endm = "\n\nFULLWIDTHEND" if star else ""
        return f"{mark}CAPTAB TABLE {k}. {cap}\n\n{body.strip()}{note}{endm}\n\n"
    tex = re.sub(r"\\begin\{table(\*?)\}\[[^\]]*\](.*?)\\end\{table\*?\}", tab, tex, flags=re.S)
    tex = tex.replace(">{\\centering\\arraybackslash}", "").replace(">{\\raggedright\\arraybackslash}", "").replace(">{\\arraybackslash}", "")
    tex = tex.replace("\\toprule", "\\hline").replace("\\midrule", "\\hline").replace("\\bottomrule", "\\hline")
    return tex


# ------------------------------------------------------------------ Word post-processing
def sect(cols, last=False):
    s = etree.Element(q("w:sectPr"))
    if not last:
        t = etree.SubElement(s, q("w:type"))
        t.set(q("w:val"), "continuous")
    pg = etree.SubElement(s, q("w:pgSz"))
    pg.set(q("w:w"), "12240"); pg.set(q("w:h"), "15840")
    mar = etree.SubElement(s, q("w:pgMar"))
    for k, v in dict(top="1080", right="893", bottom="1440", left="893", header="720", footer="720", gutter="0").items():
        mar.set(q(f"w:{k}"), v)
    c = etree.SubElement(s, q("w:cols"))
    c.set(q("w:num"), str(cols)); c.set(q("w:space"), "288")
    if ONECOL:
        c.set(q("w:num"), "1")
        for k in ("top", "right", "bottom", "left"):
            mar.set(q(f"w:{k}"), "1440")
    return s


RPR_AFTER = {"highlight", "u", "effect", "bdr", "shd", "fitText", "vertAlign", "rtl", "cs", "em", "lang",
             "eastAsianLayout", "specVanish", "oMath"}


def set_size(rpr, half_points):
    for e in rpr.findall(q("w:sz")) + rpr.findall(q("w:szCs")):
        rpr.remove(e)
    pos = len(rpr)
    for i, ch in enumerate(rpr):
        if etree.QName(ch).localname in RPR_AFTER:
            pos = i
            break
    for j, tag in enumerate(("w:sz", "w:szCs")):
        e = etree.Element(q(tag)); e.set(q("w:val"), str(half_points)); rpr.insert(pos + j, e)


def text_of_el(p):
    return "".join(t.text or "" for t in p.iter(q("w:t")))


def text_of(p):
    return "".join(t.text or "" for t in p.iter(q("w:t")))


def set_props(p, **kw):
    ppr = p.find(q("w:pPr"))
    if ppr is None:
        ppr = etree.Element(q("w:pPr")); p.insert(0, ppr)
    return ppr


def layout(docx_in, docx_out, title_paras):
    tmp = tempfile.mkdtemp()
    with zipfile.ZipFile(docx_in) as z:
        z.extractall(tmp)
    dp = os.path.join(tmp, "word", "document.xml")
    tree = etree.parse(dp)
    body = tree.getroot().find(q("w:body"))
    kids = [k for k in body if k.tag != q("w:sectPr")]
    final = body.find(q("w:sectPr"))
    if final is not None:
        body.remove(final)

    def end_section_at(el, cols):
        """Close a section whose last element is `el` (a paragraph); tables get a paragraph after them."""
        if el.tag != q("w:p"):
            p = etree.Element(q("w:p"))
            el.addnext(p)
            el = p
        ppr = el.find(q("w:pPr"))
        if ppr is None:
            ppr = etree.Element(q("w:pPr")); el.insert(0, ppr)
        ppr.append(sect(cols))

    wide_tables, inside = set(), False
    for k in kids:
        t = text_of(k).strip() if k.tag == q("w:p") else ""
        if t == "FULLWIDTHSTART":
            inside = True
        elif t == "FULLWIDTHEND":
            inside = False
        elif inside and k.tag == q("w:tbl"):
            wide_tables.add(k)
    # remove marker paragraphs, remembering their neighbours
    marks = []
    for k in kids:
        t = text_of(k) if k.tag == q("w:p") else ""
        if t.strip() in ("FULLWIDTHSTART", "FULLWIDTHEND"):
            marks.append((t.strip(), k))
    # title block = everything before the first heading of the body (or the first 'Index Terms' paragraph)
    first_body = None
    for k in kids:
        if k.tag == q("w:p") and text_of(k).startswith(title_paras):
            first_body = k
            break
    if first_body is not None and not ONECOL:
        end_section_at(first_body.getprevious(), 1)
    # caption markers -> caption styles
    styles = {"CAPFIG": ("ImageCaption", "both"), "CAPTAB": ("TableCaption", "center"), "CAPNOTE": ("ImageCaption", "both")}
    for p in body.iter(q("w:p")):
        ts = list(p.iter(q("w:t")))
        if not ts or not (ts[0].text or "").startswith(tuple(styles)):
            continue
        key = ts[0].text.split(" ")[0]
        ts[0].text = ts[0].text[len(key):].lstrip()
        if not ts[0].text and len(ts) > 1:
            ts[1].text = (ts[1].text or "").lstrip()
        ppr = p.find(q("w:pPr"))
        if ppr is None:
            ppr = etree.Element(q("w:pPr")); p.insert(0, ppr)
        for e in ppr.findall(q("w:pStyle")) + ppr.findall(q("w:jc")):
            ppr.remove(e)
        ps = etree.Element(q("w:pStyle")); ps.set(q("w:val"), styles[key][0]); ppr.insert(0, ps)
        jc = etree.Element(q("w:jc")); jc.set(q("w:val"), styles[key][1])
        later = [ch for ch in ppr if etree.QName(ch).localname in
                 ("textDirection", "textAlignment", "textboxTightWrap", "outlineLvl", "divId", "cnfStyle", "rPr", "sectPr", "pPrChange")]
        (later[0].addprevious(jc) if later else ppr.append(jc))
        if key == "CAPTAB":
            kn = etree.Element(q("w:keepNext")); ppr.insert(1, kn)
        for r in p.iter(q("w:rPr")):
            for e in r.findall(q("w:sz")) + r.findall(q("w:szCs")):
                r.remove(e)
    # tables: full width of their section, small type, centered
    for tbl in body.iter(q("w:tbl")):
        tpr = tbl.find(q("w:tblPr"))
        tw = tpr.find(q("w:tblW"))
        if tw is None:
            tw = etree.Element(q("w:tblW"))
            st_ = tpr.find(q("w:tblStyle"))
            (st_.addnext(tw) if st_ is not None else tpr.insert(0, tw))
        # column widths proportional to content length, filling the column (or page) width
        full = tbl.getprevious() is not None and False
        rows = tbl.findall(q("w:tr"))
        ncol = max(len(r.findall(q("w:tc"))) for r in rows)
        lens = [4] * ncol
        for r in rows:
            for j, tc in enumerate(r.findall(q("w:tc"))):
                words = text_of_el(tc).split()
                if ONECOL:
                    lens[j] = max(lens[j], min(30, len(text_of_el(tc)) + 2))
                else:
                    lens[j] = max(lens[j], min(26, max([len(w) for w in words] + [0]) + 2), min(26, len(text_of_el(tc)) // 2))
        avail = int(TEXT_IN * 1440) if ONECOL else (10454 if tbl in wide_tables else 5083)
        widths = [int(avail * l / sum(lens)) for l in lens]
        grid = tbl.find(q("w:tblGrid"))
        for e in list(grid):
            grid.remove(e)
        for wdt in widths:
            g = etree.SubElement(grid, q("w:gridCol")); g.set(q("w:w"), str(wdt))
        for r in rows:
            for j, tc in enumerate(r.findall(q("w:tc"))):
                tcpr = tc.find(q("w:tcPr"))
                if tcpr is None:
                    tcpr = etree.Element(q("w:tcPr")); tc.insert(0, tcpr)
                span = tcpr.find(q("w:gridSpan"))
                k = int(span.get(q("w:val"))) if span is not None else 1
                for e in tcpr.findall(q("w:tcW")):
                    tcpr.remove(e)
                tw_ = etree.Element(q("w:tcW")); tw_.set(q("w:w"), str(sum(widths[j:j + k]))); tw_.set(q("w:type"), "dxa")
                tcpr.insert(0, tw_)
        tw.set(q("w:w"), str(sum(widths))); tw.set(q("w:type"), "dxa")
        for e in tpr.findall(q("w:jc")):
            e.set(q("w:val"), "center")
        for e in tpr.findall(q("w:tblCellMar")):
            tpr.remove(e)
        cm = etree.Element(q("w:tblCellMar"))
        for side in ("left", "right"):
            e = etree.SubElement(cm, q(f"w:{side}")); e.set(q("w:w"), "50"); e.set(q("w:type"), "dxa")
        look = tpr.find(q("w:tblLook"))
        (look.addprevious(cm) if look is not None else tpr.append(cm))
        for e in tpr.findall(q("w:tblLayout")):
            tpr.remove(e)
        for r in tbl.iter(q("w:r")):
            rpr = r.find(q("w:rPr"))
            if rpr is None:
                rpr = etree.Element(q("w:rPr")); r.insert(0, rpr)
            set_size(rpr, (17 if ncol <= 4 else 15) if ONECOL else 15)
    for kind, el in marks:
        prev = el.getprevious()
        if kind == "FULLWIDTHSTART":
            if prev is not None:
                end_section_at(prev, 2)
        else:
            end_section_at(prev, 1)
        body.remove(el)
    body.append(sect(2, last=True))
    # images: keep pandoc's sizes; center them
    tree.write(dp, xml_declaration=True, encoding="UTF-8", standalone=True)
    # styles: Times New Roman 10 pt, single spacing, compact paragraphs
    sp = os.path.join(tmp, "word", "styles.xml")
    st = open(sp, encoding="utf-8").read()
    st = re.sub(r'w:ascii="[^"]*"', 'w:ascii="Times New Roman"', st)
    st = re.sub(r'w:hAnsi="[^"]*"', 'w:hAnsi="Times New Roman"', st)
    st = re.sub(r'w:eastAsia="[^"]*"', 'w:eastAsia="Times New Roman"', st)
    st = re.sub(r'w:cs="[^"]*"', 'w:cs="Times New Roman"', st)
    st = re.sub(r'w:asciiTheme="[^"]*"', "", st)
    st = re.sub(r'w:hAnsiTheme="[^"]*"', "", st)
    st = re.sub(r'w:eastAsiaTheme="[^"]*"', "", st)
    st = re.sub(r'w:cstheme="[^"]*"', "", st)
    open(sp, "w", encoding="utf-8").write(st)
    if os.path.exists(docx_out):
        os.remove(docx_out)
    with zipfile.ZipFile(docx_out, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(tmp):
            for f in files:
                full = os.path.join(root, f)
                z.write(full, os.path.relpath(full, tmp))
    shutil.rmtree(tmp)


def reference_docx(path):
    """Pandoc reference document with IEEE-like styles."""
    subprocess.run(["pandoc", "-o", path, "--print-default-data-file", "reference.docx"], check=True)
    tmp = tempfile.mkdtemp()
    with zipfile.ZipFile(path) as z:
        z.extractall(tmp)
    sp = os.path.join(tmp, "word", "styles.xml")
    tree = etree.parse(sp)
    root = tree.getroot()

    def style(sid):
        for s in root.findall(q("w:style")):
            if s.get(q("w:styleId")) == sid:
                return s
        return None

    def set_run(s, size=None, bold=None, italic=None, caps=None, color="000000"):
        rpr = s.find(q("w:rPr"))
        if rpr is None:
            rpr = etree.SubElement(s, q("w:rPr"))
        for tag in ("w:sz", "w:szCs", "w:b", "w:i", "w:smallCaps", "w:color", "w:rFonts"):
            for e in rpr.findall(q(tag)):
                rpr.remove(e)
        f = etree.SubElement(rpr, q("w:rFonts"))
        for a in ("ascii", "hAnsi", "eastAsia", "cs"):
            f.set(q(f"w:{a}"), "Times New Roman")
        if bold is not None:
            b = etree.SubElement(rpr, q("w:b")); b.set(q("w:val"), "1" if bold else "0")
        if italic is not None:
            i = etree.SubElement(rpr, q("w:i")); i.set(q("w:val"), "1" if italic else "0")
        if caps:
            etree.SubElement(rpr, q("w:smallCaps"))
        c = etree.SubElement(rpr, q("w:color")); c.set(q("w:val"), color)
        if size:
            for tag in ("w:sz", "w:szCs"):
                e = etree.SubElement(rpr, q(tag)); e.set(q("w:val"), str(size))

    def set_par(s, before=0, after=0, jc=None, line=None, ind_first=None):
        ppr = s.find(q("w:pPr"))
        if ppr is None:
            ppr = etree.Element(q("w:pPr")); s.insert(1, ppr)
        for tag in ("w:spacing", "w:jc", "w:ind"):
            for e in ppr.findall(q(tag)):
                ppr.remove(e)
        sp_ = etree.SubElement(ppr, q("w:spacing"))
        sp_.set(q("w:before"), str(before)); sp_.set(q("w:after"), str(after))
        if line:
            sp_.set(q("w:line"), str(line)); sp_.set(q("w:lineRule"), "auto")
        if jc:
            j = etree.SubElement(ppr, q("w:jc")); j.set(q("w:val"), jc)
        if ind_first is not None:
            ind = etree.SubElement(ppr, q("w:ind")); ind.set(q("w:firstLine"), str(ind_first))

    B, L = (22, 264) if ONECOL else (20, 240)
    for sid, run, par in [("Normal", dict(size=B), dict(after=0, jc="both", line=L)),
                          ("BodyText", dict(size=B), dict(after=0, jc="both", line=L, ind_first=202)),
                          ("FirstParagraph", dict(size=B), dict(after=0, jc="both", line=L, ind_first=0)),
                          ("Compact", dict(size=B), dict(after=0, jc="both", line=L, ind_first=0)),
                          ("Title", dict(size=48), dict(before=0, after=120, jc="center")),
                          ("Author", dict(size=22), dict(after=120, jc="center")),
                          ("Abstract", dict(size=18, bold=True), dict(after=120, jc="both")),
                          ("Heading1", dict(size=B, caps=True), dict(before=200, after=80, jc="center")),
                          ("Heading2", dict(size=B, italic=True), dict(before=120, after=60)),
                          ("Heading3", dict(size=B, italic=True), dict(before=60, after=0)),
                          ("ImageCaption", dict(size=B - 4), dict(after=120, jc="both", ind_first=0)),
                          ("TableCaption", dict(size=B - 4), dict(before=120, after=60, jc="center", ind_first=0)),
                          ("Bibliography", dict(size=B - 4), dict(after=0, ind_first=0))]:
        s = style(sid)
        if s is not None:
            set_run(s, **run)
            set_par(s, **par)
    tree.write(sp, xml_declaration=True, encoding="UTF-8", standalone=True)
    os.remove(path)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for root_, _, files in os.walk(tmp):
            for f in files:
                full = os.path.join(root_, f)
                z.write(full, os.path.relpath(full, tmp))
    shutil.rmtree(tmp)


def convert(tex_name, out_name, prefix="", title_start=("I. Introduction", "Introduction")):
    src = open(os.path.join(HERE, tex_name), encoding="utf-8").read()
    src = re.sub(r"\\input\{([^}]*)\}", lambda k: open(os.path.join(HERE, k.group(1)), encoding="utf-8").read(), src)
    tex = preprocess(src, prefix)
    pre = os.path.join(HERE, "_word_" + tex_name)
    open(pre, "w", encoding="utf-8").write(tex)
    ref = os.path.join(HERE, "_reference.docx")
    reference_docx(ref)
    raw = os.path.join(HERE, "_raw_" + out_name)
    subprocess.run(["pandoc", pre, "-f", "latex", "-t", "docx", "--reference-doc", ref, "--resource-path", HERE,
                    "--number-sections" if False else "--standalone", "-o", raw], check=True, cwd=HERE)
    out = os.path.join(HERE, "submission", out_name)
    layout(raw, out, title_start)
    for f in (pre, ref, raw):
        os.remove(f)
    print("wrote", out)


if __name__ == "__main__":
    convert("main.tex", "JBHI_manuscript.docx", title_start=("I. Introduction",))
    convert("supp.tex", "JBHI_supplementary_material.docx", prefix="S", title_start=("S1.",))
