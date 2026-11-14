"""Model comparison: DeLong test for paired AUCs, Holm correction, NRI/IDI."""
from __future__ import annotations

import numpy as np
from scipy import stats as sps


def _midrank(x):
    order = np.argsort(x)
    xs = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def delong(y, p1, p2):
    """Paired DeLong test (fast algorithm of Sun & Xu, 2014).

    Returns dict(auc1, auc2, diff, se, z, p, ci_lo, ci_hi) for diff = auc2 - auc1.
    """
    y = np.asarray(y).astype(int)
    preds = np.vstack([np.asarray(p1, float), np.asarray(p2, float)])
    order = np.argsort(-y, kind="stable")  # positives first
    preds = preds[:, order]
    m = int(y.sum())
    n = len(y) - m
    if m == 0 or n == 0:
        raise ValueError("need both classes")
    k = preds.shape[0]
    tx, ty, tz = np.empty((k, m)), np.empty((k, n)), np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(preds[r, :m])
        ty[r] = _midrank(preds[r, m:])
        tz[r] = _midrank(preds[r])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    diff = aucs[1] - aucs[0]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    se = float(np.sqrt(max(var, 1e-12)))
    z = diff / se
    return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "diff": float(diff), "se": se, "z": float(z),
            "p": float(2 * sps.norm.sf(abs(z))), "ci_lo": float(diff - 1.96 * se), "ci_hi": float(diff + 1.96 * se)}


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def nri_idi(y, p_old, p_new, thr):
    y, po, pn = np.asarray(y), np.asarray(p_old), np.asarray(p_new)
    up, down = (pn >= thr) & (po < thr), (pn < thr) & (po >= thr)
    ev, ne = y == 1, y == 0
    nri = (up[ev].mean() - down[ev].mean()) + (down[ne].mean() - up[ne].mean())
    idi = (pn[ev].mean() - po[ev].mean()) - (pn[ne].mean() - po[ne].mean())
    return {"nri": float(nri), "idi": float(idi)}


def verdict(d, margin=0.03):
    """Pre-specified reading of the primary comparison (multimodal minus clinical AUC)."""
    if d["ci_lo"] > 0:
        return "imaging adds value" + ("" if d["ci_lo"] >= margin else " (lower bound below the margin)")
    if d["ci_lo"] > -margin and d["ci_hi"] < margin:
        return "no clinically meaningful added value (interval inside the margin)"
    return "inconclusive"
