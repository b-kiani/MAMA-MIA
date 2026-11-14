"""Discrimination, classification, calibration, clinical utility and survival metrics."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def sigmoid(z):
    return 1 / (1 + np.exp(-np.asarray(z, dtype=float)))


def bootstrap_ci(fn, y, *preds, n_boot=2000, seed=0, alpha=0.05):
    """Stratified percentile bootstrap. Returns (estimate, lo, hi)."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals.append(fn(y[idx], *[np.asarray(p)[idx] for p in preds]))
    lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(fn(y, *preds)), float(lo), float(hi)


def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def classification(y, p, thr):
    y, yhat = np.asarray(y), (np.asarray(p) >= thr).astype(int)
    tp, tn = int(((yhat == 1) & (y == 1)).sum()), int(((yhat == 0) & (y == 0)).sum())
    fp, fn = int(((yhat == 1) & (y == 0)).sum()), int(((yhat == 0) & (y == 1)).sum())
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    prec = tp / max(1, tp + fp)
    f1 = 2 * prec * sens / max(1e-12, prec + sens)
    return {"sensitivity": sens, "specificity": spec, "precision": prec, "recall": sens, "f1": f1,
            "accuracy": (tp + tn) / len(y), "balanced_accuracy": (sens + spec) / 2,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def calibration(y, p, n_bins=10):
    """Intercept (calibration-in-the-large), slope, ECE, Brier."""
    y, p = np.asarray(y), np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    slope = LogisticRegression(C=1e6, max_iter=1000).fit(lp, y).coef_[0, 0]
    # intercept with slope fixed at 1: Newton steps on a single parameter
    a = 0.0
    for _ in range(50):
        q = sigmoid(a + lp[:, 0])
        a += (y - q).sum() / max(1e-9, (q * (1 - q)).sum())
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    ece = sum((b == k).mean() * abs(y[b == k].mean() - p[b == k].mean()) for k in range(n_bins) if (b == k).any())
    return {"intercept": float(a), "slope": float(slope), "ece": float(ece), "brier": float(brier_score_loss(y, p))}


def reliability_points(y, p, n_bins=10):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    ks = [k for k in range(n_bins) if (b == k).any()]
    return np.array([p[b == k].mean() for k in ks]), np.array([y[b == k].mean() for k in ks])


def net_benefit(y, p, thresholds):
    y, p, n = np.asarray(y), np.asarray(p), len(y)
    out = []
    for t in thresholds:
        pred = p >= t
        out.append(((pred & (y == 1)).sum() - (pred & (y == 0)).sum() * t / (1 - t)) / n)
    return np.array(out)


def net_benefit_all(y, thresholds):
    prev = np.mean(y)
    return np.array([prev - (1 - prev) * t / (1 - t) for t in thresholds])


def summary(y, p, thr, n_boot=2000, seed=0):
    """One row of a performance table."""
    auc = bootstrap_ci(roc_auc_score, y, p, n_boot=n_boot, seed=seed)
    ap = bootstrap_ci(average_precision_score, y, p, n_boot=n_boot, seed=seed)
    row = {"n": len(y), "events": int(np.sum(y)), "auc": auc[0], "auc_lo": auc[1], "auc_hi": auc[2],
           "ap": ap[0], "ap_lo": ap[1], "ap_hi": ap[2]}
    row.update(classification(y, p, thr))
    row.update(calibration(y, p))
    return row


def c_index(time, event, risk):
    from lifelines.utils import concordance_index

    return float(concordance_index(time, -np.asarray(risk), event))
