"""Audit of how the clinical anchor is nested inside the cross-validation (second review, point 4).

    python paper/jbhi/audit_nesting.py  ->  results/revision/rev_audit_nesting.csv

As run: the anchor's out-of-fold logits z were cross-fitted once over all of I-SPY2, with the same
five folds as the outer cross-validation (same seed). For a network trained in outer fold k, the
anchors of its training patients therefore came from clinical models that had seen the labels of
fold k (the test fold) and of the inner early-stopping patients. The anchors of the test patients
themselves came from a model fitted on the other four folds only, and external anchors from a model
fitted on all of I-SPY2, so external results are not affected.
This script compares the anchors as used with fully nested anchors (cross-fitted within the
training patients of each outer fold, excluding the test fold and the early-stopping patients).
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
from mmonc.models import ClinicalAnchor  # noqa: E402

T = pd.read_csv(os.path.join(ROOT, "results", "round2", "raw", "table.csv"))
T = T[(T.collection == "ISPY2") & T.pcr.notna()].reset_index(drop=True)
X, y = T[["age", "hr", "her2"]].to_numpy(float), T.pcr.to_numpy().astype(int)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
z_used = ClinicalAnchor(n_splits=5).fit(X, y)
anchor_folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y))
outer = list(cv.split(np.arange(len(y)), y))
same = all((a[1] == b[1]).all() for a, b in zip(anchor_folds, outer))
rows = []
for k, (tr, te) in enumerate(outer):
    perm = np.random.default_rng(k).permutation(tr)  # round-2 inner split, seed 0 * 100 + k
    n_val = max(2, int(0.15 * len(tr)))
    tr2, va = perm[n_val:], perm[:n_val]
    z_nested = ClinicalAnchor(n_splits=5).fit(X[tr2], y[tr2])
    d = z_nested - z_used[tr2]
    rows.append({"fold": k, "n_train": len(tr2), "mean_abs_diff": np.abs(d).mean(), "p99_abs_diff": np.percentile(np.abs(d), 99),
                 "max_abs_diff": np.abs(d).max(), "corr": np.corrcoef(z_nested, z_used[tr2])[0, 1],
                 "sd_anchor": z_used[tr2].std()})
R = pd.DataFrame(rows)
R["folds_identical"] = same
R.to_csv(os.path.join(ROOT, "results", "revision", "rev_audit_nesting.csv"), index=False)
print("anchor folds identical to outer folds:", same)
print(R.round(4).to_string(index=False))
