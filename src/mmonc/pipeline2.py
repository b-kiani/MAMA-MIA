"""Round-2 experiments. They were specified after the round-1 external results were known
(paper/analysis_plan_round2.md), so they are secondary analyses; the primary end point is unchanged.

    python -m mmonc.pipeline2 --data P --out O --exp devext   --models image_only_k,carf_k --tag g0
    python -m mmonc.pipeline2 --data P --out O --exp controls --tag g1
    python -m mmonc.pipeline2 --data P --out O --exp loco     --models image_only_k,carf_k --tag g0

Each experiment only writes per-case predictions; statistics are computed by mmonc.report2.
A deadline (env MMONC_DEADLINE, unix seconds) stops new fits from starting so that a 12-hour
session always ends with its predictions written.
"""
from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import metrics
from .models import SLICE_MODELS, ClinicalAnchor, build, make_logreg
from .prep2 import KIN_NAMES
from .train import Cases, fit, predict, seed_all

LABELS = {"image_only_k": "Image only, kinetic 3D ResNet", "concat_k": "Concatenation fusion, kinetic",
          "carf_k": "CARF, kinetic", "s25d": "Image only, pretrained 2.5D", "carf_25d": "CARF, pretrained 2.5D"}
DEFAULTS = dict(dev="ISPY2", clinical=["age", "hr", "her2"], folds=5, seeds=[0], epochs=60, patience=12, batch_size=16,
                lr=3e-4, lr_pretrained=1e-4, widths=[32, 64, 128, 256], dim=128, control_folds=3,
                controls=["hr", "her2", "field3t", "perm"])


class Data2:
    """Folder written by mmonc.prep2, cached in RAM."""

    def __init__(self, root: str):
        self.table = pd.read_csv(os.path.join(root, "table.csv"))
        k = pd.read_csv(os.path.join(root, "kinetics.csv"), index_col=0)
        self.kin = k.loc[self.table.case_id, KIN_NAMES].to_numpy(np.float32)
        self._vol, self._mask, self._sl = [], [], []
        for c in self.table.case_id:
            z = np.load(os.path.join(root, f"{c}.npz"))
            self._vol.append(z["vol"]); self._mask.append(z["mask"]); self._sl.append(z["sl"])

    def vol6(self, i):  # pre, early, late, early - pre, late - early, mask
        pre, early, late = self._vol[i].astype(np.float32)
        return np.stack([pre - 1, early - 1, late - 1, early - pre, late - early, self._mask[i].astype(np.float32)])

    def slices(self, i):  # (12, 3, P, P)
        return self._sl[i].astype(np.float32) - 1.0


class InMemory2:
    def __init__(self, table, vols, masks, sls, kin):
        self.table, self._vol, self._mask, self._sl, self.kin = table.reset_index(drop=True), vols, masks, sls, kin

    vol6, slices = Data2.vol6, Data2.slices


class Cases2(Cases):
    """Flips over the spatial axes only; intensity jitter leaves the mask channel untouched."""

    def __init__(self, *a, kind="vol", **kw):
        super().__init__(*a, **kw)
        self.kind = kind

    def __getitem__(self, i):
        img = torch.as_tensor(np.ascontiguousarray(self.loader(i)), dtype=torch.float32)
        if self.augment:
            for ax in ((1, 2, 3) if self.kind == "vol" else (2, 3)):
                if random.random() < 0.5:
                    img = img.flip(ax)
            s, b = 1 + 0.1 * torch.randn(1), 0.1 * torch.randn(1)
            if self.kind == "vol":
                img[:5] = img[:5] * s
                img[:3] = img[:3] + b
            else:
                img = img * s + b
        return {"image": img, "tab": self.tab[i], "tab_missing": self.miss[i], "dt": self.dt[i],
                "z_clin": self.z[i], "y": self.y[i]}


def _standardise(X, ref):
    mu, sd = np.nanmean(ref, 0), np.nanstd(ref, 0) + 1e-6
    Z = (X - mu) / sd
    return np.nan_to_num(Z, nan=0.0).astype(np.float32), np.isnan(Z).astype(np.float32)


def _dev():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _late():
    d = float(os.environ.get("MMONC_DEADLINE", "0") or 0)
    return d > 0 and time.time() > d


class Runner:
    def __init__(self, data, out, cfg, log):
        self.data, self.out, self.cfg, self.log = data, out, {**DEFAULTS, **(cfg or {})}, log
        os.makedirs(os.path.join(out, "ck"), exist_ok=True)
        tb = data.table
        self.tb = tb
        self.Xc = tb[self.cfg["clinical"]].to_numpy(float)
        self.Xk = np.asarray(data.kin, float)
        cols = [c for c in ("dt0", "dt1", "dt_last") if c in tb.columns]
        dt = tb[cols].to_numpy(float) if len(cols) == 3 else np.zeros((len(tb), 3))
        self.dt = np.nan_to_num(dt, nan=0.0)

    def ds(self, name, idx, tab, miss, z, y, augment=False):
        kind = "sl" if name in SLICE_MODELS else "vol"
        get = self.data.slices if kind == "sl" else self.data.vol6
        return Cases2(None, tab[idx], miss[idx], self.dt[idx], z[idx], y[idx], augment,
                      loader=lambda j, idx=idx: get(int(idx[j])), kind=kind)

    def train_one(self, name, tr, va, tab, miss, z, y, seed, tag):
        c = self.cfg
        seed_all(seed)
        model = build(name, tab.shape[1], tuple(c["widths"]), c["dim"])
        t0 = time.time()
        ck = os.path.join(self.out, "ck", f"{tag}.pt")
        hist = fit(model, self.ds(name, tr, tab, miss, z, y, True), self.ds(name, va, tab, miss, z, y),
                   epochs=c["epochs"], lr=c["lr_pretrained"] if name in SLICE_MODELS else c["lr"],
                   batch_size=c["batch_size"], patience=c["patience"], seed=seed, ckpt_path=ck, log=lambda *m: None)
        if os.path.exists(ck):
            os.remove(ck)  # only needed to resume inside a session; keeps the output small
        best = max((h["val_auc"] for h in hist if h["val_auc"] == h["val_auc"]), default=float("nan"))
        self.log(f"[{tag}] {len(hist)} epochs, best inner-validation AUC {best:.3f}, {(time.time() - t0) / 60:.1f} min")
        return model

    def score(self, model, name, idx, tab, miss, z, y):
        r = predict(model, self.ds(name, idx, tab, miss, z, y), batch_size=self.cfg["batch_size"], device=_dev())
        return metrics.sigmoid(r["logit"]), r.get("gate")

    @staticmethod
    def inner_split(tr, seed):
        perm = np.random.default_rng(seed).permutation(tr)
        n_val = max(2, int(0.15 * len(tr)))
        return perm[n_val:], perm[:n_val]

    # ------------------------------------------------------------------ ISPY2 -> external
    def devext(self, models, tag):
        c, tb = self.cfg, self.tb
        keep = tb.pcr.notna().to_numpy()
        y = tb.pcr.fillna(0).to_numpy().astype(int)
        dev = np.flatnonzero((tb.collection == c["dev"]).to_numpy() & keep)
        ext = np.flatnonzero((tb.collection != c["dev"]).to_numpy() & keep)
        cv = StratifiedKFold(c["folds"], shuffle=True, random_state=0)
        anchor = ClinicalAnchor(n_splits=c["folds"])
        z = np.zeros(len(tb), np.float32)
        z[dev] = anchor.fit(self.Xc[dev], y[dev])
        z[ext] = anchor.logit(self.Xc[ext])
        P = pd.DataFrame({"case_id": tb.case_id, "collection": tb.collection, "y": y})
        P["set"] = np.where(np.isin(np.arange(len(tb)), dev), "dev", np.where(np.isin(np.arange(len(tb)), ext), "ext", ""))
        P = P[P.set != ""].copy()
        pos = {int(i): k for k, i in enumerate(P.index)}
        di, ei = [pos[int(i)] for i in dev], [pos[int(i)] for i in ext]

        def put(label, o, e):
            v = np.full(len(P), np.nan)
            v[di], v[ei] = o, e
            P[label] = v
            P.to_csv(os.path.join(self.out, f"preds_devext_{tag}.csv"), index=False)

        if "tabular" in models:
            put("Clinical", metrics.sigmoid(z[dev]), metrics.sigmoid(z[ext]))
            ftv = self.Xk[:, [KIN_NAMES.index("log_ftv_mm3")]]
            for label, X, pen, C in [("Clinical + FTV", np.hstack([self.Xc, ftv]), "l2", 1.0),
                                     ("Kinetic radiomics", self.Xk, "l1", 0.3),
                                     ("Clinical + kinetic radiomics", np.hstack([self.Xc, self.Xk]), "l1", 0.3)]:
                m = make_logreg(C=C, penalty=pen)
                o = cross_val_predict(m, X[dev], y[dev], cv=cv, method="predict_proba")[:, 1]
                put(label, o, m.fit(X[dev], y[dev]).predict_proba(X[ext])[:, 1])
        Xt = np.hstack([self.Xc, self.Xk])
        tab, miss = _standardise(Xt, Xt[dev])
        for name in [m for m in models if m != "tabular"]:
            o, e, g_o, g_e, n, ok = np.zeros(len(dev)), np.zeros(len(ext)), np.zeros(len(dev)), np.zeros(len(ext)), 0, True
            for seed in c["seeds"]:
                for k, (tr, te) in enumerate(cv.split(dev, y[dev])):
                    if _late():
                        ok = False
                        break
                    tr2, va = self.inner_split(tr, seed * 100 + k)
                    model = self.train_one(name, dev[tr2], dev[va], tab, miss, z, y, seed * 100 + k, f"{name}_s{seed}_f{k}")
                    p, g = self.score(model, name, dev[te], tab, miss, z, y)
                    o[te] += p / len(c["seeds"])
                    pe, ge = self.score(model, name, ext, tab, miss, z, y)
                    e += pe
                    if g is not None:
                        g_o[te] += g / len(c["seeds"]); g_e += ge
                    n += 1
                    del model
            if not ok:
                self.log(f"[{name}] SKIPPED: time budget reached before all folds were trained")
                continue
            put(LABELS[name], o, e / n)
            if name.startswith("carf"):
                put("gate:" + LABELS[name], g_o, g_e / n)
            self.log(f"[{name}] done")

    # ------------------------------------------------------------------ control tasks
    def controls(self, tag):
        """Can the same image-only network learn other targets from the same crops?

        hr, her2: known biology. field3t: an acquisition property (shortcut probe).
        perm: pCR labels shuffled within ISPY2 (negative control; expected AUC 0.5).
        """
        c, tb = self.cfg, self.tb
        tgt = {"hr": tb.hr.to_numpy(float), "her2": tb.her2.to_numpy(float),
               "field3t": np.where(tb.field_strength.isna(), np.nan, (tb.field_strength >= 2.5).astype(float))}
        perm = tb.pcr.to_numpy(float).copy()
        d = np.flatnonzero((tb.collection == c["dev"]).to_numpy() & ~np.isnan(perm))
        perm[d] = np.random.default_rng(7).permutation(perm[d])
        tgt["perm"] = perm
        rows = []
        tab = np.zeros((len(tb), 1), np.float32); miss = np.zeros_like(tab); z = np.zeros(len(tb), np.float32)
        for t in c["controls"]:
            yv = tgt[t]
            ok = ~np.isnan(yv)
            y = np.nan_to_num(yv).astype(int)
            dev = np.flatnonzero((tb.collection == c["dev"]).to_numpy() & ok)
            ext = np.flatnonzero((tb.collection != c["dev"]).to_numpy() & ok)
            if len(np.unique(y[dev])) < 2:
                continue
            cv = StratifiedKFold(c["control_folds"], shuffle=True, random_state=0)
            o, e, n, done = np.zeros(len(dev)), np.zeros(len(ext)), 0, True
            for k, (tr, te) in enumerate(cv.split(dev, y[dev])):
                if _late():
                    done = False
                    break
                tr2, va = self.inner_split(tr, k)
                model = self.train_one("image_only_k", dev[tr2], dev[va], tab, miss, z, y, k, f"ctl_{t}_f{k}")
                o[te] = self.score(model, "image_only_k", dev[te], tab, miss, z, y)[0]
                e += self.score(model, "image_only_k", ext, tab, miss, z, y)[0]
                n += 1
                del model
            if not done:
                self.log(f"[control {t}] SKIPPED: time budget")
                continue
            rows.append(pd.DataFrame({"case_id": tb.case_id.iloc[dev], "collection": tb.collection.iloc[dev], "set": "dev",
                                      "target": t, "y": y[dev], "p": o}))
            if t != "perm" and len(ext):
                rows.append(pd.DataFrame({"case_id": tb.case_id.iloc[ext], "collection": tb.collection.iloc[ext], "set": "ext",
                                          "target": t, "y": y[ext], "p": e / n}))
            pd.concat(rows).to_csv(os.path.join(self.out, f"preds_controls_{tag}.csv"), index=False)
            self.log(f"[control {t}] done")

    # ------------------------------------------------------------------ leave one collection out
    def loco(self, models, tag):
        c, tb = self.cfg, self.tb
        keep = tb.pcr.notna().to_numpy()
        y = tb.pcr.fillna(0).to_numpy().astype(int)
        out = []
        for H in sorted(tb.collection.unique()):
            te = np.flatnonzero((tb.collection == H).to_numpy() & keep)
            tr = np.flatnonzero((tb.collection != H).to_numpy() & keep)
            anchor = ClinicalAnchor(n_splits=c["folds"])
            z = np.zeros(len(tb), np.float32)
            z[tr] = anchor.fit(self.Xc[tr], y[tr])
            z[te] = anchor.logit(self.Xc[te])
            P = pd.DataFrame({"case_id": tb.case_id.iloc[te], "collection": H, "y": y[te], "Clinical": metrics.sigmoid(z[te])})
            Xt = np.hstack([self.Xc, self.Xk])
            P["Clinical + kinetic radiomics"] = make_logreg(C=0.3, penalty="l1").fit(Xt[tr], y[tr]).predict_proba(Xt[te])[:, 1]
            tab, miss = _standardise(Xt, Xt[tr])
            for name in models:
                e, g_e, n = np.zeros(len(te)), np.zeros(len(te)), 0
                for seed in c["seeds"]:
                    if _late():
                        break
                    tr2, va = self.inner_split(tr, seed)
                    model = self.train_one(name, tr2, va, tab, miss, z, y, seed, f"loco_{H}_{name}_s{seed}")
                    p, g = self.score(model, name, te, tab, miss, z, y)
                    e += p
                    if g is not None:
                        g_e += g
                    n += 1
                    del model
                if n == 0:
                    self.log(f"[loco {H} {name}] SKIPPED: time budget")
                    continue
                P[LABELS[name]] = e / n
                if name.startswith("carf"):
                    P["gate:" + LABELS[name]] = g_e / n
            out.append(P)
            pd.concat(out).to_csv(os.path.join(self.out, f"preds_loco_{tag}.csv"), index=False)
            self.log(f"[loco {H}] done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exp", required=True, choices=["devext", "controls", "loco"])
    ap.add_argument("--models", default="")
    ap.add_argument("--tag", default="a")
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = {}
    if a.config:
        import yaml

        cfg = yaml.safe_load(open(a.config))
    if a.synthetic:
        from .synthetic import make_cohort2

        data = make_cohort2()
        cfg = {"epochs": 2, "folds": 2, "control_folds": 2, "widths": [8, 16], "dim": 16, "patience": 2, "batch_size": 8, **cfg}
    else:
        data = Data2(a.data)
    log = lambda *m: print(time.strftime("%H:%M:%S"), f"<{a.tag}>", *m, flush=True)  # noqa: E731
    log(f"{len(data.table)} cases; exp {a.exp}; models {a.models}; device {_dev()}")
    r = Runner(data, a.out, cfg, log)
    models = [m for m in a.models.split(",") if m]
    t0 = time.time()
    if a.exp == "devext":
        r.devext(models, a.tag)
    elif a.exp == "controls":
        r.controls(a.tag)
    else:
        r.loco(models, a.tag)
    log(f"total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
