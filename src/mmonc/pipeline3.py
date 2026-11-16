"""Revision experiments (reviewer response). All are secondary or exploratory analyses.

    python -m mmonc.pipeline3 --data P2 [--whole P3] --out O --exp devext  --models carfm_k --tag a
    python -m mmonc.pipeline3 --data P2 --out O --exp posctl  --models image_only_k,s25d --tag b
    python -m mmonc.pipeline3 --data P2 --out O --exp lc      --models s25d --tag c
    python -m mmonc.pipeline3 --data P2 --out O --exp overfit --models image_only_k,s25d --tag d
    python -m mmonc.pipeline3 --data P2 --out O --exp dino --tag e

devext   : I-SPY2 cross-validation and frozen external scoring (corrected CARF, whole-breast
           baseline); for CARFMix also an input-dependency test per fold
posctl   : learnable positive controls on the same crops (mean enhancement, washout fraction, vendor)
lc       : learning curves (25% and 50% of each training fold) with training-set AUC
overfit  : can the network memorise 120 patients when regularisation is removed?
dino     : frozen DINOv2 slice features + penalised logistic regression (foundation-model baseline)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import metrics
from .models import SLICE_MODELS, WHOLE_MODELS, build
from .pipeline2 import LABELS, Cases2, Data2, Runner, _dev, _late
from .prep2 import KIN_NAMES
from .train import predict, seed_all

LABELS.update({"carfm_k": "CARF-mix, kinetic", "carfm_25d": "CARF-mix, pretrained 2.5D",
               "wb_img": "Whole-breast 3D ResNet, image only", "wb_fusion": "Whole-breast 3D ResNet + clinical"})


class CasesWB(Cases2):
    """Whole-breast volumes: left-right flips only (anatomy is in LPS order), intensity jitter."""

    def __getitem__(self, i):
        img = torch.as_tensor(np.ascontiguousarray(self.loader(i)), dtype=torch.float32)
        if self.augment:
            if random.random() < 0.5:
                img = img.flip(3)
            img = img * (1 + 0.1 * torch.randn(1))
        return {"image": img, "tab": self.tab[i], "tab_missing": self.miss[i], "dt": self.dt[i],
                "z_clin": self.z[i], "y": self.y[i]}


class Runner3(Runner):
    def __init__(self, data, out, cfg, log, whole=None):
        super().__init__(data, out, cfg, log)
        self.whole = whole  # (N, 48, 160, 160) uint8 aligned to data.table, or None
        self.dep_rows = []

    def ds(self, name, idx, tab, miss, z, y, augment=False):
        if name in WHOLE_MODELS:
            W = self.whole
            return CasesWB(None, tab[idx], miss[idx], self.dt[idx], z[idx], y[idx], augment,
                           loader=lambda j, idx=idx: W[int(idx[j])][None].astype(np.float32) / 85.0, kind="wb")
        return super().ds(name, idx, tab, miss, z, y, augment)

    # ------------------------------------------------------------- input dependency (reviewer 1, 2)
    def score(self, model, name, idx, tab, miss, z, y):
        p, g = super().score(model, name, idx, tab, miss, z, y)
        if name.startswith("carf"):
            self.dependency(model, name, idx, tab, miss, z, y)
        return p, g

    def dependency(self, model, name, idx, tab, miss, z, y):
        rng = np.random.default_rng(len(self.dep_rows))
        bs, dev = self.cfg["batch_size"], _dev()
        base = predict(model, self.ds(name, idx, tab, miss, z, y), batch_size=bs, device=dev)
        perm = rng.permutation(len(idx))
        t2, m2 = tab.copy(), miss.copy()
        t2[idx], m2[idx] = tab[idx[perm]], miss[idx[perm]]
        r_tab = predict(model, self.ds(name, idx, t2, m2, z, y), batch_size=bs, device=dev)
        get = self.data.slices if name in SLICE_MODELS else self.data.vol6
        ds_img = Cases2(None, tab[idx], miss[idx], self.dt[idx], z[idx], y[idx], False,
                        loader=lambda j: get(int(idx[perm[j]])), kind="sl" if name in SLICE_MODELS else "vol")
        r_img = predict(model, ds_img, batch_size=bs, device=dev)
        gd = base["gate"] * base["delta"]
        row = {"model": LABELS[name], "set": "test fold" if len(self.dep_rows) % 2 == 0 else "external", "n": len(idx),
               "mean_abs_change_logit_tab_permuted": float(np.abs(r_tab["logit"] - base["logit"]).mean()),
               "mean_abs_change_logit_image_permuted": float(np.abs(r_img["logit"] - base["logit"]).mean()),
               "mean_abs_gdelta": float(np.abs(gd).mean()), "p99_abs_gdelta": float(np.percentile(np.abs(gd), 99)),
               "max_abs_gdelta": float(np.abs(gd).max()), "mean_gate": float(base["gate"].mean())}
        self.dep_rows.append(row)
        pd.DataFrame(self.dep_rows).to_csv(os.path.join(self.out, f"dependency_{self.tag}.csv"), index=False)

    # ------------------------------------------------------------- positive controls (reviewer 4)
    def posctl(self, models, tag):
        c, tb = self.cfg, self.tb
        dev_all = (tb.collection == c["dev"]).to_numpy()
        tgt = {}
        for f in ("pe_mean", "frac_washout"):
            v = self.Xk[:, KIN_NAMES.index(f)]
            med = np.nanmedian(v[dev_all])
            tgt[f + "_high"] = np.where(np.isnan(v), np.nan, (v > med).astype(float))
        tgt["vendor_ge"] = np.where(tb.vendor.isna(), np.nan, (tb.vendor.astype(str).str.upper() == "GE").astype(float))
        tab = np.zeros((len(tb), 1), np.float32); miss = np.zeros_like(tab); z = np.zeros(len(tb), np.float32)
        rows = []
        for name in models:
            for t, yv in tgt.items():
                ok = ~np.isnan(yv)
                y = np.nan_to_num(yv).astype(int)
                dev = np.flatnonzero(dev_all & ok)
                ext = np.flatnonzero(~dev_all & ok)
                cv = StratifiedKFold(c["control_folds"], shuffle=True, random_state=0)
                o, e, n, done = np.zeros(len(dev)), np.zeros(len(ext)), 0, True
                for k, (tr, te) in enumerate(cv.split(dev, y[dev])):
                    if _late():
                        done = False
                        break
                    tr2, va = self.inner_split(tr, k)
                    model = self.train_one(name, dev[tr2], dev[va], tab, miss, z, y, k, f"pos_{name}_{t}_f{k}")
                    o[te] = Runner.score(self, model, name, dev[te], tab, miss, z, y)[0]
                    e += Runner.score(self, model, name, ext, tab, miss, z, y)[0]
                    n += 1
                    del model
                if not done:
                    self.log(f"[posctl {name} {t}] SKIPPED: time budget")
                    continue
                rows.append(pd.DataFrame({"model": LABELS[name], "target": t, "case_id": tb.case_id.iloc[dev],
                                          "collection": tb.collection.iloc[dev], "set": "dev", "y": y[dev], "p": o}))
                if len(ext) and len(np.unique(y[ext])) > 1:
                    rows.append(pd.DataFrame({"model": LABELS[name], "target": t, "case_id": tb.case_id.iloc[ext],
                                              "collection": tb.collection.iloc[ext], "set": "ext", "y": y[ext], "p": e / n}))
                pd.concat(rows).to_csv(os.path.join(self.out, f"preds_posctl_{tag}.csv"), index=False)
                self.log(f"[posctl {name} {t}] internal AUC {roc_auc_score(y[dev], o):.3f}")

    # ------------------------------------------------------------- learning curves (reviewer 4)
    def lc(self, models, tag, fractions=(0.25, 0.5)):
        c, tb = self.cfg, self.tb
        keep = tb.pcr.notna().to_numpy()
        y = tb.pcr.fillna(0).to_numpy().astype(int)
        dev = np.flatnonzero((tb.collection == c["dev"]).to_numpy() & keep)
        ext = np.flatnonzero((tb.collection != c["dev"]).to_numpy() & keep)
        cv = StratifiedKFold(c["folds"], shuffle=True, random_state=0)
        Xt = np.hstack([self.Xc, self.Xk])
        from .pipeline2 import _standardise
        tab, miss = _standardise(Xt, Xt[dev])
        z = np.zeros(len(tb), np.float32)
        preds, trains = [], []
        for name in models:
            for frac in fractions:
                o, e, n, done = np.zeros(len(dev)), np.zeros(len(ext)), 0, True
                for k, (tr, te) in enumerate(cv.split(dev, y[dev])):
                    if _late():
                        done = False
                        break
                    rng = np.random.default_rng(1000 + k)
                    sub = np.concatenate([rng.choice(tr[y[dev[tr]] == v], int(round(frac * (y[dev[tr]] == v).sum())), replace=False)
                                          for v in (0, 1)])
                    tr2, va = self.inner_split(sub, k)
                    model = self.train_one(name, dev[tr2], dev[va], tab, miss, z, y, k, f"lc_{name}_{frac}_f{k}")
                    o[te] = Runner.score(self, model, name, dev[te], tab, miss, z, y)[0]
                    e += Runner.score(self, model, name, ext, tab, miss, z, y)[0]
                    ptr = Runner.score(self, model, name, dev[tr2], tab, miss, z, y)[0]
                    trains.append({"model": LABELS[name], "fraction": frac, "fold": k, "n_train": len(tr2),
                                   "train_auc": roc_auc_score(y[dev[tr2]], ptr), "test_fold_auc": roc_auc_score(y[dev[te]], o[te])})
                    n += 1
                    del model
                    pd.DataFrame(trains).to_csv(os.path.join(self.out, f"lc_train_{tag}.csv"), index=False)
                if not done:
                    self.log(f"[lc {name} {frac}] SKIPPED: time budget")
                    continue
                preds.append(pd.DataFrame({"model": LABELS[name], "fraction": frac, "case_id": tb.case_id.iloc[dev], "set": "dev",
                                           "collection": tb.collection.iloc[dev], "y": y[dev], "p": o}))
                preds.append(pd.DataFrame({"model": LABELS[name], "fraction": frac, "case_id": tb.case_id.iloc[ext], "set": "ext",
                                           "collection": tb.collection.iloc[ext], "y": y[ext], "p": e / n}))
                pd.concat(preds).to_csv(os.path.join(self.out, f"preds_lc_{tag}.csv"), index=False)
                self.log(f"[lc {name} {frac}] OOF AUC {roc_auc_score(y[dev], o):.3f}")

    # ------------------------------------------------------------- overfitting sanity check (reviewer 4)
    def overfit(self, models, tag, n_train=120, n_hold=200, epochs=60):
        c, tb = self.cfg, self.tb
        keep = tb.pcr.notna().to_numpy()
        y = tb.pcr.fillna(0).to_numpy().astype(int)
        dev = np.flatnonzero((tb.collection == c["dev"]).to_numpy() & keep)
        rng = np.random.default_rng(3)
        dev = rng.permutation(dev)
        tr, ho = dev[:n_train], dev[n_train:n_train + n_hold]
        tab = np.zeros((len(tb), 1), np.float32); miss = np.zeros_like(tab); z = np.zeros(len(tb), np.float32)
        rows = []
        for name in models:
            seed_all(0)
            model = build(name, 1, tuple(c["widths"]), c["dim"]).to(_dev())
            opt = torch.optim.AdamW(model.parameters(), lr=c["lr_pretrained"] if name in SLICE_MODELS else c["lr"], weight_decay=0.0)
            dl = torch.utils.data.DataLoader(self.ds(name, tr, tab, miss, z, y, augment=False), batch_size=c["batch_size"], shuffle=True)
            use_amp = _dev() == "cuda"
            scaler = torch.amp.GradScaler(enabled=use_amp)
            for ep in range(epochs):
                if _late():
                    break
                model.train()
                for b in dl:
                    b = {k: v.to(_dev()) for k, v in b.items()}
                    with torch.autocast(device_type=_dev(), enabled=use_amp):
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(b)["logit"].float(), b["y"])
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(opt); scaler.update()
                if ep % 5 == 4 or ep == epochs - 1:
                    ptr = Runner.score(self, model, name, tr, tab, miss, z, y)[0]
                    pho = Runner.score(self, model, name, ho, tab, miss, z, y)[0]
                    rows.append({"model": LABELS.get(name, name), "epoch": ep + 1, "train_auc": roc_auc_score(y[tr], ptr),
                                 "heldout_auc": roc_auc_score(y[ho], pho), "loss": float(loss)})
                    self.log(f"[overfit {name}] epoch {ep + 1} train {rows[-1]['train_auc']:.3f} held-out {rows[-1]['heldout_auc']:.3f}")
                    pd.DataFrame(rows).to_csv(os.path.join(self.out, f"overfit_{tag}.csv"), index=False)
            del model

    # ------------------------------------------------------------- foundation-model baseline (reviewer 4)
    def dino(self, tag):
        import torch.nn.functional as F

        c, tb = self.cfg, self.tb
        dev_ = _dev()
        try:
            net = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True)
            src = "torch.hub facebookresearch/dinov2 dinov2_vitb14"
        except Exception as e:  # noqa: BLE001
            import timm

            self.log(f"torch.hub failed ({e!r}); using timm")
            net = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224)
            src = "timm vit_base_patch14_dinov2.lvd142m"
        net = net.eval().to(dev_)
        mean = torch.tensor([0.485, 0.456, 0.406], device=dev_).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=dev_).view(1, 3, 1, 1)
        feats = []
        with torch.no_grad():
            for i in range(len(tb)):
                x = torch.as_tensor(self.data.slices(i) + 1.0, device=dev_)  # (12, 3, P, P), signal / mean pre
                lo = x.flatten(2).amin(2)[..., None, None]
                hi = x.flatten(2).amax(2)[..., None, None]
                x = (x - lo) / (hi - lo + 1e-6)
                x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
                f = net((x - mean) / std)
                feats.append(f.float().mean(0).cpu().numpy())
                if (i + 1) % 300 == 0:
                    self.log(f"[dino] {i + 1}/{len(tb)}")
        Fm = np.stack(feats)
        np.save(os.path.join(self.out, f"dino_feats_{tag}.npy"), Fm.astype(np.float16))
        keep = tb.pcr.notna().to_numpy()
        y = tb.pcr.fillna(0).to_numpy().astype(int)
        dev = np.flatnonzero((tb.collection == c["dev"]).to_numpy() & keep)
        ext = np.flatnonzero((tb.collection != c["dev"]).to_numpy() & keep)
        cv = StratifiedKFold(c["folds"], shuffle=True, random_state=0)
        Xc = np.nan_to_num(self.Xc - np.nanmean(self.Xc[dev], 0))
        P = pd.DataFrame({"case_id": tb.case_id.iloc[np.r_[dev, ext]].to_numpy(), "collection": tb.collection.iloc[np.r_[dev, ext]].to_numpy(),
                          "y": y[np.r_[dev, ext]], "set": ["dev"] * len(dev) + ["ext"] * len(ext)})
        for label, X in [("DINOv2 linear probe, image only", Fm), ("DINOv2 linear probe + clinical", np.hstack([Fm, Xc]))]:
            mk = lambda: make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=[1e-4, 1e-3, 1e-2, 1e-1, 1], cv=5, scoring="roc_auc",  # noqa: E731
                                                                              max_iter=5000))
            o = np.zeros(len(dev))
            for tr, te in cv.split(dev, y[dev]):
                o[te] = mk().fit(X[dev[tr]], y[dev[tr]]).predict_proba(X[dev[te]])[:, 1]
            m = mk().fit(X[dev], y[dev])
            P[label] = np.r_[o, m.predict_proba(X[ext])[:, 1]]
            self.log(f"[dino] {label}: OOF AUC {roc_auc_score(y[dev], o):.3f}; C {m[-1].C_[0]}")
        P.to_csv(os.path.join(self.out, f"preds_dino_{tag}.csv"), index=False)
        json.dump({"source": src}, open(os.path.join(self.out, f"dino_{tag}.json"), "w"))


def load_whole(path, table):
    ids = pd.read_csv(os.path.join(path, "wb_ids.csv"))
    arr = np.load(os.path.join(path, "wb.npy"), mmap_mode="r")
    pos = {c: k for k, c in enumerate(ids.case_id)}
    out = np.zeros((len(table),) + arr.shape[1:], np.uint8)
    for i, cid in enumerate(table.case_id):
        if cid in pos:
            out[i] = arr[pos[cid]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--whole")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exp", required=True, choices=["devext", "posctl", "lc", "overfit", "dino"])
    ap.add_argument("--models", default="")
    ap.add_argument("--tag", default="a")
    ap.add_argument("--quick", action="store_true", help="1 epoch, 2 folds: a test of the real-data path")
    a = ap.parse_args()
    cfg = {"epochs": 1, "folds": 2, "control_folds": 2, "patience": 1} if a.quick else {}
    whole = None
    if a.synthetic:
        from .synthetic import make_cohort2

        data = make_cohort2()
        cfg = {"epochs": 2, "folds": 2, "control_folds": 2, "widths": [8, 16], "dim": 16, "patience": 2, "batch_size": 8}
        whole = np.random.default_rng(0).integers(0, 255, (len(data.table), 48, 160, 160), dtype=np.uint8) if "wb" in a.models else None
    else:
        data = Data2(a.data)
        if a.whole:
            whole = load_whole(a.whole, data.table)
    os.makedirs(a.out, exist_ok=True)
    log = lambda *m: print(time.strftime("%H:%M:%S"), f"<{a.tag}>", *m, flush=True)  # noqa: E731
    log(f"{len(data.table)} cases; exp {a.exp}; models {a.models}; device {_dev()}")
    r = Runner3(data, a.out, cfg, log, whole)
    r.tag = a.tag
    models = [m for m in a.models.split(",") if m]
    t0 = time.time()
    if a.exp == "devext":
        r.devext(models, a.tag)
    elif a.exp == "posctl":
        r.posctl(models, a.tag)
    elif a.exp == "lc":
        r.lc(models, a.tag)
    elif a.exp == "overfit":
        r.overfit(models, a.tag, **({"n_train": 16, "n_hold": 16, "epochs": 2} if (a.synthetic or a.quick) else {}))
    else:
        r.dino(a.tag)
    log(f"total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
