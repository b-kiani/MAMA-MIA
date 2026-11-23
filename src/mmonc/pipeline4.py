"""Second-review experiments (exploratory, post hoc). See paper/analysis_plan_revision2.md.

    python -m mmonc.pipeline4 --data P2 --out O --exp opt    --configs base,long_auc,long_loss --tag a
    python -m mmonc.pipeline4 --data P2 --out O --exp nested --models carf_k,carfm_k,carf_25d,carfm_25d --tag b
    python -m mmonc.pipeline4 --data P2 --out O --exp overfit --tag c
    python -m mmonc.pipeline4 --synthetic --out O --exp opt --configs base,fixed --tag t   (smoke test)

opt     : optimization study for the kinetic 3D ResNet trained from scratch. Each configuration is
          evaluated by fivefold cross-validation on I-SPY2 (the same folds as before); the configuration
          used for the external comparison is the one with the highest mean I-SPY2 out-of-fold AUC.
          All configurations are scored externally and all results are reported.
nested  : the anchored models refitted with fully nested preprocessing. In each outer fold, the clinical
          anchor of the network-training patients is cross-fitted within those patients only; the
          early-stopping and test patients receive logits from an anchor fitted on the training patients;
          tabular standardisation uses training-patient statistics only.
overfit : can the 3D network fit 120 patients when trained long enough? (no augmentation or weight decay)

Training runs entirely on the GPU: all crops are cached there as float16 and augmented per batch
(per-sample flips over the spatial axes; intensity scale and shift with the mask channel untouched),
matching mmonc.pipeline2.Cases2 in distribution.
"""
from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from . import metrics
from .models import SLICE_MODELS, ClinicalAnchor, build
from .pipeline2 import LABELS, Data2, _late, _standardise
from .train import seed_all

LABELS.update({"carfm_k": "CARF-mix, kinetic", "carfm_25d": "CARF-mix, pretrained 2.5D"})

# Configurations fixed before the run (analysis_plan_revision2.md). "base" is the round-2 protocol.
CONFIGS = {
    "base": dict(lr=3e-4, wd=1e-2, epochs=60, warmup=0, stop="auc", patience=12, widths=(32, 64, 128, 256)),
    "long_auc": dict(lr=1e-3, wd=1e-4, epochs=150, warmup=5, stop="auc", patience=40, widths=(32, 64, 128, 256)),
    "long_loss": dict(lr=1e-3, wd=1e-4, epochs=150, warmup=5, stop="loss", patience=40, widths=(32, 64, 128, 256)),
    "fixed": dict(lr=1e-3, wd=1e-4, epochs=150, warmup=5, stop=None, patience=0, widths=(32, 64, 128, 256)),
    "small_loss": dict(lr=1e-3, wd=1e-4, epochs=150, warmup=5, stop="loss", patience=40, widths=(16, 32, 64, 128)),
}
ANCHORED = dict(lr=3e-4, lr_pretrained=1e-4, wd=1e-2, epochs=60, warmup=0, stop="auc", patience=12, widths=(32, 64, 128, 256))


def _dev():
    return "cuda" if torch.cuda.is_available() else "cpu"


class GPUCache:
    """All crops of one kind, float16, on the training device."""

    def __init__(self, data, kind, device):
        get = data.vol6 if kind == "vol" else data.slices
        self.kind = kind
        self.x = torch.stack([torch.from_numpy(np.ascontiguousarray(get(i))).half() for i in range(len(data.table))]).to(device)

    def batch(self, idx, augment):
        x = self.x[torch.as_tensor(idx, device=self.x.device)].float()
        if augment:
            n = x.shape[0]
            axes = (2, 3, 4) if self.kind == "vol" else (3, 4)
            for ax in axes:
                m = (torch.rand(n, device=x.device) < 0.5).view(-1, *([1] * (x.dim() - 1)))
                x = torch.where(m, x.flip(ax), x)
            s = (1 + 0.1 * torch.randn(n, device=x.device)).view(-1, *([1] * (x.dim() - 1)))
            b = (0.1 * torch.randn(n, device=x.device)).view(-1, *([1] * (x.dim() - 1)))
            if self.kind == "vol":
                x = torch.cat([x[:, :3] * s + b, x[:, 3:5] * s, x[:, 5:]], 1)
            else:
                x = x * s + b
        return x


class Fast:
    def __init__(self, data, out, log, synthetic=False):
        self.data, self.out, self.log, self.synthetic = data, out, log, synthetic
        tb = data.table
        self.tb = tb
        self.Xc = tb[["age", "hr", "her2"]].to_numpy(float)
        self.Xk = np.asarray(data.kin, float)
        cols = [c for c in ("dt0", "dt1", "dt_last") if c in tb.columns]
        self.dt = np.nan_to_num(tb[cols].to_numpy(float) if len(cols) == 3 else np.zeros((len(tb), 3)), nan=0.0).astype(np.float32)
        keep = tb.pcr.notna().to_numpy()
        self.y = tb.pcr.fillna(0).to_numpy().astype(np.float32)
        self.dev = np.flatnonzero((tb.collection == "ISPY2").to_numpy() & keep)
        self.ext = np.flatnonzero((tb.collection != "ISPY2").to_numpy() & keep)
        self.cv = StratifiedKFold(2 if synthetic else 5, shuffle=True, random_state=0)
        self.cache = {}
        os.makedirs(out, exist_ok=True)

    def imgs(self, kind):
        if kind not in self.cache:
            t0 = time.time()
            self.cache[kind] = GPUCache(self.data, kind, _dev())
            self.log(f"cached {kind} crops on {_dev()}: {tuple(self.cache[kind].x.shape)} in {time.time() - t0:.0f} s")
        return self.cache[kind]

    @staticmethod
    def inner_split(tr, seed):
        perm = np.random.default_rng(seed).permutation(tr)
        n_val = max(2, int(0.15 * len(tr)))
        return perm[n_val:], perm[:n_val]

    # ---------------------------------------------------------------- training on the GPU
    def _batch(self, kind, idx, tab, miss, z, augment):
        d = _dev()
        T = lambda a: torch.as_tensor(a[idx], device=d)  # noqa: E731
        return {"image": self.imgs(kind).batch(idx, augment), "tab": T(tab), "tab_missing": T(miss), "dt": T(self.dt),
                "z_clin": T(z), "y": T(self.y)}

    @torch.no_grad()
    def predict(self, model, kind, idx, tab, miss, z, bs=32):
        model.eval()
        out = {}
        for s in range(0, len(idx), bs):
            b = self._batch(kind, idx[s:s + bs], tab, miss, z, False)
            with torch.autocast(device_type=_dev(), enabled=_dev() == "cuda"):
                r = model(b)
            for k, v in r.items():
                if v.dim() == 1:
                    out.setdefault(k, []).append(v.float().cpu().numpy())
        return {k: np.concatenate(v) for k, v in out.items()}

    def fit(self, model, kind, tr, va, tab, miss, z, cfg, seed, lr=None, bs=16):
        """AdamW with optional linear warm-up and cosine decay. Early stopping on inner-validation AUC
        or loss (keeping the best weights), or none (keeping the final weights)."""
        seed_all(seed)
        d = _dev()
        model.to(d)
        lr = lr or cfg["lr"]
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["wd"])
        E, W = cfg["epochs"], cfg["warmup"]
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: (e + 1) / W if e < W else 0.5 * (1 + math.cos(math.pi * (e - W) / max(1, E - W))))
        amp = d == "cuda"
        scaler = torch.amp.GradScaler(enabled=amp)
        aux_w = getattr(model, "aux_weight", 0.5)
        rng = np.random.default_rng(seed)
        best, best_state, bad, hist = None, None, 0, []
        yv = self.y[va]
        for ep in range(E):
            model.train()
            perm = rng.permutation(tr)
            nb = len(perm) // bs if len(perm) > bs else 1
            tot = 0.0
            for j in range(nb):
                b = self._batch(kind, perm[j * bs:(j + 1) * bs], tab, miss, z, True)
                with torch.autocast(device_type=d, enabled=amp):
                    r = model(b)
                    loss = F.binary_cross_entropy_with_logits(r["logit"].float(), b["y"])
                    if "gate" in r:
                        aux = F.binary_cross_entropy_with_logits((b["z_clin"] + r["delta"]).float(), b["y"])
                        loss = loss + aux_w * aux + getattr(model, "gate_penalty", 0.01) * r["gate"].float().mean()
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
                tot += loss.item()
            sched.step()
            lg = self.predict(model, kind, va, tab, miss, z)["logit"]
            auc = roc_auc_score(yv, lg) if len(np.unique(yv)) > 1 else float("nan")
            vloss = float(F.binary_cross_entropy_with_logits(torch.as_tensor(lg), torch.as_tensor(yv)).item())
            hist.append({"epoch": ep, "train_loss": tot / nb, "val_auc": auc, "val_loss": vloss})
            if cfg["stop"] is None:
                continue
            score = auc if cfg["stop"] == "auc" else -vloss
            if best is None or score > best:
                best, bad = score, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg["patience"]:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        kept = len(hist) - 1 if cfg["stop"] is None else int(np.argmax([h["val_auc"] if cfg["stop"] == "auc" else -h["val_loss"] for h in hist]))
        return hist, kept

    # ---------------------------------------------------------------- optimization study
    def opt(self, configs, tag):
        y, dev, ext = self.y, self.dev, self.ext
        Xt = np.hstack([self.Xc, self.Xk])
        P = pd.DataFrame({"case_id": self.tb.case_id.iloc[np.r_[dev, ext]].to_numpy(), "collection": self.tb.collection.iloc[np.r_[dev, ext]].to_numpy(),
                          "y": y[np.r_[dev, ext]].astype(int), "set": ["dev"] * len(dev) + ["ext"] * len(ext)})
        rows = []
        for cname in configs:
            cfg = dict(CONFIGS[cname])
            if self.synthetic:
                cfg.update(epochs=2, warmup=min(cfg["warmup"], 1), widths=(8, 16))
            o, e, n = np.zeros(len(dev)), np.zeros(len(ext)), 0
            for k, (tr, te) in enumerate(self.cv.split(dev, y[dev])):
                if _late():
                    self.log(f"[opt {cname}] SKIPPED: time budget")
                    break
                tr2, va = self.inner_split(tr, k)
                tab, miss = _standardise(Xt, Xt[dev[tr2]])
                z = np.zeros(len(y), np.float32)
                model = build("image_only_k", tab.shape[1], cfg["widths"], 128)
                t0 = time.time()
                hist, kept = self.fit(model, "vol", dev[tr2], dev[va], tab, miss, z, cfg, seed=k)
                p_tr = self.predict(model, "vol", dev[tr2], tab, miss, z)["logit"]
                o[te] = metrics.sigmoid(self.predict(model, "vol", dev[te], tab, miss, z)["logit"])
                e += metrics.sigmoid(self.predict(model, "vol", ext, tab, miss, z)["logit"])
                n += 1
                rows.append({"config": cname, "fold": k, "epochs_run": len(hist), "epoch_kept": kept,
                             "train_auc": roc_auc_score(y[dev[tr2]], p_tr), "inner_val_auc": hist[kept]["val_auc"],
                             "test_fold_auc": roc_auc_score(y[dev[te]], o[te]), "minutes": (time.time() - t0) / 60})
                pd.DataFrame(rows).to_csv(os.path.join(self.out, f"opt_train_{tag}.csv"), index=False)
                self.log(f"[opt {cname} f{k}] {len(hist)} epochs (kept {kept}); train AUC {rows[-1]['train_auc']:.3f}; "
                         f"test fold {rows[-1]['test_fold_auc']:.3f}; {rows[-1]['minutes']:.1f} min")
                del model
                torch.cuda.empty_cache()
            if n == self.cv.get_n_splits():
                P[f"opt:{cname}"] = np.r_[o, e / n]
                P.to_csv(os.path.join(self.out, f"preds_opt_{tag}.csv"), index=False)
                self.log(f"[opt {cname}] I-SPY2 OOF AUC {roc_auc_score(y[dev], o):.3f}")

    # ---------------------------------------------------------------- anchored models, fully nested
    def nested(self, models, tag):
        y, dev, ext = self.y, self.dev, self.ext
        Xt = np.hstack([self.Xc, self.Xk])
        full = ClinicalAnchor(n_splits=5)
        full.fit(self.Xc[dev], y[dev].astype(int))
        z_ext_full = full.logit(self.Xc[ext]).astype(np.float32)
        P = pd.DataFrame({"case_id": self.tb.case_id.iloc[np.r_[dev, ext]].to_numpy(), "collection": self.tb.collection.iloc[np.r_[dev, ext]].to_numpy(),
                          "y": y[np.r_[dev, ext]].astype(int), "set": ["dev"] * len(dev) + ["ext"] * len(ext)})
        oc = np.zeros(len(dev))
        for name in models:
            kind = "sl" if name in SLICE_MODELS else "vol"
            cfg = dict(ANCHORED)
            if self.synthetic:
                cfg.update(epochs=2, widths=(8, 16))
            o, e, n = np.zeros(len(dev)), np.zeros(len(ext)), 0
            for k, (tr, te) in enumerate(self.cv.split(dev, y[dev])):
                if _late():
                    self.log(f"[nested {name}] SKIPPED: time budget")
                    break
                tr2, va = self.inner_split(tr, k)
                z = np.zeros(len(y), np.float32)
                anc = ClinicalAnchor(n_splits=5)
                z[dev[tr2]] = anc.fit(self.Xc[dev[tr2]], y[dev[tr2]].astype(int))  # cross-fitted within training patients
                z[dev[va]] = anc.logit(self.Xc[dev[va]])
                z[dev[te]] = anc.logit(self.Xc[dev[te]])
                z[ext] = z_ext_full
                oc[te] = metrics.sigmoid(z[dev[te]])
                tab, miss = _standardise(Xt, Xt[dev[tr2]])
                model = build(name, tab.shape[1], cfg["widths"], 128)
                t0 = time.time()
                hist, kept = self.fit(model, kind, dev[tr2], dev[va], tab, miss, z, cfg, seed=k,
                                      lr=cfg["lr_pretrained"] if name in SLICE_MODELS else cfg["lr"])
                o[te] = metrics.sigmoid(self.predict(model, kind, dev[te], tab, miss, z)["logit"])
                e += metrics.sigmoid(self.predict(model, kind, ext, tab, miss, z)["logit"])
                n += 1
                self.log(f"[nested {name} f{k}] {len(hist)} epochs; {(time.time() - t0) / 60:.1f} min")
                del model
                torch.cuda.empty_cache()
            if n == self.cv.get_n_splits():
                P[f"nested:{LABELS[name]}"] = np.r_[o, e / n]
                P["nested:Clinical"] = np.r_[oc, metrics.sigmoid(z_ext_full)]
                P.to_csv(os.path.join(self.out, f"preds_nested_{tag}.csv"), index=False)
                self.log(f"[nested {name}] I-SPY2 OOF AUC {roc_auc_score(y[dev], o):.3f} (clinical, nested {roc_auc_score(y[dev], oc):.3f})")

    # ---------------------------------------------------------------- overfitting test
    def overfit(self, tag, n_train=120, n_hold=200, epochs=200):
        y, dev = self.y, self.dev
        rng = np.random.default_rng(0)
        perm = rng.permutation(dev)
        tr, ho = perm[:n_train], perm[n_train:n_train + n_hold]
        tab = np.zeros((len(y), 1), np.float32); miss = tab.copy(); z = np.zeros(len(y), np.float32)
        rows = []
        for lr in (1e-3, 3e-4):
            cfg = dict(lr=lr, wd=0.0, epochs=2 if self.synthetic else epochs, warmup=0, stop=None, patience=0)
            seed_all(0)
            model = build("image_only_k", 1, (8, 16) if self.synthetic else (32, 64, 128, 256), 128).to(_dev())
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
            amp = _dev() == "cuda"
            scaler = torch.amp.GradScaler(enabled=amp)
            for ep in range(cfg["epochs"]):
                model.train()
                p = rng.permutation(tr)
                for j in range(0, len(p), 16):
                    b = self._batch("vol", p[j:j + 16], tab, miss, z, False)
                    with torch.autocast(device_type=_dev(), enabled=amp):
                        loss = F.binary_cross_entropy_with_logits(model(b)["logit"].float(), b["y"])
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                if (ep + 1) % 10 == 0 or ep == cfg["epochs"] - 1:
                    a_tr = roc_auc_score(y[tr], self.predict(model, "vol", tr, tab, miss, z)["logit"])
                    a_ho = roc_auc_score(y[ho], self.predict(model, "vol", ho, tab, miss, z)["logit"])
                    rows.append({"lr": lr, "epoch": ep + 1, "train_auc": a_tr, "heldout_auc": a_ho})
                    self.log(f"[overfit lr {lr}] epoch {ep + 1} train {a_tr:.3f} held-out {a_ho:.3f}")
            pd.DataFrame(rows).to_csv(os.path.join(self.out, f"overfit2_{tag}.csv"), index=False)
            del model
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exp", required=True, choices=["opt", "nested", "overfit"])
    ap.add_argument("--configs", default="")
    ap.add_argument("--models", default="")
    ap.add_argument("--tag", default="a")
    a = ap.parse_args()
    if a.synthetic:
        from .synthetic import make_cohort2

        data = make_cohort2()
    else:
        data = Data2(a.data)
    log = lambda *m: print(time.strftime("%H:%M:%S"), f"<{a.tag}>", *m, flush=True)  # noqa: E731
    log(f"{len(data.table)} cases; exp {a.exp}; device {_dev()}")
    r = Fast(data, a.out, log, a.synthetic)
    t0 = time.time()
    if a.exp == "opt":
        r.opt([c for c in a.configs.split(",") if c], a.tag)
    elif a.exp == "nested":
        r.nested([m for m in a.models.split(",") if m], a.tag)
    else:
        r.overfit(a.tag, **({"n_train": 16, "n_hold": 16} if a.synthetic else {}))
    log(f"total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
