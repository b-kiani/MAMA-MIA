"""Third-review experiments (exploratory, post hoc). See paper/analysis_plan_revision3.md.

    python -m mmonc.pipeline5 --root P4 --out O --queue q0 --tag g0      (P4 written by mmonc.prep4)
    python -m mmonc.pipeline5 --synthetic --out O --queue smoke --tag t  (code test)

Each queue is a fixed, ordered list of runs (QUEUES below, fixed before the run). A run is one model
trained by fivefold cross-validation on the development patients (I-SPY2 unless a subset is given), with
  variant : input version from prep4 (orig = stored voxel order as in round 2; lps = common anatomical
            orientation; lps_fov = lps with a fixed 96-mm field of view)
  seed    : 0 reproduces the round-2/3 folds and initialisation; seed r > 0 changes the outer folds
            (StratifiedKFold random_state=r), the inner early-stopping split and the initialisation
  labels  : real pCR; ("sig", beta, draw) semi-synthetic outcome y* ~ Bernoulli(sigmoid(z + beta * s)),
            z = clinical-model logit fitted on the real I-SPY2 labels, s = mean early percentage enhancement
            in the tumour, residualised on age/HR/HER2 and standardised on I-SPY2; ("null", perm) real pCR
            with the images permuted within collection x clinical-risk quintile
  tab     : tabular inputs of the anchored models: clinical + kinetic ("clinkin") or clinical only ("clin")
  subset  : development subset ("ctl" = I-SPY2 control arm; "rnd<r>" = random subset of the same size)
  folds   : "regimen" = I-SPY2 regimen-group-held-out folds instead of random folds
Anchored models use fully nested anchors (as mmonc.pipeline4.nested). For every run the paired nested
clinical model on the same folds and labels is stored too. Each run writes one predictions file
(out-of-fold for development, mean of the five fold models for external, every fold model separately,
and for anchored models the correction gate * delta and the gate) and one summary row.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from . import metrics
from .models import SLICE_MODELS, ClinicalAnchor, build
from .pipeline2 import Data2, _late, _standardise
from .pipeline4 import ANCHORED, CONFIGS, Fast, _dev

# Signal strengths fixed before the run: beta giving an expected external oracle AUC gain of
# 0, 0.02, 0.035, 0.05 and 0.10 (calibrated by simulation on the round-2 features, analysis plan).
BETAS = {"0": 0.0, "0.02": 0.22, "0.035": 0.33, "0.05": 0.43, "0.10": 0.74}
CONTROL_ARM = ("Paclitaxel", "Paclitaxel + Trastuzumab")
REGIMEN_GROUPS = {  # I-SPY2 regimen groups for regimen-held-out folds
    "control": CONTROL_ARM,
    "her2_targeted": ("T-DM1 + Pertuzumab", "Paclitaxel + Pertuzumab + Trastuzumab", "Paclitaxel + MK-2206 + Trastuzumab",
                      "Paclitaxel + AMG 386 + Trastuzumab", "Paclitaxel + Neratinib"),
    "immunotherapy": ("Paclitaxel + Pembrolizumab",),
    "other_experimental": ("Paclitaxel + AMG 386", "Paclitaxel + Ganitumab", "Paclitaxel + Ganetespib",
                           "Paclitaxel + ABT 888 + Carboplatin", "Paclitaxel + MK-2206"),
}
ABL = [f"abl25d_m{m}b{b}g{g}" for m in (0, 1) for b in (0, 1) for g in (0, 1) if (m, b, g) != (1, 1, 1)]  # (1,1,1) = carfm_25d
SIG = [("sig", k, d) for d in (0, 1) for k in BETAS]


def R(model, variant="lps", seed=0, labels="real", tab="clinkin", subset=None, folds=None):
    return dict(model=model, variant=variant, seed=seed, labels=labels, tab=tab, subset=subset, folds=folds)


def _queue_q0():
    q = [R(m, v) for m in ("s25d", "carfm_25d") for v in ("orig", "lps")]                                  # orientation, seed 0
    q += [R(m, labels=lab, tab="clin") for lab in SIG[:5] for m in ("s25d", "carfm_25d")]                  # signal recovery, draw 0
    q += [R(m, labels=("null", 0), tab="clin") for m in ("s25d", "carfm_25d")]                             # permutation null
    q += [R(m, v, seed=s) for v in ("lps", "orig") for s in (1, 2, 3, 4) for m in ("s25d", "carfm_25d")]   # repeated training
    q += [R(m) for m in ABL]                                                                               # ablation, real labels
    q += [R(m, labels=lab, tab="clin") for lab in SIG[5:] for m in ("s25d", "carfm_25d")]                  # signal recovery, draw 1
    q += [R(m, labels=("null", 1), tab="clin") for m in ("s25d", "carfm_25d")]
    q += [R(m, seed=1) for m in ABL]                                                                       # ablation, seed 1
    q += [R(m, "lps_fov", seed=s) for s in (0, 1, 2) for m in ("s25d", "carfm_25d")]                       # fixed field of view
    return q


def _queue_q1():
    q = [R("carfm_k"), R("image_only_k")]                                                                 # 3D, oriented
    q += [R(m, labels=("sig", "0.05", 0), tab="clin") for m in ABL]                                        # ablation, planted signal
    q += [R("s25d", subset="ctl")] + [R("s25d", subset=f"rnd{r}") for r in range(5)]                       # treatment: matched size
    q += [R(m, folds="regimen") for m in ("s25d", "carfm_25d")]                                            # regimen held out
    q += [R("carfm_k", seed=s) for s in (1, 2)]                                                            # 3D repeats
    return q


QUEUES = {"q0": _queue_q0, "q1": _queue_q1,
          "smoke": lambda: [R("s25d", "orig"), R("carfm_25d", labels=("sig", "0.05", 0), tab="clin"), R("abl25d_m0b1g0", labels=("null", 0), tab="clin"),
                            R("carfm_k", seed=1), R("s25d", subset="ctl"), R("carfm_25d", folds="regimen"), R("image_only_k", "lps_fov")]}


def run_id(r):
    lab = r["labels"] if isinstance(r["labels"], str) else "-".join(map(str, r["labels"]))
    return "|".join([r["model"], r["variant"], f"s{r['seed']}", lab, r["tab"], r["subset"] or "all", r["folds"] or "cv"])


class Data3(Data2):
    """Data2 that can skip the 3D volumes (2.5D-only processes)."""

    def __init__(self, root, need_vol=True):
        self.table = pd.read_csv(os.path.join(root, "table.csv"))
        from .prep2 import KIN_NAMES
        k = pd.read_csv(os.path.join(root, "kinetics.csv"), index_col=0)
        self.kin = k.loc[self.table.case_id, KIN_NAMES].to_numpy(np.float32)
        self._vol, self._mask, self._sl = [], [], []
        for c in self.table.case_id:
            z = np.load(os.path.join(root, f"{c}.npz"))
            if need_vol:
                self._vol.append(z["vol"]); self._mask.append(z["mask"])
            self._sl.append(z["sl"])


class Fast5(Fast):
    img_map = None

    def _batch(self, kind, idx, tab, miss, z, augment):
        b = super()._batch(kind, idx, tab, miss, z, augment)
        if self.img_map is not None:
            b["image"] = self.imgs(kind).batch(self.img_map[idx], augment)
        return b


class Rev3:
    def __init__(self, root, out, tag, log, synthetic=False, need_vol=True):
        self.root, self.out, self.tag, self.log, self.synthetic, self.need_vol = root, out, tag, log, synthetic, need_vol
        self.fast = {}
        os.makedirs(out, exist_ok=True)
        self.summary = os.path.join(out, f"runs_{tag}.csv")

    def F(self, variant):
        if variant not in self.fast:
            if len(self.fast) and not self.synthetic:  # one variant on the GPU at a time
                for f in self.fast.values():
                    f.cache.clear()
                torch.cuda.empty_cache()
            if self.synthetic:
                from .synthetic import make_cohort2
                d = make_cohort2(n=120)
                ag = sum(REGIMEN_GROUPS.values(), ())
                d.table["nac_agent"] = np.random.default_rng(0).choice(ag, len(d.table), p=[0.3, 0.3] + [0.4 / (len(ag) - 2)] * (len(ag) - 2))
            else:
                t0 = time.time()
                d = Data3(os.path.join(self.root, variant), self.need_vol)
                self.log(f"loaded {variant}: {len(d.table)} cases in {time.time() - t0:.0f} s")
            self.fast[variant] = Fast5(d, self.out, self.log, self.synthetic)
        return self.fast[variant]

    # ------------------------------------------------------------------ labels
    def labels(self, f, spec):
        y = f.y.copy()
        dev, ext = f.dev, f.ext
        if spec == "real":
            return y, None
        anc = ClinicalAnchor(n_splits=5)
        anc.fit(f.Xc[dev], y[dev].astype(int))
        z = anc.logit(f.Xc)
        if spec[0] == "sig":
            from .prep2 import KIN_NAMES
            pe = f.Xk[:, KIN_NAMES.index("pe_mean")].astype(float)
            Xi = np.nan_to_num(f.Xc, nan=0.0) + np.isnan(f.Xc) * np.nanmean(f.Xc[dev], 0)
            ok = np.isfinite(pe)
            pe = np.where(ok, pe, np.nanmean(pe[dev]))
            r = pe - LinearRegression().fit(Xi[dev], pe[dev]).predict(Xi)
            s = (r - r[dev].mean()) / r[dev].std()
            beta = BETAS[spec[1]]
            u = np.random.default_rng(1000 + int(spec[2])).random(len(y))
            ys = (u < metrics.sigmoid(z + beta * s)).astype(np.float32)
            keep = np.r_[dev, ext]
            y[keep] = ys[keep]
            info = {"beta": beta, "oracle_dev": roc_auc_score(y[dev], z[dev] + beta * s[dev]) - roc_auc_score(y[dev], z[dev]),
                    "oracle_ext": roc_auc_score(y[ext], z[ext] + beta * s[ext]) - roc_auc_score(y[ext], z[ext]),
                    "prev_dev": float(y[dev].mean()), "prev_ext": float(y[ext].mean())}
            return y, info
        # null: permute images within collection x clinical-risk quintile
        rng = np.random.default_rng(2000 + int(spec[1]))
        m = np.arange(len(y))
        tb = f.tb
        for c in tb.collection.unique():
            idx = np.flatnonzero((tb.collection == c).to_numpy())
            q = pd.qcut(pd.Series(z[idx]).rank(method="first"), 5, labels=False).to_numpy()
            for k in range(5):
                g = idx[q == k]
                m[g] = rng.permutation(g)
        return y, {"perm_fraction_moved": float((m != np.arange(len(y))).mean()), "img_map": m}

    # ------------------------------------------------------------------ one run
    def run(self, r):
        rid = run_id(r)
        os.makedirs(os.path.join(self.out, "preds"), exist_ok=True)
        fn = os.path.join(self.out, "preds", f"pred_{self.tag}_{len(self._done())}.csv")
        f = self.F(r["variant"])
        name, seed = r["model"], r["seed"]
        anchored = name.startswith(("carf", "abl25d"))
        slice_model = name in SLICE_MODELS or name.startswith("abl25d")
        kind = "sl" if slice_model else "vol"
        y, info = self.labels(f, r["labels"])
        f.img_map = info.pop("img_map") if info and "img_map" in info else None
        tb, ext = f.tb, f.ext
        devset = f.dev
        if r["subset"]:
            arm = tb.nac_agent.iloc[f.dev].isin(CONTROL_ARM).to_numpy()
            ctl = f.dev[arm]
            devset = ctl if r["subset"] == "ctl" else np.sort(np.random.default_rng(int(r["subset"][3:])).choice(f.dev, len(ctl), replace=False))
        if r["folds"] == "regimen":
            grp = tb.nac_agent.iloc[devset].map({a: g for g, aa in REGIMEN_GROUPS.items() for a in aa}).to_numpy()
            splits = [(np.flatnonzero(grp != g), np.flatnonzero(grp == g)) for g in REGIMEN_GROUPS]
        else:
            splits = list(StratifiedKFold(2 if self.synthetic else 5, shuffle=True, random_state=seed).split(devset, y[devset]))
        Xt = f.Xc if r["tab"] == "clin" else np.hstack([f.Xc, f.Xk])
        if name == "image_only_k":
            cfg = dict(CONFIGS["base"])
        else:
            cfg = dict(ANCHORED)
        if self.synthetic:
            cfg.update(epochs=2, widths=(8, 16))
        full = ClinicalAnchor(n_splits=5)
        full.fit(f.Xc[devset], y[devset].astype(int))
        z_ext = full.logit(f.Xc[ext]).astype(np.float32)
        nd, ne, K = len(devset), len(ext), len(splits)
        P = {k: np.full(nd + ne, np.nan) for k in ("p", "zc", "corr", "gate", "fold")}
        PF = np.full((ne, K), np.nan)
        GF, CF = np.full((ne, K), np.nan), np.full((ne, K), np.nan)
        P["zc"][nd:] = metrics.sigmoid(z_ext)
        t0, rows = time.time(), []
        for k, (tr, te) in enumerate(splits):
            if _late():
                self.log(f"[{rid}] SKIPPED from fold {k}: time budget")
                return None
            s_k = k if seed == 0 else 1000 * seed + k
            tr2, va = f.inner_split(devset[tr], s_k)
            z = np.zeros(len(y), np.float32)
            anc = ClinicalAnchor(n_splits=5)
            z[tr2] = anc.fit(f.Xc[tr2], y[tr2].astype(int))
            z[va] = anc.logit(f.Xc[va])
            z[devset[te]] = anc.logit(f.Xc[devset[te]])
            P["zc"][te] = metrics.sigmoid(z[devset[te]])
            z[ext] = z_ext
            zin = z if anchored else np.zeros_like(z)
            tab, miss = _standardise(Xt, Xt[tr2])
            f_y = f.y
            f.y = y  # labels used by fit/predict batches
            try:
                model = build(name, tab.shape[1], cfg["widths"], 128)
                t1 = time.time()
                hist, kept = f.fit(model, kind, tr2, va, tab, miss, zin, cfg, seed=s_k,
                                   lr=cfg.get("lr_pretrained") if slice_model else cfg["lr"])
                o = f.predict(model, kind, devset[te], tab, miss, zin)
                e = f.predict(model, kind, ext, tab, miss, zin)
                tr_auc = roc_auc_score(y[tr2], f.predict(model, kind, tr2, tab, miss, zin)["logit"])
            finally:
                f.y = f_y
            P["p"][te] = metrics.sigmoid(o["logit"])
            P["fold"][te] = k
            PF[:, k] = metrics.sigmoid(e["logit"])
            if "gate" in o:
                P["corr"][te] = o["gate"] * o["delta"]
                P["gate"][te] = o["gate"]
                CF[:, k], GF[:, k] = e["gate"] * e["delta"], e["gate"]
            rows.append({"fold": k, "epochs": len(hist), "kept": kept, "train_auc": tr_auc, "minutes": (time.time() - t1) / 60})
            del model
            torch.cuda.empty_cache()
        P["p"][nd:] = PF.mean(1)
        P["corr"][nd:], P["gate"][nd:] = CF.mean(1), GF.mean(1)
        P["fold"][nd:] = -1
        idx = np.r_[devset, ext]
        D = pd.DataFrame({"case_id": tb.case_id.iloc[idx].to_numpy(), "collection": tb.collection.iloc[idx].to_numpy(), "y": y[idx].astype(int),
                          "set": ["dev"] * nd + ["ext"] * ne, **{k: v for k, v in P.items()}})
        for k in range(K):
            D[f"pf{k}"] = np.r_[np.full(nd, np.nan), PF[:, k]]
        if anchored:
            for k in range(K):
                D[f"cf{k}"] = np.r_[np.full(nd, np.nan), CF[:, k]]
        D.to_csv(fn, index=False, float_format="%.4g")
        dv, ex = D[D.set == "dev"], D[D.set == "ext"]
        S = {"run_id": rid, "file": os.path.basename(fn), **{k: (json.dumps(v) if isinstance(v, tuple) else v) for k, v in r.items()},
             "n_dev": nd, "oof_auc": roc_auc_score(dv.y, dv.p), "oof_clin": roc_auc_score(dv.y, dv.zc),
             "ext_auc": roc_auc_score(ex.y, ex.p), "ext_clin": roc_auc_score(ex.y, ex.zc),
             "ext_auc_fold_mean": float(np.mean([roc_auc_score(ex.y, ex[f"pf{k}"]) for k in range(K)])),
             "train_auc_mean": float(np.mean([x["train_auc"] for x in rows])), "epochs": "/".join(str(x["epochs"]) for x in rows),
             "minutes": (time.time() - t0) / 60, "corr_abs_ext": float(np.nanmean(np.abs(ex["corr"]))) if anchored else np.nan,
             "gate_ext": float(np.nanmean(ex["gate"])) if anchored else np.nan, **(info or {})}
        old = pd.read_csv(self.summary) if os.path.exists(self.summary) else pd.DataFrame()
        pd.concat([old, pd.DataFrame([S])], ignore_index=True).to_csv(self.summary, index=False)
        self.log(f"[{rid}] OOF {S['oof_auc']:.3f} (clin {S['oof_clin']:.3f}); ext {S['ext_auc']:.3f} (clin {S['ext_clin']:.3f}; "
                 f"fold models {S['ext_auc_fold_mean']:.3f}); train {S['train_auc_mean']:.3f}; {S['minutes']:.1f} min")
        return S

    def _done(self):
        return pd.read_csv(self.summary).run_id.tolist() if os.path.exists(self.summary) else []

    def queue(self, q):
        for r in QUEUES[q]():
            if run_id(r) in self._done():
                continue
            if _late():
                self.log(f"time budget reached before {run_id(r)}")
                break
            try:
                self.run(r)
            except Exception as e:  # noqa: BLE001
                import traceback
                self.log(f"[{run_id(r)}] FAILED {e!r}\n{traceback.format_exc()[-1500:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--queue", required=True, choices=list(QUEUES))
    ap.add_argument("--tag", default="a")
    a = ap.parse_args()
    log = lambda *m: print(time.strftime("%H:%M:%S"), f"<{a.tag}>", *m, flush=True)  # noqa: E731
    need_vol = any(not (r["model"] in SLICE_MODELS or r["model"].startswith("abl25d")) for r in QUEUES[a.queue]())
    log(f"queue {a.queue}: {len(QUEUES[a.queue]())} runs; device {_dev()}; 3D volumes {'loaded' if need_vol else 'not needed'}")
    t0 = time.time()
    Rev3(a.root, a.out, a.tag, log, a.synthetic, need_vol).queue(a.queue)
    log(f"total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
