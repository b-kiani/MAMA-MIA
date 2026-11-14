"""End-to-end study pipeline.

    python -m mmonc.pipeline --synthetic --out results/synthetic_smoke     # code test only
    python -m mmonc.pipeline --data /kaggle/input/mamamia-crops --out results/run1

Development = one collection (ISPY2); every other collection is external and is only
scored after models, thresholds and the clinical anchor are fixed on development data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import SYNTHETIC_WATERMARK, __version__, explain, figures, metrics, stats
from .data import InMemory, Prepared
from .models import ClinicalAnchor, build, make_logreg
from .train import Cases, fit, predict, seed_all

CLIN = ["age", "hr", "her2", "menopausal", "multifocal"]  # synthetic default; real runs set cfg["clinical"]
DEFAULTS = dict(dev="ISPY2", folds=5, seeds=[0], epochs=30, batch_size=8, lr=3e-4, widths=[32, 64, 128, 256],
                dim=128, deep_models=["image_only", "concat", "carf"], n_boot=2000, margin=0.03, patience=8,
                score_external=True)


def _standardise(X, ref):
    mu, sd = np.nanmean(ref, 0), np.nanstd(ref, 0) + 1e-6
    Z = (X - mu) / sd
    miss = np.isnan(Z).astype(np.float32)
    return np.nan_to_num(Z, nan=0.0).astype(np.float32), miss


def run(data, out: str, cfg: dict | None = None, synthetic: bool = False, log=print):
    cfg = {**DEFAULTS, **(cfg or {})}
    os.makedirs(out, exist_ok=True)
    fig_dir, ck_dir = os.path.join(out, "figures"), os.path.join(out, "checkpoints")
    os.makedirs(ck_dir, exist_ok=True)
    t0 = time.time()
    tb = data.table
    keep = tb.pcr.notna().to_numpy()
    y_all = tb.pcr.fillna(-1).to_numpy().astype(int)
    dev = np.flatnonzero((tb.collection == cfg["dev"]).to_numpy() & keep)
    ext = np.flatnonzero((tb.collection != cfg["dev"]).to_numpy() & keep)
    if len(dev) == 0 or len(ext) == 0:
        raise ValueError("need a development collection and at least one external collection")
    y_dev, y_ext = y_all[dev], y_all[ext]
    clin_cols = [c for c in cfg.get("clinical", CLIN) if c in tb.columns]
    Xc = tb[clin_cols].to_numpy(float)
    Xr = np.asarray(data.radiomics, float)
    dt = tb[["dt0", "dt1", "dt2"]].to_numpy(float) if "dt1" in tb else np.zeros((len(tb), 3))
    dt = np.nan_to_num(dt, nan=0.0)
    cv = StratifiedKFold(cfg["folds"], shuffle=True, random_state=0)
    oof, extp = {}, {}  # model name -> probabilities

    # ---- 1. tabular models (cross-fitted on development, refit on all development)
    anchor = ClinicalAnchor(n_splits=cfg["folds"])
    z_dev = anchor.fit(Xc[dev], y_dev)
    z_ext = anchor.logit(Xc[ext])
    oof["Clinical"], extp["Clinical"] = metrics.sigmoid(z_dev), metrics.sigmoid(z_ext)
    for name, X, pen in [("Radiomics", Xr, "l1"), ("Clinical + radiomics", np.hstack([Xc, Xr]), "l1")]:
        m = make_logreg(C=0.3, penalty=pen)
        oof[name] = cross_val_predict(m, X[dev], y_dev, cv=cv, method="predict_proba")[:, 1]
        extp[name] = m.fit(X[dev], y_dev).predict_proba(X[ext])[:, 1]
    SE = cfg["score_external"]  # False during development: external cases are not even predicted

    # ---- 2. deep models: out-of-fold on development, fold ensemble on external
    tab, miss = _standardise(np.hstack([Xc, Xr]), np.hstack([Xc, Xr])[dev])
    z_all = np.zeros(len(tb), np.float32)
    z_all[dev], z_all[ext] = z_dev, z_ext

    def ds(idx, augment=False):
        return Cases(None, tab[idx], miss[idx], dt[idx], z_all[idx], y_all[idx].clip(0), augment,
                     loader=lambda j, idx=idx: data.image(int(idx[j])))

    extra, fold_models = {}, {}
    for name in cfg["deep_models"]:
        o = np.zeros(len(dev)); e = np.zeros(len(ext)); g_dev = np.zeros(len(dev)); g_ext = np.zeros(len(ext))
        n_models = 0
        for seed in cfg["seeds"]:
            for k, (tr, te) in enumerate(cv.split(dev, y_dev)):
                # early stopping uses an inner split, so out-of-fold predictions stay clean
                rng = np.random.default_rng(seed * 100 + k)
                perm = rng.permutation(tr)
                n_val = max(2, int(0.15 * len(tr)))
                va, tr2 = perm[:n_val], perm[n_val:]
                seed_all(seed * 100 + k)
                model = build(name, tab.shape[1], tuple(cfg["widths"]), cfg["dim"])
                log(f"[{name}] seed {seed} fold {k}")
                fit(model, ds(dev[tr2], True), ds(dev[va]), epochs=cfg["epochs"], lr=cfg["lr"],
                    batch_size=cfg["batch_size"], patience=cfg["patience"], seed=seed * 100 + k,
                    ckpt_path=os.path.join(ck_dir, f"{name}_s{seed}_f{k}.pt"), log=log)
                pd_ = predict(model, ds(dev[te]), batch_size=cfg["batch_size"], device=_dev())
                o[te] += metrics.sigmoid(pd_["logit"]) / len(cfg["seeds"])
                if "gate" in pd_:
                    g_dev[te] += pd_["gate"] / len(cfg["seeds"])
                if SE:
                    pe_ = predict(model, ds(ext), batch_size=cfg["batch_size"], device=_dev())
                    e += metrics.sigmoid(pe_["logit"])
                    if "gate" in pe_:
                        g_ext += pe_["gate"]
                n_models += 1
                fold_models[name] = model.cpu()
        label = {"image_only": "Image only (ResNet)", "concat": "Concatenation fusion", "carf": "CARF (proposed)",
                 "densenet121": "Image only (DenseNet-121)", "efficientnet_b0": "Image only (EfficientNet-B0)",
                 "vit": "Image only (ViT)"}[name]
        oof[label], extp[label] = o, e / n_models
        if name == "carf":
            extra["gate_dev"], extra["gate_ext"] = g_dev, g_ext / n_models

    # ---- 3. freeze thresholds on development, then score everything
    thr = {m: metrics.youden_threshold(y_dev, p) for m, p in oof.items()}
    frozen = {"version": __version__, "thresholds": thr, "config": cfg,
              "split_hash": hashlib.sha256(",".join(tb.case_id.iloc[dev].astype(str)).encode()).hexdigest()}
    json.dump(frozen, open(os.path.join(out, "frozen.json"), "w"), indent=1, default=float)

    rows_int = [{"model": m, **metrics.summary(y_dev, p, thr[m], cfg["n_boot"])} for m, p in oof.items()]
    if not SE:
        pd.DataFrame(rows_int).to_csv(os.path.join(out, "table2_internal.csv"), index=False)
        pd.DataFrame({"case_id": tb.case_id.iloc[dev], "y": y_dev, **oof}).to_csv(
            os.path.join(out, "predictions_internal_oof.csv"), index=False)
        log("development only: external collections were not scored")
        return {"synthetic": synthetic, "n_dev": int(len(dev)), "external_scored": False}
    rows_ext = []
    coll_ext = tb.collection.to_numpy()[ext]
    for m, p in extp.items():
        rows_ext.append({"model": m, "cohort": "Pooled external", **metrics.summary(y_ext, p, thr[m], cfg["n_boot"])})
        for c in sorted(set(coll_ext)):
            s = coll_ext == c
            if min(y_ext[s].sum(), (1 - y_ext[s]).sum()) >= 5:
                rows_ext.append({"model": m, "cohort": c, **metrics.summary(y_ext[s], p[s], thr[m], cfg["n_boot"])})
    t2, t3 = pd.DataFrame(rows_int), pd.DataFrame(rows_ext)

    # ---- 4. ablation: every model against the clinical model (DeLong, Holm)
    abl = []
    for m, p in extp.items():
        if m == "Clinical":
            continue
        d = stats.delong(y_ext, extp["Clinical"], p)
        abl.append({"model": m, "auc": d["auc2"], "delta_auc_vs_clinical": d["diff"], "ci_lo": d["ci_lo"],
                    "ci_hi": d["ci_hi"], "p": d["p"], **stats.nri_idi(y_ext, extp["Clinical"], p, thr["Clinical"])})
    t5 = pd.DataFrame(abl)
    t5["p_holm"] = stats.holm(t5.p)
    primary_name = "CARF (proposed)" if "CARF (proposed)" in extp else "Clinical + radiomics"
    prim = stats.delong(y_ext, extp["Clinical"], extp[primary_name])
    primary = {"comparison": f"{primary_name} minus Clinical, pooled external AUC", **prim,
               "verdict": stats.verdict(prim, cfg["margin"])}

    # ---- 5. Table 1, subgroup table
    t1 = _table1(tb[keep], clin_cols)
    sub = _subgroups(tb.iloc[ext], y_ext, extp[primary_name], ["vendor", "field_strength", "ethnicity"])

    # ---- 6. explainability
    t6 = _explain(data, fold_models, ds, ext, y_ext, extp, extra, anchor, Xc, clin_cols, fig_dir, synthetic, cfg)

    # ---- 7. figures
    figures.workflow(os.path.join(fig_dir, "fig1_workflow"), synthetic)
    figures.roc_pr(y_ext, extp, os.path.join(fig_dir, "fig2_roc_pr_external"), synthetic, "(pooled external)")
    figures.calibration_plot(y_ext, extp, os.path.join(fig_dir, "fig3_calibration_external"), synthetic)
    figures.decision_curve(y_ext, extp, os.path.join(fig_dir, "fig4_decision_curve_external"), synthetic)
    figures.confusion({m: r for m, r in ((r["model"], r) for r in rows_ext if r["cohort"] == "Pooled external")},
                      os.path.join(fig_dir, "fig5_confusion_external"), synthetic)
    surv = None
    if {"rfs_months", "rfs_event"} <= set(tb.columns):
        s = tb.iloc[ext][["rfs_months", "rfs_event"]].notna().all(1).to_numpy()
        if s.sum() >= 30 and tb.iloc[ext].rfs_event[s].sum() >= 10:
            tm, ev = tb.iloc[ext].rfs_months.to_numpy()[s], tb.iloc[ext].rfs_event.to_numpy()[s].astype(int)
            risk = 1 - extp[primary_name][s]  # non-response as the risk score (exploratory)
            surv = {"n": int(s.sum()), "events": int(ev.sum()), "c_index": metrics.c_index(tm, ev, risk)}
            figures.kaplan_meier(tm, ev, risk, os.path.join(fig_dir, "fig6_km_external"), synthetic)

    # ---- 8. write
    for name, t in [("table1_cohort", t1), ("table2_internal", t2), ("table3_external", t3),
                    ("table5_ablation", t5), ("table6_explainability", t6), ("tableS_subgroups", sub)]:
        t.to_csv(os.path.join(out, name + ".csv"), index=False)
    pd.DataFrame({"case_id": tb.case_id.iloc[ext], "collection": coll_ext, "y": y_ext, **extp}).to_csv(
        os.path.join(out, "predictions_external.csv"), index=False)
    pd.DataFrame({"case_id": tb.case_id.iloc[dev], "y": y_dev, **oof}).to_csv(
        os.path.join(out, "predictions_internal_oof.csv"), index=False)
    summary = {"synthetic": synthetic, "note": SYNTHETIC_WATERMARK if synthetic else "",
               "n_dev": int(len(dev)), "n_ext": int(len(ext)), "events_ext": int(y_ext.sum()),
               "primary": primary, "survival_exploratory": surv, "minutes": round((time.time() - t0) / 60, 2),
               "torch": torch.__version__, "device": _dev()}
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1, default=float)
    log(json.dumps(summary["primary"], indent=1, default=float))
    return summary


def _dev():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _table1(tb, clin_cols):
    rows = []
    for c, g in list(tb.groupby("collection")) + [("All", tb)]:
        r = {"collection": c, "n": len(g), "pcr_n": int(g.pcr.sum()), "pcr_pct": round(100 * g.pcr.mean(), 1)}
        for v in clin_cols:
            x = g[v]
            r[v] = f"{x.mean():.1f} ({x.std():.1f})" if v == "age" else f"{100 * x.mean():.1f}%"
            r[v + "_missing_pct"] = round(100 * x.isna().mean(), 1)
        rows.append(r)
    return pd.DataFrame(rows)


def _subgroups(tb, y, p, cols):
    rows = []
    for c in cols:
        if c not in tb.columns:
            continue
        for lvl in pd.unique(tb[c].dropna()):
            s = (tb[c] == lvl).to_numpy()
            ok = min(y[s].sum(), (1 - y[s]).sum()) >= 5
            a = metrics.bootstrap_ci(roc_auc_score, y[s], p[s], n_boot=500) if ok else (np.nan,) * 3
            rows.append({"variable": c, "level": lvl, "n": int(s.sum()), "events": int(y[s].sum()),
                         "auc": a[0], "auc_lo": a[1], "auc_hi": a[2]})
    return pd.DataFrame(rows)


def _explain(data, fold_models, ds, ext, y_ext, extp, extra, anchor, Xc, clin_cols, fig_dir, synthetic, cfg):
    rows = []
    phi, names = explain.linear_shap(anchor.model, Xc[ext], clin_cols)
    for j in np.argsort(-np.abs(phi).mean(0))[:5]:
        rows.append({"analysis": "Clinical anchor SHAP (mean |logit contribution|)", "item": names[j],
                     "value": float(np.abs(phi[:, j]).mean())})
    cands = [k for k in ("carf", "concat", "image_only") if k in fold_models]
    key = cands[0] if cands else None
    if key is None:
        return pd.DataFrame(rows)
    model = fold_models[key]
    n = min(len(ext), 24)
    sub = ds(ext[:n])
    batch = torch.utils.data.default_collate([sub[i] for i in range(n)])
    cam = explain.gradcam(model, batch)
    enr = [explain.mask_enrichment(cam[i], data.mask(int(ext[i]))) for i in range(n)]
    rows.append({"analysis": f"Grad-CAM tumour enrichment ({key}; 1.0 = chance)", "item": f"median, n={n}",
                 "value": float(np.nanmedian(enr)) if np.isfinite(enr).any() else float("nan")})
    rows.append({"analysis": "Deletion test: mean |logit change|, top 25% voxels removed", "item": f"n={n}",
                 "value": float(explain.deletion_auc(model, batch, cam).mean())})
    if "gate_ext" in extra:
        g = extra["gate_ext"]
        rows.append({"analysis": "CARF gate, external", "item": "mean", "value": float(g.mean())})
        rows.append({"analysis": "CARF gate, development", "item": "mean", "value": float(extra["gate_dev"].mean())})
        no_img = predict(model, ds(ext), batch_size=cfg["batch_size"], drop_image=True)["logit"]
        rows.append({"analysis": "CARF with image removed equals clinical anchor", "item": "max |logit difference|",
                     "value": float(np.abs(no_img - np.log(extp["Clinical"] / (1 - extp["Clinical"]))).max())})
    i = int(np.nanargmax(enr)) if np.isfinite(enr).any() else 0
    figures.gradcam_panel(np.asarray(data.image(int(ext[i])), np.float32), data.mask(int(ext[i])), cam[i],
                          os.path.join(fig_dir, "fig7_gradcam_example"), synthetic)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config")
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--score-external", action="store_true", help="score external collections (run once, after freezing)")
    a = ap.parse_args()
    cfg = {}
    if a.config:
        import yaml

        cfg = yaml.safe_load(open(a.config))
    if a.synthetic:
        from .radiomics import extract_simple
        from .synthetic import make_cohort

        tb, im, mk = make_cohort(a.n, size=16, seed=0)
        data = InMemory(tb, im, mk, extract_simple(im, mk))
        cfg = {"epochs": 3, "folds": 3, "widths": [8, 16, 32], "dim": 32, "n_boot": 200, "patience": 3, **cfg}
    elif a.data:
        data = Prepared(a.data)
        print(f"loaded {len(data.table)} cases, {data.radiomics.shape[1]} radiomic features", flush=True)
    else:
        ap.error("pass --synthetic or --data")
    if a.score_external:
        cfg["score_external"] = True
    t0 = time.time()
    run(data, a.out, cfg, synthetic=a.synthetic, log=lambda *m: print(time.strftime("%H:%M:%S"), *m, flush=True))
    print(f"total {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
