"""Explainability with quantitative checks: Grad-CAM, linear SHAP, counterfactuals, ablation."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def gradcam(model, batch, device="cpu"):
    """3D Grad-CAM on the encoder output. Returns (B, D, H, W) maps in [0, 1].

    For CARF the target is the imaging contribution gate*delta, not the full logit,
    so the map shows what the image added on top of the clinical anchor.
    """
    model.eval().to(device)
    batch = {k: v.to(device) for k, v in batch.items()}
    store = {}

    def hook(_m, _i, o):
        o.retain_grad()
        store["a"] = o  # returns None, so the encoder output is left unchanged

    h = model.encoder.register_forward_hook(hook)
    try:
        out = model(batch, drop_image=False) if hasattr(model, "gate_head") else model(batch)
        target = (out["gate"] * out["delta"]) if "gate" in out else out["logit"]
        # explain evidence in the direction the image pushed each case (the residual is signed)
        target = target * torch.sign(target.detach())
        model.zero_grad(set_to_none=True)
        target.sum().backward()
    finally:
        h.remove()
    a, g = store["a"], store["a"].grad
    cam = F.relu((a * g.mean(dim=(2, 3, 4), keepdim=True)).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=batch["image"].shape[2:], mode="trilinear", align_corners=False)[:, 0]
    mx = cam.flatten(1).max(1)[0].clamp_min(1e-8)
    return (cam / mx[:, None, None, None]).detach().cpu().numpy()


def mask_enrichment(cam: np.ndarray, mask: np.ndarray) -> float:
    """Share of attribution inside the tumour mask divided by the mask's share of the volume.

    1.0 means the map is no more focused on the tumour than chance.
    """
    m = mask.astype(bool)
    tot = cam.sum()
    if tot <= 0 or m.sum() == 0:
        return float("nan")
    return float((cam[m].sum() / tot) / m.mean())


def deletion_auc(model, batch, cam, steps=8, device="cpu"):
    """Faithfulness: mean drop in the model's logit as the most-attributed voxels are zeroed."""
    model.eval().to(device)
    b = {k: v.clone().to(device) for k, v in batch.items()}
    fw = {"drop_image": False} if hasattr(model, "gate_head") else {}
    with torch.no_grad():
        base = model(b, **fw)["logit"].cpu().numpy()
        order = np.argsort(-cam.reshape(cam.shape[0], -1), axis=1)
        drops = []
        for s in range(1, steps + 1):
            k = int(order.shape[1] * s / (steps * 4))  # delete up to 25% of voxels
            img = b["image"].clone()
            flat = img.view(img.shape[0], img.shape[1], -1)
            for i in range(img.shape[0]):
                flat[i][:, torch.as_tensor(order[i, :k].copy(), device=device)] = 0
            drops.append(base - model({**b, "image": img}, **fw)["logit"].cpu().numpy())
    return np.mean(np.abs(drops), axis=0)


def linear_shap(pipeline, X: np.ndarray, names: list[str]):
    """Exact SHAP values for a logistic pipeline (imputer -> scaler -> LR), on the logit scale.

    For a linear model with independent features, phi_j = coef_j * (x_j - mean_j).
    Missing-indicator columns added by the imputer are reported under their own names.
    """
    Xt = pipeline[:-1].transform(X)
    coef = pipeline[-1].coef_[0]
    phi = coef * (Xt - Xt.mean(0))
    extra = [f"missing_{i}" for i in range(Xt.shape[1] - len(names))]
    return phi, names + extra


def counterfactual_clinical(pipeline, x: np.ndarray, names, mutable, grid, thr_logit=0.0):
    """Smallest single-variable change (from `grid[name]`) that flips the clinical prediction."""
    def lg(v):
        p = np.clip(pipeline.predict_proba(v.reshape(1, -1))[0, 1], 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    base = lg(x)
    best = None
    for name in mutable:
        j = names.index(name)
        for val in grid[name]:
            x2 = x.copy()
            x2[j] = val
            if (lg(x2) - thr_logit) * (base - thr_logit) < 0:
                cost = abs(val - x[j]) if not np.isnan(x[j]) else np.inf
                if best is None or cost < best[2]:
                    best = (name, float(val), float(cost))
    return best
