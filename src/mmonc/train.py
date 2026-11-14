"""Training loop with per-epoch checkpoints and automatic resume (for 12-hour sessions)."""
from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Cases(Dataset):
    """In-memory dataset. For real data pass `loader`, a function idx -> image array."""

    def __init__(self, images, tab, tab_missing, dt, z_clin, y, augment=False, loader=None):
        self.images, self.loader = images, loader
        self.tab = torch.as_tensor(tab, dtype=torch.float32)
        self.miss = torch.as_tensor(tab_missing, dtype=torch.float32)
        self.dt = torch.as_tensor(dt, dtype=torch.float32)
        self.z = torch.as_tensor(z_clin, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img = self.loader(i) if self.loader is not None else self.images[i]
        img = torch.as_tensor(np.ascontiguousarray(img), dtype=torch.float32)
        if self.augment:
            for ax in (1, 2, 3):
                if random.random() < 0.5:
                    img = img.flip(ax)
            img = img * (1 + 0.1 * torch.randn(1)) + 0.1 * torch.randn(1)
        return {"image": img, "tab": self.tab[i], "tab_missing": self.miss[i], "dt": self.dt[i],
                "z_clin": self.z[i], "y": self.y[i]}


def _to(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def predict(model, ds, device="cpu", batch_size=8, **fw):
    model.eval().to(device)
    out = {}
    for b in DataLoader(ds, batch_size=batch_size):
        r = model(_to(b, device), **fw)
        for k, v in r.items():
            if v.dim() == 1:
                out.setdefault(k, []).append(v.float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def fit(model, train_ds, val_ds, *, epochs=30, lr=3e-4, weight_decay=1e-2, batch_size=8,
        gate_penalty=0.01, aux_weight=0.5, patience=8, device=None, ckpt_path=None, seed=0, log=print):
    """Train with BCE. Resumes from `ckpt_path` if it exists. Returns the history list."""
    from sklearn.metrics import roc_auc_score

    seed_all(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    start, best, bad, hist, best_state = 0, -1.0, 0, [], None
    aux_weight = getattr(model, "aux_weight", aux_weight)

    if ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start, best, bad, hist, best_state = ck["epoch"] + 1, ck["best"], ck["bad"], ck["hist"], ck["best_state"]
        log(f"resumed from epoch {start}")

    dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=len(train_ds) > batch_size,
                    num_workers=0, pin_memory=use_amp)
    for ep in range(start, epochs):
        if bad >= patience:
            break
        model.train()
        tot = 0.0
        for b in dl:
            b = _to(b, device)
            with torch.autocast(device_type=device, enabled=use_amp):
                r = model(b)
                loss = F.binary_cross_entropy_with_logits(r["logit"].float(), b["y"])
                if "gate" in r:
                    # Auxiliary loss trains the residual with the gate fully open. Without it the
                    # gate closes early (while the residual is still noise) and never reopens.
                    aux = F.binary_cross_entropy_with_logits((b["z_clin"] + r["delta"]).float(), b["y"])
                    loss = loss + aux_weight * aux + gate_penalty * r["gate"].float().mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tot += loss.item() * len(b["y"])
        sched.step()
        p = predict(model, val_ds, device, batch_size)["logit"]
        yv = val_ds.y.numpy()
        auc = roc_auc_score(yv, p) if len(np.unique(yv)) > 1 else float("nan")
        hist.append({"epoch": ep, "loss": tot / max(1, len(dl.dataset)), "val_auc": float(auc)})
        log(f"epoch {ep:3d}  loss {hist[-1]['loss']:.4f}  val_auc {auc:.3f}")
        if auc > best:
            best, bad = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ckpt_path:
            tmp = ckpt_path + ".tmp"
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "epoch": ep, "best": best, "bad": bad, "hist": hist, "best_state": best_state}, tmp)
            os.replace(tmp, ckpt_path)  # atomic: a killed session never leaves a half-written file
    if best_state is not None:
        model.load_state_dict(best_state)
    return hist
