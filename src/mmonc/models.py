"""Models: clinical anchor, 3D image encoder, fusion baselines and CARF."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------------- tabular
def make_logreg(C: float = 1.0, penalty: str = "l2"):
    import sklearn

    if penalty == "l1":
        new_api = tuple(int(v) for v in sklearn.__version__.split(".")[:2]) >= (1, 8)
        kw = {"l1_ratio": 1.0} if new_api else {"penalty": "l1"}  # `penalty` is deprecated from 1.8
        lr = LogisticRegression(C=C, solver="liblinear", max_iter=2000, random_state=0, **kw)
    else:
        lr = LogisticRegression(C=C, max_iter=2000)
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), lr)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class ClinicalAnchor:
    """Penalised logistic model on clinical variables.

    `fit` returns cross-fitted (out-of-fold) logits for the development cases, so the
    fusion model never trains on an anchor that has already seen that patient's label.
    """

    def __init__(self, C: float = 1.0, n_splits: int = 5, seed: int = 0):
        self.model = make_logreg(C)
        self.n_splits, self.seed = n_splits, seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        cv = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.seed)
        oof = cross_val_predict(make_logreg(self.model[-1].C), X, y, cv=cv, method="predict_proba")[:, 1]
        self.model.fit(X, y)
        return _logit(oof)

    def logit(self, X: np.ndarray) -> np.ndarray:
        return _logit(self.model.predict_proba(X)[:, 1])


# --------------------------------------------------------------------------- image
class _Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv3d(cin, cout, 3, stride, 1, bias=False)
        self.n1 = nn.GroupNorm(min(8, cout), cout)
        self.c2 = nn.Conv3d(cout, cout, 3, 1, 1, bias=False)
        self.n2 = nn.GroupNorm(min(8, cout), cout)
        self.skip = None
        if stride != 1 or cin != cout:
            self.skip = nn.Sequential(nn.Conv3d(cin, cout, 1, stride, bias=False), nn.GroupNorm(min(8, cout), cout))

    def forward(self, x):
        y = F.relu(self.n1(self.c1(x)))
        y = self.n2(self.c2(y))
        return F.relu(y + (x if self.skip is None else self.skip(x)))


class Encoder3D(nn.Module):
    """Small 3D ResNet. GroupNorm, because Kaggle batch sizes are small."""

    def __init__(self, in_ch: int = 3, widths=(32, 64, 128, 256)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, widths[0], 3, 1, 1, bias=False), nn.GroupNorm(8, widths[0]), nn.ReLU()
        )
        layers, cin = [], widths[0]
        for i, w in enumerate(widths):
            layers += [_Block(cin, w, 1 if i == 0 else 2), _Block(w, w, 1)]
            cin = w
        self.layers = nn.Sequential(*layers)
        self.out_dim = widths[-1]

    def forward(self, x):  # -> (B, C, d, h, w)
        return self.layers(self.stem(x))


class ImageOnly(nn.Module):
    def __init__(self, widths=(32, 64, 128, 256), in_ch: int = 3):
        super().__init__()
        self.encoder = Encoder3D(in_ch, widths=widths)
        self.head = nn.Linear(self.encoder.out_dim, 1)

    def forward(self, batch):
        f = self.encoder(batch["image"]).mean(dim=(2, 3, 4))
        return {"logit": self.head(f).squeeze(1)}


class ConcatFusion(nn.Module):
    """Common baseline: pooled image features concatenated with tabular features."""

    def __init__(self, n_tab: int, widths=(32, 64, 128, 256), hidden: int = 128, in_ch: int = 3):
        super().__init__()
        self.encoder = Encoder3D(in_ch, widths=widths)
        self.tab = nn.Sequential(nn.Linear(2 * n_tab, hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(self.encoder.out_dim + hidden, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, 1))

    def forward(self, batch):
        f = self.encoder(batch["image"]).mean(dim=(2, 3, 4))
        t = self.tab(torch.cat([batch["tab"], batch["tab_missing"]], 1))
        return {"logit": self.head(torch.cat([f, t], 1)).squeeze(1)}


# --------------------------------------------------------------------------- CARF
class TabTokens(nn.Module):
    """One token per tabular feature, with a learned embedding for missing values."""

    def __init__(self, n_tab: int, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_tab, dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_tab, dim))
        self.missing = nn.Parameter(torch.zeros(n_tab, dim))
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x, miss):
        tok = x.unsqueeze(-1) * self.weight + self.bias
        tok = torch.where(miss.unsqueeze(-1) > 0.5, self.missing.expand_as(tok), tok)
        return torch.cat([self.cls.expand(x.shape[0], -1, -1), tok], 1)


class CARF(nn.Module):
    """Clinically anchored residual fusion.

    logit = z_clin + gate * delta
      z_clin : logit of the frozen clinical anchor (an input, not a parameter)
      delta  : residual from cross-attention of tabular tokens over image tokens
      gate   : in [0, 1]; sees the fused state and how far the image embedding is
               from the training distribution (diagonal Mahalanobis distance)
    """

    def __init__(self, n_tab: int, dim: int = 128, heads: int = 4, widths=(32, 64, 128, 256),
                 p_drop_image: float = 0.3, n_phases: int = 3,
                 gate_init: float = 0.0, use_gate: bool = True, use_time: bool = True,
                 aux_weight: float | None = None, in_ch: int = 3):
        super().__init__()
        self._make_encoder(in_ch, widths, dim)
        self.time = nn.Sequential(nn.Linear(n_phases, dim), nn.GELU(), nn.Linear(dim, dim))
        self.tab = TabTokens(n_tab, dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.1)
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Dropout(0.1), nn.Linear(2 * dim, dim))
        self.null_image = nn.Parameter(torch.zeros(1, 1, dim))
        self.delta_head = nn.Linear(dim, 1)
        self.gate_head = nn.Linear(dim + 1, 1)
        nn.init.constant_(self.gate_head.bias, gate_init)  # 0 -> gate starts at 0.5
        self.p_drop_image = p_drop_image
        self.use_gate, self.use_time = use_gate, use_time
        if aux_weight is not None:  # read by train.fit; None keeps fit's default
            self.aux_weight = aux_weight
        self.register_buffer("mu", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("n_seen", torch.zeros(()))

    def _make_encoder(self, in_ch, widths, dim):
        self.encoder = Encoder3D(in_ch, widths=widths)
        self.proj = nn.Conv3d(self.encoder.out_dim, dim, 1)

    def image_tokens(self, image, dt):
        fmap = self.proj(self.encoder(image))  # (B, dim, d, h, w)
        tok = fmap.flatten(2).transpose(1, 2)  # (B, N, dim)
        if self.use_time:
            tok = tok + self.time(torch.log1p(dt) / 6.0).unsqueeze(1)
        return tok, fmap

    def shift_score(self, pooled):
        if self.training:  # running moments of the pooled image embedding
            with torch.no_grad():
                m = 0.99 if self.n_seen > 0 else 0.0
                self.mu.mul_(m).add_((1 - m) * pooled.mean(0))
                if pooled.shape[0] > 1:
                    self.var.mul_(m).add_((1 - m) * pooled.var(0))
                self.n_seen += 1
        d2 = ((pooled - self.mu) ** 2 / (self.var + 1e-4)).mean(1, keepdim=True)
        return torch.log1p(d2).detach()

    def forward(self, batch, drop_image: bool | None = None):
        B = batch["image"].shape[0]
        img, _ = self.image_tokens(batch["image"], batch["dt"])
        shift = self.shift_score(img.mean(1))
        if drop_image is None:
            drop = (torch.rand(B, device=img.device) < self.p_drop_image) if self.training else torch.zeros(B, dtype=torch.bool, device=img.device)
        else:
            drop = torch.full((B,), bool(drop_image), device=img.device)
        img = torch.where(drop[:, None, None], self.null_image.expand_as(img), img)
        q = self.tab(batch["tab"], batch["tab_missing"])
        a, w = self.attn(self.n1(q), img, img, need_weights=True)
        h = q + a
        h = h + self.ffn(self.n2(h))
        cls = h[:, 0]
        delta = self.delta_head(cls).squeeze(1)
        gate = torch.sigmoid(self.gate_head(torch.cat([cls, shift], 1))).squeeze(1)
        if not self.use_gate:  # ablation: plain residual, logit = z_clin + delta
            gate = torch.ones_like(gate)
        gate = torch.where(drop, torch.zeros_like(gate), gate)  # no image -> clinical model
        return {"logit": batch["z_clin"] + gate * delta, "gate": gate, "delta": delta,
                "attn": w[:, 0], "shift": shift.squeeze(1)}


class MonaiImageOnly(nn.Module):
    """Image-only baselines from MONAI (3D DenseNet-121, EfficientNet-B0, ViT)."""

    def __init__(self, kind: str, size: int = 64):
        super().__init__()
        from monai.networks import nets

        if kind == "densenet121":
            self.net = nets.DenseNet121(spatial_dims=3, in_channels=3, out_channels=1)
        elif kind == "efficientnet_b0":
            self.net = nets.EfficientNetBN("efficientnet-b0", pretrained=False, spatial_dims=3, in_channels=3, num_classes=1)
        elif kind == "vit":
            self.net = nets.ViT(in_channels=3, img_size=(size,) * 3, patch_size=(16,) * 3, hidden_size=384, mlp_dim=1536,
                                num_layers=6, num_heads=6, classification=True, num_classes=1, post_activation=None)
        else:
            raise ValueError(kind)
        self.kind = kind

    def forward(self, batch):
        out = self.net(batch["image"])
        if isinstance(out, tuple):  # ViT returns (logits, hidden_states)
            out = out[0]
        return {"logit": out.reshape(-1)}


class SliceEncoder(nn.Module):
    """ImageNet-pretrained 2D ResNet applied to every slice. Input (B, S, 3, P, P) -> (B, S, 512)."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        import torchvision

        self.pretrained = False
        try:
            net = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
            self.pretrained = pretrained
        except Exception as e:  # noqa: BLE001  (no internet): say so loudly, never silently
            print(f"PRETRAINED_WEIGHTS_UNAVAILABLE {e!r}; using random initialisation", flush=True)
            net = torchvision.models.resnet18(weights=None)
        net.fc = nn.Identity()
        self.net, self.out_dim = net, 512

    def forward(self, x):
        B, S = x.shape[:2]
        return self.net(x.flatten(0, 1)).view(B, S, -1)


class Slice25D(nn.Module):
    """2.5D image-only model: pretrained slice encoder + attention pooling over 12 tumour slices."""

    def __init__(self, pretrained: bool = True, hidden: int = 128):
        super().__init__()
        self.encoder = SliceEncoder(pretrained)
        self.att = nn.Sequential(nn.Linear(512, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(512, 1))

    def forward(self, batch):
        f = self.encoder(batch["image"])
        w = torch.softmax(self.att(f), 1)
        return {"logit": self.head((w * f).sum(1)).squeeze(1)}


class CARF25D(CARF):
    """CARF whose image tokens are the 12 slice embeddings of the pretrained 2.5D encoder."""

    def __init__(self, n_tab: int, dim: int = 128, pretrained: bool = True, **kw):
        self._pretrained = pretrained
        super().__init__(n_tab, dim=dim, **kw)

    def _make_encoder(self, in_ch, widths, dim):
        self.encoder = SliceEncoder(self._pretrained)
        self.proj = nn.Linear(512, dim)

    def image_tokens(self, image, dt):
        tok = self.proj(self.encoder(image))
        if self.use_time:
            tok = tok + self.time(torch.log1p(dt) / 6.0).unsqueeze(1)
        return tok, None


MONAI_KINDS = ("densenet121", "efficientnet_b0", "vit")
SLICE_MODELS = ("s25d", "carf_25d")  # these read data.slices(i); every other model reads the 3D volume


def build(name: str, n_tab: int, widths=(32, 64, 128, 256), dim: int = 128, size: int = 64) -> nn.Module:
    if name in MONAI_KINDS:
        return MonaiImageOnly(name, size)
    if name == "image_only":
        return ImageOnly(widths)
    if name == "concat":
        return ConcatFusion(n_tab, widths)
    if name == "carf":
        return CARF(n_tab, dim=dim, widths=widths)
    if name == "carf_nogate":
        return CARF(n_tab, dim=dim, widths=widths, use_gate=False)
    if name == "carf_notime":
        return CARF(n_tab, dim=dim, widths=widths, use_time=False)
    if name == "carf_noaux":
        return CARF(n_tab, dim=dim, widths=widths, aux_weight=0.0)
    # round 2: kinetic inputs (pre, early, late, early-pre, late-early, mask) and pretrained 2.5D
    if name == "image_only_k":
        return ImageOnly(widths, in_ch=6)
    if name == "concat_k":
        return ConcatFusion(n_tab, widths, in_ch=6)
    if name == "carf_k":
        return CARF(n_tab, dim=dim, widths=widths, in_ch=6)
    if name == "s25d":
        return Slice25D()
    if name == "carf_25d":
        return CARF25D(n_tab, dim=dim)
    raise ValueError(name)
