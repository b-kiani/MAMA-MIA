import json
import os

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from mmonc import explain, metrics, stats
from mmonc.data import InMemory
from mmonc.models import CARF, build
from mmonc.pipeline import run
from mmonc.radiomics import SIMPLE_NAMES, extract_simple
from mmonc.synthetic import make_cohort
from mmonc.train import Cases, fit

TINY = {"epochs": 2, "folds": 2, "widths": [8, 16], "dim": 16, "n_boot": 50, "patience": 2}


def _batch(n=4, n_tab=6, size=16):
    g = torch.Generator().manual_seed(0)
    return {"image": torch.randn(n, 3, size, size, size, generator=g), "tab": torch.randn(n, n_tab, generator=g),
            "tab_missing": torch.zeros(n, n_tab), "dt": torch.rand(n, 3, generator=g) * 400,
            "z_clin": torch.randn(n, generator=g), "y": torch.tensor([0.0, 1.0, 0.0, 1.0])}


def test_delong_matches_auc_and_is_antisymmetric():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.3, 400)
    p1 = y * 0.5 + rng.normal(0, 1, 400)
    p2 = y * 1.5 + rng.normal(0, 1, 400)
    d = stats.delong(y, p1, p2)
    assert d["auc1"] == pytest.approx(roc_auc_score(y, p1), abs=1e-9)
    assert d["auc2"] == pytest.approx(roc_auc_score(y, p2), abs=1e-9)
    assert d["diff"] > 0 and d["p"] < 0.01
    assert stats.delong(y, p2, p1)["diff"] == pytest.approx(-d["diff"])


def test_delong_se_agrees_with_bootstrap():
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.4, 600)
    p1, p2 = y + rng.normal(0, 1.2, 600), y + rng.normal(0, 1.0, 600)
    d = stats.delong(y, p1, p2)
    diffs = []
    for _ in range(600):
        i = rng.integers(0, 600, 600)
        if 0 < y[i].sum() < 600:
            diffs.append(roc_auc_score(y[i], p2[i]) - roc_auc_score(y[i], p1[i]))
    assert d["se"] == pytest.approx(np.std(diffs), rel=0.2)


def test_holm_known_values():
    assert np.allclose(stats.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_calibration_of_perfectly_calibrated_predictions():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.05, 0.95, 20000)
    y = rng.binomial(1, p)
    c = metrics.calibration(y, p)
    assert c["slope"] == pytest.approx(1.0, abs=0.06)
    assert c["intercept"] == pytest.approx(0.0, abs=0.06)
    assert c["ece"] < 0.02


def test_net_benefit_treat_all_identity():
    y = np.array([1, 0, 0, 1, 0])
    t = np.array([0.2, 0.5])
    assert np.allclose(metrics.net_benefit(y, np.ones(5), t), metrics.net_benefit_all(y, t))


def test_classification_counts():
    m = metrics.classification([1, 1, 0, 0], [0.9, 0.2, 0.8, 0.1], 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 1)
    assert m["balanced_accuracy"] == 0.5


def test_radiomics_shape_and_volume():
    tb, im, mk = make_cohort(6, size=16)
    f = extract_simple(im, mk)
    assert f.shape == (6, len(SIMPLE_NAMES))
    assert np.allclose(f[:, SIMPLE_NAMES.index("volume_vox")], mk.reshape(6, -1).sum(1))


def test_carf_without_image_is_exactly_the_clinical_anchor():
    m = CARF(n_tab=6, dim=16, widths=(8, 16)).eval()
    b = _batch()
    out = m(b, drop_image=True)
    assert torch.allclose(out["logit"], b["z_clin"])
    assert torch.all(out["gate"] == 0)


def test_carf_gate_range_and_gradients():
    m = CARF(n_tab=6, dim=16, widths=(8, 16)).train()
    out = m(_batch(), drop_image=False)
    assert torch.all((out["gate"] > 0) & (out["gate"] < 1))
    out["logit"].sum().backward()
    assert m.encoder.stem[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("name", ["image_only", "concat", "carf"])
def test_models_forward(name):
    out = build(name, 6, (8, 16), 16).eval()(_batch())
    assert out["logit"].shape == (4,)


def test_gradcam_shape_and_range():
    cam = explain.gradcam(build("carf", 6, (8, 16), 16), _batch())
    assert cam.shape == (4, 16, 16, 16) and cam.min() >= 0 and cam.max() <= 1 + 1e-6


def test_checkpoint_resume(tmp_path):
    tb, im, mk = make_cohort(24, size=16)
    n_tab = 4
    ds = Cases(im, np.zeros((24, n_tab)), np.zeros((24, n_tab)), np.zeros((24, 3)), np.zeros(24), tb.pcr.to_numpy())
    ck = str(tmp_path / "m.pt")
    h1 = fit(build("image_only", n_tab, (8, 16)), ds, ds, epochs=1, ckpt_path=ck, log=lambda *a: None, patience=5)
    h2 = fit(build("image_only", n_tab, (8, 16)), ds, ds, epochs=3, ckpt_path=ck, log=lambda *a: None, patience=5)
    assert len(h1) == 1 and [h["epoch"] for h in h2] == [0, 1, 2]


def test_pipeline_end_to_end(tmp_path):
    tb, im, mk = make_cohort(160, size=16, seed=3)
    out = str(tmp_path / "run")
    s = run(InMemory(tb, im, mk, extract_simple(im, mk)), out, TINY, synthetic=True, log=lambda *a: None)
    assert s["synthetic"] and s["n_dev"] + s["n_ext"] == 160
    for f in ["table1_cohort.csv", "table2_internal.csv", "table3_external.csv", "table5_ablation.csv",
              "table6_explainability.csv", "predictions_external.csv", "frozen.json",
              "figures/fig2_roc_pr_external.png", "figures/fig4_decision_curve_external.pdf"]:
        assert os.path.exists(os.path.join(out, f)), f
    assert "split_hash" in json.load(open(os.path.join(out, "frozen.json")))


def test_external_cases_never_used_for_thresholds(tmp_path):
    """Changing external labels must not change anything frozen on development data."""
    tb, im, mk = make_cohort(160, size=16, seed=3)
    rad = extract_simple(im, mk)
    cfg = {**TINY, "deep_models": []}
    run(InMemory(tb, im, mk, rad), str(tmp_path / "a"), cfg, True, log=lambda *a: None)
    tb2 = tb.copy()
    ext = tb2.collection != "ISPY2"
    tb2.loc[ext, "pcr"] = 1 - tb2.loc[ext, "pcr"]
    run(InMemory(tb2, im, mk, rad), str(tmp_path / "b"), cfg, True, log=lambda *a: None)
    a, b = (json.load(open(tmp_path / k / "frozen.json")) for k in "ab")
    assert a["thresholds"] == b["thresholds"] and a["split_hash"] == b["split_hash"]


def test_development_mode_writes_no_external_outputs(tmp_path):
    tb, im, mk = make_cohort(120, size=16, seed=4)
    out = str(tmp_path / "dev")
    s = run(InMemory(tb, im, mk, extract_simple(im, mk)), out, {**TINY, "deep_models": [], "score_external": False},
            True, log=lambda *a: None)
    assert s["external_scored"] is False
    assert os.path.exists(os.path.join(out, "table2_internal.csv"))
    assert not os.path.exists(os.path.join(out, "table3_external.csv"))
    assert not os.path.exists(os.path.join(out, "predictions_external.csv"))
