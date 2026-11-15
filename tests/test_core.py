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


@pytest.mark.parametrize("kind", ["densenet121", "efficientnet_b0", "vit"])
def test_monai_backbones_forward(kind):
    pytest.importorskip("monai")
    b = _batch(n=2, size=64)
    b["y"] = b["y"][:2]
    out = build(kind, 6, size=64).eval()(b)
    assert out["logit"].shape == (2,)


def test_real_table_builder_parses_mamamia_columns(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("openpyxl")
    from mmonc.prep import build_table

    raw = pd.DataFrame({
        "patient_id": ["DUKE_001", "ISPY2_1"], "dataset": ["DUKE", "ISPY2"], "pcr": [0, 1], "age": [41, 50],
        "hr": [0, 1], "er": [0, 1], "pr": [0, 0], "her2": [1, 0], "tumor_subtype": ["her2_pure", "luminal"],
        "menopause": ["pre", "post"], "multifocal_cancer": [0, 1], "bilateral_breast_cancer": [0, 0],
        "ethnicity": ["caucasian", "asian"], "bmi_group": [None, None], "manufacturer": ["GE", "SIEMENS"],
        "field_strength": [1.5, 3.0], "view": ["axial", "axial"], "site": ["Duke", "UCSF"], "slice_thickness": [1.1, 2.0],
        "num_phases": [4, 6], "acquisition_date": ["1984/10/13", None], "nac_agent": [None, None],
        "acquisition_times": ["[0, 584, 714, 846]", "[0, 132, 223]"],
        "days_to_follow_up": [2940.0, None], "days_to_recurrence": [407.0, None]})
    raw.to_excel(tmp_path / "clinical_and_imaging_info.xlsx", index=False)
    t = build_table(str(tmp_path))
    assert list(t.menopausal) == [0.0, 1.0]
    assert t.dt1.tolist() == [584.0, 132.0]
    assert t.rfs_event.iloc[0] == 1 and abs(t.rfs_months.iloc[0] - 407 / 30.44) < 1e-6
    assert np.isnan(t.rfs_event.iloc[1])


@pytest.mark.parametrize("name", ["carf_nogate", "carf_notime", "carf_noaux"])
def test_carf_ablations(name):
    m = build(name, 6, (8, 16), 16).eval()
    b = _batch()
    out = m(b, drop_image=False)
    assert out["logit"].shape == (4,)
    assert torch.allclose(m(b, drop_image=True)["logit"], b["z_clin"])  # fallback holds for every variant
    if name == "carf_nogate":
        assert torch.all(out["gate"] == 1)
    if name == "carf_noaux":
        assert m.aux_weight == 0.0


# ----------------------------------------------------------------------------- round 2
def test_kinetic_features_recover_planted_kinetics():
    from mmonc.prep2 import KIN_NAMES, kinetic_features

    rng = np.random.default_rng(0)
    g = np.stack(np.meshgrid(*[np.arange(40)] * 3, indexing="ij"))
    m = np.sqrt(((g - 20) ** 2).sum(0)) < 10
    pre = np.full((40,) * 3, 100.0) + rng.normal(0, 1, (40,) * 3)
    early = pre + m * 150.0          # PE = 1.5 inside the tumour
    late = pre + m * 100.0           # washout: SER = 1.5
    f = kinetic_features(pre, early, late, m)
    assert set(f) == set(KIN_NAMES)
    assert f["pe_mean"] == pytest.approx(1.5, abs=0.1)
    assert f["ftv_frac"] > 0.95 and f["frac_washout"] > 0.95
    assert f["ser_mean"] == pytest.approx(1.5, abs=0.1)
    assert abs(f["peri_pe_mean"]) < 0.1
    f2 = kinetic_features(pre, early, early, m, has_late=False)
    assert np.isnan(f2["ser_mean"]) and np.isnan(f2["frac_washout"])


def test_round2_models_forward_and_carf25d_fallback():
    from mmonc.models import CARF25D, Slice25D

    b = _batch(n_tab=5)
    b["image"] = torch.randn(4, 6, 16, 16, 16)
    for name in ("image_only_k", "concat_k", "carf_k"):
        assert build(name, 5, (8, 16), 16)(b)["logit"].shape == (4,)
    s = dict(b, image=torch.randn(4, 12, 3, 32, 32))
    assert Slice25D(pretrained=False)(s)["logit"].shape == (4,)
    m = CARF25D(5, dim=16, pretrained=False).eval()
    r = m(s, drop_image=True)
    assert torch.allclose(r["logit"], s["z_clin"])  # no image -> exactly the clinical logit


def test_round2_pipeline_runs_and_keeps_external_out_of_training(tmp_path):
    import pandas as pd

    from mmonc.pipeline2 import Runner
    from mmonc.synthetic import make_cohort2

    data = make_cohort2(96)
    cfg = {"epochs": 1, "folds": 2, "control_folds": 2, "widths": [8, 16], "dim": 16, "patience": 1, "batch_size": 8,
           "controls": ["hr", "perm"]}
    seen = []
    r = Runner(data, str(tmp_path), cfg, lambda *m: None)
    orig = r.train_one

    def spy(name, tr, va, *a, **k):
        seen.append((set(tr) | set(va)))
        return orig(name, tr, va, *a, **k)

    r.train_one = spy
    r.devext(["tabular", "image_only_k", "carf_k"], "t")
    P = pd.read_csv(tmp_path / "preds_devext_t.csv")
    ext = set(np.flatnonzero((data.table.collection != "ISPY2").to_numpy()))
    assert all(not (s & ext) for s in seen)  # no external case ever reaches training or early stopping
    for col in ("Clinical", "Clinical + FTV", "Clinical + kinetic radiomics", "Image only, kinetic 3D ResNet", "CARF, kinetic"):
        assert P[col].between(0, 1).all()
    assert set(P.set) == {"dev", "ext"}
    r.controls("t")
    C = pd.read_csv(tmp_path / "preds_controls_t.csv")
    assert set(C.target) == {"hr", "perm"} and not ((C.target == "perm") & (C.set == "ext")).any()
    seen.clear()
    r.loco(["carf_k"], "t")
    L = pd.read_csv(tmp_path / "preds_loco_t.csv")
    assert len(L) == len(data.table) and L["CARF, kinetic"].between(0, 1).all()
    held = [set(np.flatnonzero((data.table.collection == h).to_numpy())) for h in sorted(data.table.collection.unique())]
    assert all(not (s & h) for s, h in zip(seen, held))  # the held-out collection is never trained on


def test_deadline_skips_new_fits(tmp_path, monkeypatch):
    from mmonc.pipeline2 import Runner
    from mmonc.synthetic import make_cohort2

    monkeypatch.setenv("MMONC_DEADLINE", "1")
    r = Runner(make_cohort2(64), str(tmp_path), {"folds": 2, "widths": [8, 16], "dim": 16}, lambda *m: None)
    r.devext(["tabular", "image_only_k"], "t")
    import pandas as pd

    P = pd.read_csv(tmp_path / "preds_devext_t.csv")
    assert "Clinical" in P and "Image only, kinetic 3D ResNet" not in P


def test_prep2_crop_and_pack_on_anisotropic_nifti(tmp_path):
    sitk = pytest.importorskip("SimpleITK")
    from mmonc.prep2 import crop_case, kinetic_features, pack

    rng = np.random.default_rng(0)
    shape, sp = (30, 90, 100), (0.7, 0.7, 2.5)  # z, y, x voxels; spacing x, y, z in mm
    zz, yy, xx = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    d = np.sqrt(((zz - 12) * 2.5) ** 2 + ((yy - 50) * 0.7) ** 2 + ((xx - 30) * 0.7) ** 2)
    m = d < 9  # 18 mm sphere, off-centre
    pre = 200 + rng.normal(0, 2, shape)
    paths = []
    for k, a in enumerate([pre, pre + m * 300, pre + m * 240]):
        img = sitk.GetImageFromArray(a.astype(np.float32)); img.SetSpacing(sp)
        p = str(tmp_path / f"c_{k:04d}.nii.gz"); sitk.WriteImage(img, p); paths.append(p)
    mi = sitk.GetImageFromArray(m.astype(np.uint8)); mi.SetSpacing(sp)
    mp = str(tmp_path / "c.nii.gz"); sitk.WriteImage(mi, mp)
    ph, mk = crop_case(paths, mp)
    assert ph.shape[0] == 3 and ph.shape[1] == ph.shape[2] == ph.shape[3] == mk.shape[0]
    assert 44 <= mk.shape[0] <= 52                       # 48 mm minimum cube on a 1 mm grid
    assert mk.sum() == pytest.approx(4 / 3 * np.pi * 9 ** 3, rel=0.15)  # volume preserved in mm^3
    f = kinetic_features(ph[0], ph[1], ph[2], mk)
    assert f["pe_mean"] == pytest.approx(1.5, abs=0.15) and f["ser_mean"] == pytest.approx(1.25, abs=0.1)
    vol, m64, sl = pack(ph, mk, size=32, plane=48)
    assert vol.shape == (3, 32, 32, 32) and sl.shape == (12, 3, 48, 48) and m64.any()
    v0 = vol[0].astype(np.float32)
    assert float(v0[v0 > 0.5].mean()) == pytest.approx(1.0, abs=0.05)  # pre-contrast scaled to about 1 (zeros = padding)
    assert all(float(s[1].max()) > 2.0 for s in sl.astype(np.float32))  # every slice passes through enhancing tumour
