import torch

from mmonc.models import CARF, CARFMix


def _batch(n_tab=5, tab=None):
    g = torch.Generator().manual_seed(0)
    return dict(image=torch.randn(3, 6, 16, 16, 16, generator=g), dt=torch.rand(3, 3, generator=g) * 100,
                tab=torch.randn(3, n_tab, generator=g) if tab is None else tab, tab_missing=torch.zeros(3, n_tab),
                z_clin=torch.zeros(3))


def test_original_carf_ignores_tabular_tokens():
    torch.manual_seed(0)
    m = CARF(5, dim=16, widths=(8, 16), in_ch=6).eval()
    a, b = m(_batch()), m(_batch(tab=torch.randn(3, 5) * 5))
    assert torch.equal(a["delta"], b["delta"]) and torch.equal(a["gate"], b["gate"])


def test_carfmix_uses_tabular_tokens_and_bounds_residual():
    torch.manual_seed(0)
    m = CARFMix(5, dim=16, widths=(8, 16), in_ch=6, bound=1.0).eval()
    a, b = m(_batch()), m(_batch(tab=torch.randn(3, 5) * 5))
    assert (a["delta"] - b["delta"]).abs().max() > 0
    with torch.no_grad():
        m.delta_head.bias.fill_(50.0)
    assert (m(_batch())["delta"].abs() <= 1.0).all()
