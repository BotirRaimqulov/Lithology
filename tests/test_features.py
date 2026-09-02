import numpy as np

from lithology.config import FeaturesConfig
from lithology.features.engineering import build_features, resolve_context_points


def test_resolve_context_points_meters_and_points():
    assert resolve_context_points("points", 15, step=0.1) == 15
    assert resolve_context_points("meters", 7.0, step=0.1) == 70
    assert resolve_context_points("meters", 1.5, step=0.5) == 3


def test_build_features_no_nans_and_mask_present():
    n = 50
    rng = np.random.default_rng(0)
    gk = np.sin(np.linspace(0, 6, n)) + rng.normal(0, 0.05, n)
    gk[10:13] = np.nan
    curves = {"GK": gk, "KS": np.cos(np.linspace(0, 6, n)) * 10, "PS": -0.1 * np.ones(n)}

    fb = build_features(curves, step=0.1, config=FeaturesConfig())
    assert not np.isnan(fb.matrix).any()
    assert "GK_missing_mask" in fb.names
    mask_col = fb.matrix[:, fb.names.index("GK_missing_mask")]
    assert mask_col[10:13].sum() == 3
    assert fb.raw_missing_mask["GK"][10:13].all()


def test_depth_is_never_a_feature():
    curves = {"GK": np.zeros(10), "KS": np.zeros(10), "PS": np.zeros(10)}
    fb = build_features(curves, step=0.1, config=FeaturesConfig())
    assert not any("DEPT" in name.upper() for name in fb.names)


def test_entirely_missing_curve_does_not_crash():
    n = 20
    curves = {"GK": np.full(n, np.nan), "KS": np.zeros(n), "PS": np.zeros(n)}
    fb = build_features(curves, step=0.1, config=FeaturesConfig())
    assert not np.isnan(fb.matrix).any()
    assert fb.raw_missing_mask["GK"].all()
