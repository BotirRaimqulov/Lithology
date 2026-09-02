import numpy as np

from lithology.viz.plot_log import _robust_xlim


def test_robust_xlim_ignores_extreme_outlier():
    rng = np.random.default_rng(0)
    values = 20 + 5 * rng.normal(size=1000)
    values[0] = 50000  # one resistivity-tool-glitch-style spike

    lo, hi = _robust_xlim(values)
    # the outlier must not dominate the range -- bulk data (roughly 5..35)
    # should stay comfortably inside it, and the spike should stay outside.
    assert lo > -50
    assert hi < 100
    assert hi < 50000


def test_robust_xlim_handles_constant_curve():
    values = np.full(100, 7.0)
    lo, hi = _robust_xlim(values)
    assert lo < 7.0 < hi


def test_robust_xlim_handles_all_nan():
    values = np.full(100, np.nan)
    assert _robust_xlim(values) is None
