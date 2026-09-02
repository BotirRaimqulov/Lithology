import numpy as np

from lithology.dataset.export import fit_normalization


def test_robust_normalization_clips_extreme_outliers():
    rng = np.random.default_rng(0)
    normal = rng.normal(20, 2, size=(1000, 1))  # typical resistivity-ish range
    matrix = normal.copy()
    matrix[0, 0] = 1e6  # a single extreme sensor glitch

    stats = fit_normalization([matrix], feature_names=["KS"], method="robust", clip_value=10.0)
    normalized = stats.apply(matrix)

    assert normalized.max() <= 10.0
    assert normalized.min() >= -10.0
    # the well-behaved bulk of the data should not be crushed by the clip
    assert np.abs(normalized[1:]).max() < 10.0


def test_clip_value_none_disables_clipping():
    rng = np.random.default_rng(0)
    matrix = rng.normal(20, 2, size=(1000, 1))  # narrow bulk distribution
    matrix[0, 0] = 1e6                          # one extreme outlier

    stats = fit_normalization([matrix], feature_names=["KS"], method="robust", clip_value=None)
    normalized = stats.apply(matrix)
    assert normalized.max() > 10.0  # not clipped: the outlier's z-score vastly exceeds 10


def test_zero_scale_falls_back_to_one_not_division_by_zero():
    matrix = np.full((10, 1), 5.0)  # constant feature -> IQR = 0
    stats = fit_normalization([matrix], feature_names=["X"], method="robust", clip_value=None)
    normalized = stats.apply(matrix)
    assert np.isfinite(normalized).all()
