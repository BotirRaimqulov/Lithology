import pytest

from lithology.config import SplitConfig
from lithology.dataset.split import SplitError, split_wells


def test_holdout_split_disjoint():
    cfg = SplitConfig(min_wells_for_holdout_split=4, train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=1)
    wells = [f"W{i}" for i in range(10)]
    result = split_wells(wells, cfg)
    assert result.mode == "holdout"
    all_assigned = set(result.train) | set(result.val) | set(result.test)
    assert all_assigned == set(wells)
    assert set(result.train) & set(result.val) == set()
    assert set(result.train) & set(result.test) == set()
    assert set(result.val) & set(result.test) == set()


def test_small_well_count_uses_cross_validation():
    cfg = SplitConfig(min_wells_for_holdout_split=8, n_folds=3, seed=1)
    wells = [f"W{i}" for i in range(4)]
    result = split_wells(wells, cfg)
    assert result.mode == "cross_validation"
    for fold in result.folds:
        assert set(fold["train"]) & set(fold["val"]) == set()
        assert set(fold["train"]) | set(fold["val"]) == set(wells)


def test_explicit_split_never_drops_a_well():
    cfg = SplitConfig(train_wells=["A", "B"], val_wells=["C"], test_wells=[])
    result = split_wells(["A", "B", "C", "D"], cfg)
    assert result.mode == "explicit"
    assert "D" in result.train  # unaccounted-for well parked in train, never dropped


def test_explicit_split_rejects_overlap():
    cfg = SplitConfig(train_wells=["A", "B"], val_wells=["B"])
    with pytest.raises(SplitError):
        split_wells(["A", "B"], cfg)


def test_explicit_split_rejects_unknown_well():
    cfg = SplitConfig(train_wells=["A", "ZZZ"])
    with pytest.raises(SplitError):
        split_wells(["A", "B"], cfg)
