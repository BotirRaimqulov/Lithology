import numpy as np

from lithology.alignment.depth_alignment import align_well
from lithology.config import AlignmentConfig
from lithology.constants import IGNORE_INDEX
from lithology.io.lithology_csv import LithologyRecord
from lithology.io.stratigraphy_csv import StratigraphyRecord


def _depth(n=21, step=0.1):
    return np.round(np.arange(0, n * step, step), 2)[:n]


def test_point_sample_labels_only_nearest_point_not_whole_interval():
    depth = _depth()
    zones = [StratigraphyRecord("W", "Q4", 0.0, 2.0, 2.0, 2.0, 0)]
    litho = [LithologyRecord("W", 0.35, 0.35, True, "5", "5", False, "Lithology", 0)]
    aw = align_well("W", depth, 0.1, zones, litho, AlignmentConfig())

    n_labeled = sum(1 for v in aw.lithology_label_raw if v is not None)
    assert n_labeled == 1  # NOT spread across the whole zone/interval
    assert aw.lithology_label_raw[3] == "5"  # snapped to depth 0.3 (nearest within tol)


def test_overlapping_zones_are_nulled_and_reported():
    depth = _depth()
    zones = [
        StratigraphyRecord("W", "Q4", 0.0, 1.0, 1.0, 1.0, 0),
        StratigraphyRecord("W", "N1", 1.0, 2.0, 1.0, 1.0, 1),
        StratigraphyRecord("W", "OVERLAP", 0.5, 1.5, 1.0, 1.0, 2),
    ]
    aw = align_well("W", depth, 0.1, zones, [], AlignmentConfig())
    assert aw.diagnostics.zone.n_conflicting_points > 0
    # the disputed region has no label at all
    disputed = aw.zone_label[6:14]
    assert all(v is None for v in disputed)


def test_core_verified_filtering():
    depth = _depth()
    litho = [
        LithologyRecord("W", 0.1, 0.1, True, "4*", "4", True, "Lithology", 0),
        LithologyRecord("W", 0.2, 0.2, True, "5", "5", False, "Lithology", 1),
    ]
    cfg = AlignmentConfig(require_core_verified_for_lithology=True)
    aw = align_well("W", depth, 0.1, [], litho, cfg)
    assert aw.lithology_label[1] == "4"       # core-verified -> kept
    assert aw.lithology_label[2] is None      # not core-verified -> filtered out
    assert aw.lithology_label_raw[2] == "5"   # raw value preserved for transparency


def test_boundary_label_ignore_where_no_coverage_positive_near_true_boundary():
    depth = _depth()
    zones = [StratigraphyRecord("W", "Q4", 0.0, 1.0, 1.0, 1.0, 0)]  # only covers first half
    aw = align_well("W", depth, 0.1, zones, [], AlignmentConfig(boundary_tolerance_points=1))
    assert aw.boundary_label[15] == IGNORE_INDEX  # far outside any coverage
    assert aw.boundary_label[0] == 1              # well start is itself a zone top
    assert aw.boundary_label[5] == 0              # interior, covered, not near a boundary
