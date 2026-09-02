import numpy as np

from lithology.postprocess.interval_reconstruction import (
    boundary_depth_errors, interval_iou, majority_vote_smooth, merge_short_runs,
    reconstruct_single_task_intervals, reconstruct_well_intervals,
)


def test_majority_vote_removes_single_point_flicker():
    labels = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0])
    smoothed = majority_vote_smooth(labels, window=5)
    assert smoothed[4] == 0  # the lone "1" flip is voted out


def test_merge_short_runs_absorbs_into_longer_neighbor():
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2])
    merged = merge_short_runs(labels, min_run_points=3)
    assert 1 not in merged  # the length-2 run of "1" gets absorbed


def test_reconstruct_intervals_matches_expected_columns():
    depth = np.round(np.arange(0, 10, 0.1), 2)
    n = len(depth)
    litho = np.zeros(n, dtype=int)
    litho[40:] = 1
    zone = np.zeros(n, dtype=int)
    zone[50:] = 1
    conf = np.ones(n)
    boundary = np.zeros(n)

    df = reconstruct_well_intervals(
        "W1", depth, litho, conf, zone, conf, boundary, {0: "Sandstone", 1: "Shale"}, {0: "Q4", 1: "N1"},
    )
    assert list(df.columns) == [
        "well", "top", "bottom", "thickness", "lithology", "zone",
        "lithology_confidence", "zone_confidence", "boundary_confidence",
    ]
    assert len(df) == 3  # [0,4) [4,5) [5,10)
    assert df.iloc[0]["lithology"] == "Sandstone"
    assert df.iloc[-1]["zone"] == "N1"


def test_boundary_depth_errors_meters():
    result = boundary_depth_errors([4.0, 5.0], [4.05, 5.2], max_match_distance_m=0.5)
    assert result["n_missed"] == 0
    assert abs(result["mean_error_m"] - 0.125) < 1e-6


def test_single_task_intervals_are_independent_of_the_other_task():
    # Lithology changes at depth 4, zone changes at depth 7 -- a single-task
    # reconstruction of each must reflect ONLY its own boundaries, proving
    # the two tasks are not silently coupled into one segmentation.
    depth = np.round(np.arange(0, 10, 0.1), 2)
    n = len(depth)
    litho = np.zeros(n, dtype=int)
    litho[40:] = 1
    zone = np.zeros(n, dtype=int)
    zone[70:] = 1
    conf = np.ones(n)

    litho_df = reconstruct_single_task_intervals("W1", depth, litho, conf, {0: "Sandstone", 1: "Shale"}, "lithology")
    zone_df = reconstruct_single_task_intervals("W1", depth, zone, conf, {0: "Q4", 1: "N1"}, "zone")

    assert list(litho_df["top"]) == [0.0, 4.0]
    assert list(zone_df["top"]) == [0.0, 7.0]
    assert "zone" not in litho_df.columns
    assert "lithology" not in zone_df.columns


def test_interval_iou_perfect_match():
    result = interval_iou([(0, 4, "A"), (4, 10, "B")], [(0, 4, "A"), (4, 10, "B")])
    assert result["mean_iou"] == 1.0
