import numpy as np

from lithology.alignment.intervals import Interval, assign_intervals_to_depth


def test_non_overlapping_half_open():
    depth = np.round(np.arange(0, 2.01, 0.1), 2)
    intervals = [Interval(0.0, 1.0, "A", 0), Interval(1.0, 2.0, "B", 1)]
    payload, diag = assign_intervals_to_depth(depth, intervals, semantics="half_open")
    assert payload[0] == "A"
    assert payload[9] == "A"     # depth 0.9 still A
    assert payload[10] == "B"    # depth 1.0 -> B (half-open excludes 1.0 from A)
    assert payload[-1] == "B"    # last point of well is kept even though half-open would exclude it
    assert diag.n_conflicting_points == 0


def test_closed_semantics_includes_both_boundaries_and_flags_conflict():
    depth = np.round(np.arange(0, 2.01, 0.1), 2)
    intervals = [Interval(0.0, 1.0, "A", 0), Interval(1.0, 2.0, "B", 1)]
    payload, diag = assign_intervals_to_depth(depth, intervals, semantics="closed")
    assert payload[10] is None  # depth 1.0 claimed by both -> conflict, not guessed
    assert diag.n_conflicting_points == 1


def test_out_of_range_interval_is_reported_not_silently_applied():
    depth = np.round(np.arange(0, 1.01, 0.1), 2)
    intervals = [Interval(5.0, 6.0, "A", 0)]
    payload, diag = assign_intervals_to_depth(depth, intervals, semantics="half_open")
    assert (payload == None).all()  # noqa: E711
    assert diag.n_out_of_range == 1
    assert diag.out_of_range_row_indices == [0]


def test_duplicate_intervals_detected():
    depth = np.round(np.arange(0, 1.01, 0.1), 2)
    intervals = [Interval(0.0, 0.5, "A", 0), Interval(0.0, 0.5, "B", 1)]
    _, diag = assign_intervals_to_depth(depth, intervals, semantics="half_open")
    assert diag.n_duplicate_pairs == 1
    assert diag.duplicate_row_index_pairs == [(0, 1)]


def test_tuple_payload_does_not_trigger_numpy_broadcast_error():
    # Regression test: interval-schema lithology stores payload as a
    # (code, core_verified) tuple. A naive `payload[mask] = interval.payload`
    # makes NumPy try to broadcast the tuple's 2 elements across every
    # masked position instead of storing it as one object.
    depth = np.round(np.arange(0, 5.0, 0.1), 2)  # 50 points
    intervals = [Interval(0.0, 5.0, ("5", True), 0)]
    payload, diag = assign_intervals_to_depth(depth, intervals, semantics="half_open")
    assert diag.n_conflicting_points == 0
    assert all(p == ("5", True) for p in payload)
