"""Generic interval -> depth-point assignment with explicit conflict handling.

Used for both stratigraphic zones and interval-schema lithology. Overlaps,
duplicates, and out-of-range intervals are never resolved silently: every
depth point that is claimed by more than one interval is reported as a
conflict and its label is dropped back to "unassigned" (never an arbitrary
pick of one of the conflicting intervals) so training never sees a label
the source data does not unambiguously support.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class Interval:
    top: float
    bottom: float
    payload: Any            # e.g. zone name, or {"code":..., "core_verified":...}
    row_index: int           # traceability back to the source CSV row


@dataclass
class IntervalAssignmentDiagnostics:
    n_intervals: int = 0
    n_out_of_range: int = 0
    out_of_range_row_indices: list = field(default_factory=list)
    n_clipped: int = 0
    clipped_row_indices: list = field(default_factory=list)
    n_duplicate_pairs: int = 0
    duplicate_row_index_pairs: list = field(default_factory=list)
    n_conflicting_points: int = 0
    conflicting_interval_row_indices: list = field(default_factory=list)
    coverage_fraction: float = 0.0


def _ranges_overlap(a_top, a_bottom, b_top, b_bottom, eps) -> bool:
    return (a_top < b_bottom - eps) and (b_top < a_bottom - eps)


def detect_duplicates(intervals: list[Interval], eps: float = 1e-6) -> list[tuple]:
    pairs = []
    for i in range(len(intervals)):
        for j in range(i + 1, len(intervals)):
            a, b = intervals[i], intervals[j]
            if abs(a.top - b.top) <= eps and abs(a.bottom - b.bottom) <= eps:
                pairs.append((a.row_index, b.row_index))
    return pairs


def assign_intervals_to_depth(
    depth: np.ndarray,
    intervals: list[Interval],
    semantics: str = "half_open",
    eps: Optional[float] = None,
) -> tuple[np.ndarray, IntervalAssignmentDiagnostics]:
    """Assign each interval's payload to the depth points it covers.

    Returns ``(payload_per_point, diagnostics)`` where ``payload_per_point``
    is an object array of length ``len(depth)`` (``None`` = unassigned or
    conflicting).
    """
    n = len(depth)
    diag = IntervalAssignmentDiagnostics(n_intervals=len(intervals))
    payload = np.full(n, None, dtype=object)
    owner_row = np.full(n, -1, dtype=int)
    conflict = np.zeros(n, dtype=bool)

    if n == 0 or not intervals:
        return payload, diag

    if eps is None:
        eps = 1e-6

    depth_min, depth_max = float(depth.min()), float(depth.max())
    diag.duplicate_row_index_pairs = detect_duplicates(intervals, eps=eps)
    diag.n_duplicate_pairs = len(diag.duplicate_row_index_pairs)

    for interval in sorted(intervals, key=lambda iv: iv.top):
        top, bottom = interval.top, interval.bottom

        if bottom < depth_min - eps or top > depth_max + eps:
            diag.n_out_of_range += 1
            diag.out_of_range_row_indices.append(interval.row_index)
            continue
        clipped_top, clipped_bottom = max(top, depth_min), min(bottom, depth_max)
        if clipped_top != top or clipped_bottom != bottom:
            diag.n_clipped += 1
            diag.clipped_row_indices.append(interval.row_index)

        if semantics == "half_open":
            mask = (depth >= clipped_top - eps) & (depth < clipped_bottom - eps)
            # keep the very last sample of the well if this interval reaches the well's end
            if clipped_bottom >= depth_max - eps:
                mask = mask | np.isclose(depth, depth_max, atol=eps)
        elif semantics == "closed":
            mask = (depth >= clipped_top - eps) & (depth <= clipped_bottom + eps)
        else:
            raise ValueError(f"Unknown interval_semantics {semantics!r}")

        if not mask.any():
            continue

        overlap_mask = mask & (owner_row != -1)
        if overlap_mask.any():
            conflict[overlap_mask] = True
            diag.conflicting_interval_row_indices.append(interval.row_index)

        new_mask = mask & (owner_row == -1)
        payload[new_mask] = interval.payload
        owner_row[new_mask] = interval.row_index

    diag.n_conflicting_points = int(conflict.sum())
    payload[conflict] = None  # never guess which of the conflicting intervals is right

    diag.coverage_fraction = float((payload != None).sum() / n)  # noqa: E711
    return payload, diag
