"""Turn expert CSV intervals/points into per-depth-point labels for one well.

Three arrays come out of :func:`align_well`, all the same length as the
well's LAS depth curve:

  * ``zone_label``               -- stratigraphic zone name, or None
  * ``lithology_label`` / ``_raw`` -- lithology code, or None
  * ``boundary_label``            -- 1 at/near a true top/bottom, 0 elsewhere
                                      covered, IGNORE_INDEX where nothing is
                                      known at all

Lithology is handled specially per the project's core requirement: a
point-based (core/lab, MD-only) sample labels ONLY the nearest depth
sample(s) within a snap tolerance -- it is never spread across a
surrounding interval. An interval-schema lithology row (top/bottom given)
is spread across its covered points like a stratigraphy zone, but a
point-sample assignment always takes precedence at the exact depth it
covers, since it is the higher-fidelity, directly-verified measurement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from lithology.alignment.intervals import Interval, IntervalAssignmentDiagnostics, assign_intervals_to_depth
from lithology.config import AlignmentConfig
from lithology.constants import BOUNDARY_POSITIVE, IGNORE_INDEX
from lithology.io.lithology_csv import LithologyRecord
from lithology.io.stratigraphy_csv import StratigraphyRecord

LITHOLOGY_SOURCE_NONE = "none"
LITHOLOGY_SOURCE_CORE_SAMPLE = "core_sample"
LITHOLOGY_SOURCE_INTERVAL = "interval"

_CONFIDENCE_BY_SOURCE = {
    LITHOLOGY_SOURCE_NONE: 0.0,
    LITHOLOGY_SOURCE_CORE_SAMPLE: 1.0,
    LITHOLOGY_SOURCE_INTERVAL: 0.6,
}


@dataclass
class PointSampleDiagnostics:
    n_records: int = 0
    n_unmatched: int = 0
    unmatched_row_indices: list = field(default_factory=list)
    n_conflicting_points: int = 0
    conflicting_row_indices: list = field(default_factory=list)


@dataclass
class WellAlignmentDiagnostics:
    well_id: str
    n_points: int
    zone: IntervalAssignmentDiagnostics
    lithology_interval: IntervalAssignmentDiagnostics
    lithology_points: PointSampleDiagnostics
    n_boundary_positive: int = 0
    n_boundary_ignored: int = 0


@dataclass
class AlignedWell:
    well_id: str
    depth: np.ndarray
    zone_label: np.ndarray            # object array, values are str or None
    lithology_label: np.ndarray       # object array, filtered per config.require_core_verified_for_lithology
    lithology_label_raw: np.ndarray   # object array, unfiltered
    lithology_source: np.ndarray      # object array of str in {"none","core_sample","interval"}
    lithology_core_verified: np.ndarray  # bool array
    lithology_confidence: np.ndarray  # float array
    boundary_label: np.ndarray        # int array: 0/1, or IGNORE_INDEX
    diagnostics: WellAlignmentDiagnostics


def _assign_point_samples(
    depth: np.ndarray, step_actual: Optional[float], records: list[LithologyRecord]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, PointSampleDiagnostics]:
    """Snap each point (core/lab) sample to its nearest depth index.

    Returns (code_array, core_verified_array, row_index_owner_array, diagnostics).
    """
    n = len(depth)
    codes = np.full(n, None, dtype=object)
    core_verified = np.zeros(n, dtype=bool)
    owner_row = np.full(n, -1, dtype=int)
    diag = PointSampleDiagnostics(n_records=len(records))

    if n == 0 or not records:
        return codes, core_verified, owner_row, diag

    tol = max(step_actual, 1e-6) / 2.0 if step_actual else 1e-6
    conflict = np.zeros(n, dtype=bool)

    for rec in records:
        idx = int(np.searchsorted(depth, rec.top))
        best_idx = None
        best_dist = None
        for cand in (idx - 1, idx, idx + 1):
            if 0 <= cand < n:
                d = abs(depth[cand] - rec.top)
                if best_dist is None or d < best_dist:
                    best_dist, best_idx = d, cand
        if best_idx is None or best_dist > tol + 1e-9:
            diag.n_unmatched += 1
            diag.unmatched_row_indices.append(rec.row_index)
            continue

        if owner_row[best_idx] != -1 and codes[best_idx] != rec.code:
            conflict[best_idx] = True
            diag.conflicting_row_indices.append(rec.row_index)
            continue
        codes[best_idx] = rec.code
        core_verified[best_idx] = core_verified[best_idx] or rec.core_verified
        owner_row[best_idx] = rec.row_index

    codes[conflict] = None
    core_verified[conflict] = False
    diag.n_conflicting_points = int(conflict.sum())
    return codes, core_verified, owner_row, diag


def align_well(
    well_id: str,
    depth: np.ndarray,
    step_actual: Optional[float],
    zone_records: list[StratigraphyRecord],
    lithology_records: list[LithologyRecord],
    config: AlignmentConfig,
) -> AlignedWell:
    n = len(depth)
    eps = max(step_actual * 1e-3, 1e-9) if step_actual else 1e-6

    # --- stratigraphic zone -------------------------------------------------
    zone_intervals = [
        Interval(top=r.top, bottom=r.bottom, payload=r.zone_name, row_index=r.row_index)
        for r in zone_records
    ]
    zone_label, zone_diag = assign_intervals_to_depth(depth, zone_intervals, config.interval_semantics, eps)

    # --- lithology: split point vs. interval schema -------------------------
    point_records = [r for r in lithology_records if r.is_point_sample]
    interval_records = [r for r in lithology_records if not r.is_point_sample]

    interval_payloads = [
        Interval(top=r.top, bottom=r.bottom, payload=(r.code, r.core_verified), row_index=r.row_index)
        for r in interval_records
    ]
    litho_interval_label, litho_interval_diag = assign_intervals_to_depth(
        depth, interval_payloads, config.interval_semantics, eps
    )

    point_codes, point_core_verified, point_owner, point_diag = _assign_point_samples(
        depth, step_actual, point_records
    )

    lithology_label_raw = np.full(n, None, dtype=object)
    lithology_source = np.full(n, LITHOLOGY_SOURCE_NONE, dtype=object)
    lithology_core_verified = np.zeros(n, dtype=bool)

    for i in range(n):
        interval_payload = litho_interval_label[i]
        if interval_payload is not None:
            code, core_verified = interval_payload
            lithology_label_raw[i] = code
            lithology_source[i] = LITHOLOGY_SOURCE_INTERVAL
            lithology_core_verified[i] = core_verified

    # Point samples take precedence -- directly verified, higher fidelity.
    point_mask = point_owner != -1
    lithology_label_raw[point_mask] = point_codes[point_mask]
    lithology_source[point_mask] = LITHOLOGY_SOURCE_CORE_SAMPLE
    lithology_core_verified[point_mask] = point_core_verified[point_mask]

    lithology_confidence = np.array(
        [_CONFIDENCE_BY_SOURCE[s] for s in lithology_source], dtype=float
    )
    lithology_confidence[lithology_core_verified] = 1.0

    if config.require_core_verified_for_lithology:
        keep = lithology_core_verified
        lithology_label = np.where(keep, lithology_label_raw, None)
    else:
        lithology_label = lithology_label_raw.copy()

    # --- boundary label ------------------------------------------------------
    boundary_label = np.full(n, IGNORE_INDEX, dtype=int)
    covered = (zone_label != None) | (lithology_label_raw != None)  # noqa: E711
    boundary_label[covered] = 0

    boundary_depths = set()
    for r in zone_records:
        boundary_depths.add(r.top)
        boundary_depths.add(r.bottom)
    for r in interval_records:
        boundary_depths.add(r.top)
        boundary_depths.add(r.bottom)

    tol_pts = max(int(config.boundary_tolerance_points), 0)
    for d in boundary_depths:
        if d < depth.min() - eps or d > depth.max() + eps:
            continue
        idx = int(np.argmin(np.abs(depth - d)))
        lo, hi = max(0, idx - tol_pts), min(n - 1, idx + tol_pts)
        boundary_label[lo : hi + 1] = BOUNDARY_POSITIVE

    diagnostics = WellAlignmentDiagnostics(
        well_id=well_id,
        n_points=n,
        zone=zone_diag,
        lithology_interval=litho_interval_diag,
        lithology_points=point_diag,
        n_boundary_positive=int((boundary_label == BOUNDARY_POSITIVE).sum()),
        n_boundary_ignored=int((boundary_label == IGNORE_INDEX).sum()),
    )

    return AlignedWell(
        well_id=well_id,
        depth=depth,
        zone_label=zone_label,
        lithology_label=lithology_label,
        lithology_label_raw=lithology_label_raw,
        lithology_source=lithology_source,
        lithology_core_verified=lithology_core_verified,
        lithology_confidence=lithology_confidence,
        boundary_label=boundary_label,
        diagnostics=diagnostics,
    )
