"""Turn per-depth-point predictions back into continuous geological
intervals (spec section 13), plus interval-level quality metrics (spec
section 21: boundary/top/bottom depth error in meters, interval IoU).

Post-processing strategy (documented, not hidden):

1. Majority-vote smoothing over a small odd window removes single-sample
   label flicker without needing a fixed geological assumption.
2. Runs shorter than ``min_run_points`` are merged into the run they share
   the longer boundary with (a rolling-median-style cleanup), so prediction
   noise cannot fragment one true bed into many spurious one-point
   intervals.
3. The final interval table is the *common refinement* of the (smoothed)
   lithology and zone run-length segmentations: a new interval starts
   whenever either lithology OR zone changes, matching the example output
   format which reports both per interval.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lithology.constants import IGNORE_INDEX


def majority_vote_smooth(label_ids: np.ndarray, window: int = 5) -> np.ndarray:
    """Odd-window majority vote, IGNORE_INDEX positions are excluded from
    voting and left untouched."""
    if window < 3 or window % 2 == 0:
        return label_ids.copy()
    n = len(label_ids)
    radius = window // 2
    out = label_ids.copy()
    for i in range(n):
        if label_ids[i] == IGNORE_INDEX:
            continue
        lo, hi = max(0, i - radius), min(n, i + radius + 1)
        neighborhood = label_ids[lo:hi]
        valid = neighborhood[neighborhood != IGNORE_INDEX]
        if len(valid) == 0:
            continue
        values, counts = np.unique(valid, return_counts=True)
        out[i] = values[np.argmax(counts)]
    return out


def _runs(label_ids: np.ndarray) -> list:
    """Return list of (start_idx, end_idx_exclusive, label) runs."""
    runs = []
    n = len(label_ids)
    if n == 0:
        return runs
    start = 0
    for i in range(1, n + 1):
        if i == n or label_ids[i] != label_ids[start]:
            runs.append((start, i, label_ids[start]))
            start = i
    return runs


def merge_short_runs(label_ids: np.ndarray, min_run_points: int) -> np.ndarray:
    """Merge runs shorter than ``min_run_points`` into whichever
    neighboring run is longer (never drops the point -- it just adopts the
    dominant surrounding label instead of standing alone as noise)."""
    out = label_ids.copy()
    changed = True
    guard = 0
    while changed and guard < 10:
        changed = False
        guard += 1
        runs = _runs(out)
        for idx, (start, end, label) in enumerate(runs):
            length = end - start
            if label == IGNORE_INDEX or length >= min_run_points:
                continue
            prev_run = runs[idx - 1] if idx > 0 else None
            next_run = runs[idx + 1] if idx < len(runs) - 1 else None
            candidates = [r for r in (prev_run, next_run) if r is not None and r[2] != IGNORE_INDEX]
            if not candidates:
                continue
            winner = max(candidates, key=lambda r: r[1] - r[0])
            out[start:end] = winner[2]
            changed = True
    return out


@dataclass
class IntervalTableResult:
    df: pd.DataFrame


def reconstruct_single_task_intervals(
    well_id: str,
    depth: np.ndarray,
    label_ids: np.ndarray,
    confidence: np.ndarray,
    id_to_name: dict,
    label_column: str,
    smoothing_window: int = 5,
    min_run_points: int = 3,
) -> pd.DataFrame:
    """Run-length-encode a SINGLE task's predictions (lithology-only or
    zone-only) into its own interval table, with boundaries driven purely
    by that task's own label changes -- independent of the other task.

    Use this (rather than :func:`reconstruct_well_intervals`, whose interval
    boundaries are the union of lithology AND zone changes) whenever you
    need to verify the two tasks aren't just echoing each other, or want a
    lithology log and a stratigraphic column separately.
    """
    labels = merge_short_runs(majority_vote_smooth(label_ids, smoothing_window), min_run_points)
    n = len(depth)
    rows = []
    if n == 0:
        return pd.DataFrame(columns=["well", "top", "bottom", "thickness", label_column, f"{label_column}_confidence"])

    start = 0
    for i in range(1, n + 1):
        if i == n or labels[i] != labels[start]:
            end = i
            top, bottom = float(depth[start]), float(depth[min(end, n - 1)])
            label_id = int(labels[start])
            rows.append({
                "well": well_id,
                "top": top,
                "bottom": bottom,
                "thickness": bottom - top,
                label_column: id_to_name.get(label_id, "UNKNOWN") if label_id != IGNORE_INDEX else None,
                f"{label_column}_confidence": float(np.mean(confidence[start:end])),
            })
            start = i
    return pd.DataFrame(rows)


def reconstruct_well_intervals(
    well_id: str,
    depth: np.ndarray,
    lithology_pred_ids: np.ndarray,
    lithology_confidence: np.ndarray,
    zone_pred_ids: np.ndarray,
    zone_confidence: np.ndarray,
    boundary_prob: np.ndarray,
    id_to_lithology: dict,
    id_to_zone: dict,
    smoothing_window: int = 5,
    min_run_points: int = 3,
) -> pd.DataFrame:
    litho = merge_short_runs(majority_vote_smooth(lithology_pred_ids, smoothing_window), min_run_points)
    zone = merge_short_runs(majority_vote_smooth(zone_pred_ids, smoothing_window), min_run_points)

    n = len(depth)
    change = np.zeros(n, dtype=bool)
    change[1:] = (litho[1:] != litho[:-1]) | (zone[1:] != zone[:-1])
    change[0] = True

    boundaries = np.where(change)[0].tolist() + [n]
    rows = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        top, bottom = float(depth[start]), float(depth[min(end, n - 1)])
        thickness = bottom - top
        litho_id = int(litho[start])
        zone_id = int(zone[start])
        rows.append({
            "well": well_id,
            "top": top,
            "bottom": bottom,
            "thickness": thickness,
            "lithology": id_to_lithology.get(litho_id, "UNKNOWN") if litho_id != IGNORE_INDEX else None,
            "zone": id_to_zone.get(zone_id, "UNKNOWN") if zone_id != IGNORE_INDEX else None,
            "lithology_confidence": float(np.mean(lithology_confidence[start:end])),
            "zone_confidence": float(np.mean(zone_confidence[start:end])),
            "boundary_confidence": float(np.max(boundary_prob[start:end])),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Interval-level quality metrics
# --------------------------------------------------------------------------- #
def boundary_depth_errors(true_boundaries_m: list, pred_boundaries_m: list, max_match_distance_m: float) -> dict:
    """Match each true boundary to its nearest predicted boundary (in meters).

    Returns matched depth errors plus counts of missed true boundaries and
    unmatched (false-positive) predicted boundaries beyond the match radius.
    """
    if not true_boundaries_m:
        return {"n_true": 0, "n_pred": len(pred_boundaries_m), "matched_errors_m": [], "n_missed": 0,
                "n_false_positive": len(pred_boundaries_m)}
    pred = sorted(pred_boundaries_m)
    used = [False] * len(pred)
    errors, missed = [], 0
    for tb in true_boundaries_m:
        best_j, best_d = None, None
        for j, pb in enumerate(pred):
            if used[j]:
                continue
            d = abs(pb - tb)
            if best_d is None or d < best_d:
                best_d, best_j = d, j
        if best_j is not None and best_d <= max_match_distance_m:
            used[best_j] = True
            errors.append(best_d)
        else:
            missed += 1
    n_false_positive = sum(1 for u in used if not u)
    return {
        "n_true": len(true_boundaries_m), "n_pred": len(pred_boundaries_m),
        "matched_errors_m": errors, "mean_error_m": float(np.mean(errors)) if errors else None,
        "n_missed": missed, "n_false_positive": n_false_positive,
    }


def interval_iou(true_intervals: list, pred_intervals: list) -> dict:
    """``true_intervals``/``pred_intervals``: list of (top, bottom, label).

    Each true interval is matched to the predicted interval of the SAME
    label with maximum overlap; IoU is reported per match and averaged.
    """
    ious = []
    for t_top, t_bottom, t_label in true_intervals:
        best_iou = 0.0
        for p_top, p_bottom, p_label in pred_intervals:
            if p_label != t_label:
                continue
            inter = max(0.0, min(t_bottom, p_bottom) - max(t_top, p_top))
            union = (t_bottom - t_top) + (p_bottom - p_top) - inter
            if union > 0:
                best_iou = max(best_iou, inter / union)
        ious.append(best_iou)
    return {"n_true_intervals": len(true_intervals), "mean_iou": float(np.mean(ious)) if ious else None,
            "per_interval_iou": ious}
