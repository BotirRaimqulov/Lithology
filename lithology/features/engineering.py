"""Per-well feature engineering.

Deliberately small and geologically/signal-processing justified, per the
task spec ("do NOT blindly create hundreds of hand-engineered features"):

  * raw curve (gap-interpolated so the network sees a continuous signal)
  * missing-value mask (so the network can tell "measured 0" from "no data")
  * 1st/2nd derivative w.r.t. depth (local trend / curvature -- the
    "behavior" the spec asks for, not the absolute level)
  * rolling mean/std/min/max over one or more physically-sized windows
    (local variability and persistence of a pattern)
  * rolling GK-KS / GK-PS / KS-PS correlation (cross-curve relationship;
    correlation is used instead of raw differences because GK/KS/PS live
    on incompatible physical units, so a subtraction would not be a
    meaningful "relationship" signal)

Absolute DEPTH is intentionally never added as a model feature: the model
must key off curve *behavior*, not memorize "this depth number means this
zone" for wells it has already seen (spec section 6).

Everything is computed per well from that well's own curves; no statistic
here is fit across wells (that happens later, deliberately only on the
train split, in ``dataset.split``/``dataset.windowing``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from lithology.config import FeaturesConfig

RAW_CURVES = ("GK", "KS", "PS")
CROSS_PAIRS = (("GK", "KS"), ("GK", "PS"), ("KS", "PS"))


def resolve_context_points(unit: str, size: float, step: Optional[float]) -> int:
    """Convert a configured context size (points or meters) into a point radius."""
    if unit == "points":
        return max(int(round(size)), 0)
    if unit == "meters":
        if not step or step <= 0:
            raise ValueError("Cannot convert a meters-based context size without a valid STEP.")
        return max(int(round(size / step)), 0)
    raise ValueError(f"Unknown context_unit {unit!r}")


def _interpolate_gaps(curve: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate NaN gaps (edges filled by nearest valid value).

    Returns (filled, missing_mask). If the curve is entirely NaN, `filled`
    is all zeros and `missing_mask` is all True -- the caller/model must
    rely purely on the mask feature in that case, never on a fabricated
    value.
    """
    missing_mask = np.isnan(curve)
    if missing_mask.all():
        return np.zeros_like(curve), missing_mask
    filled = pd.Series(curve).interpolate(method="linear", limit_direction="both").to_numpy()
    return filled, missing_mask


def _nth_derivative(curve: np.ndarray, step: float, order: int) -> np.ndarray:
    out = curve
    spacing = step if step and step > 0 else 1.0
    for _ in range(order):
        out = np.gradient(out, spacing)
    return out


def _rolling(series: pd.Series, window_pts: int, stat: str) -> np.ndarray:
    window_pts = max(window_pts, 1)
    roller = series.rolling(window=window_pts, center=True, min_periods=1)
    if stat == "mean":
        return roller.mean().to_numpy()
    if stat == "std":
        return roller.std().fillna(0.0).to_numpy()
    if stat == "min":
        return roller.min().to_numpy()
    if stat == "max":
        return roller.max().to_numpy()
    raise ValueError(f"Unknown rolling stat {stat!r}")


@dataclass
class FeatureBundle:
    matrix: np.ndarray                  # (n_points, n_features), float32
    names: list = field(default_factory=list)
    raw_missing_mask: dict = field(default_factory=dict)   # curve -> bool array (True = was NaN)


def build_features(curves: dict, step: Optional[float], config: FeaturesConfig) -> FeatureBundle:
    """``curves`` maps canonical curve name -> 1D np.ndarray (NaN = missing)."""
    n = len(next(iter(curves.values()))) if curves else 0
    columns: list[np.ndarray] = []
    names: list[str] = []
    filled_cache: dict[str, np.ndarray] = {}
    missing_masks: dict[str, np.ndarray] = {}
    rolling_cache: dict[tuple, np.ndarray] = {}

    for c in RAW_CURVES:
        curve = curves.get(c, np.full(n, np.nan))
        filled, missing = _interpolate_gaps(curve)
        filled_cache[c] = filled
        missing_masks[c] = missing

        if config.use_raw:
            columns.append(filled)
            names.append(f"{c}_raw")
        if config.use_missing_mask:
            columns.append(missing.astype(np.float32))
            names.append(f"{c}_missing_mask")
        if config.use_derivatives:
            for order in config.derivative_orders:
                columns.append(_nth_derivative(filled, step, order))
                names.append(f"{c}_d{order}")
        if config.use_rolling_statistics:
            series = pd.Series(filled)
            for window_m in config.rolling_windows_m:
                window_pts = resolve_context_points("meters", window_m, step) * 2 + 1 if step else 3
                for stat in config.rolling_stats:
                    key = (c, window_m, stat)
                    values = _rolling(series, window_pts, stat)
                    rolling_cache[key] = values
                    columns.append(values)
                    names.append(f"{c}_roll{window_m}m_{stat}")

    if config.use_cross_curve and n > 0:
        window_m = config.rolling_windows_m[-1] if config.rolling_windows_m else 3.0
        window_pts = resolve_context_points("meters", window_m, step) * 2 + 1 if step else 5
        for a, b in CROSS_PAIRS:
            sa, sb = pd.Series(filled_cache[a]), pd.Series(filled_cache[b])
            corr = sa.rolling(window=max(window_pts, 2), center=True, min_periods=2).corr(sb)
            corr = corr.fillna(0.0).to_numpy()
            columns.append(corr)
            names.append(f"{a}_{b}_corr{window_m}m")

    matrix = np.stack(columns, axis=1).astype(np.float32) if columns else np.zeros((n, 0), dtype=np.float32)
    return FeatureBundle(matrix=matrix, names=names, raw_missing_mask=missing_masks)
