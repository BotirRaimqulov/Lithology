"""Geological log visualization (spec section 22): GK/KS/PS curves, expert
vs. predicted lithology and zone tracks, and boundary probability, all
sharing a depth axis so agreement/disagreement/uncertainty/missing data
are visible at a glance.
"""
from __future__ import annotations

from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless-safe; caller decides whether to also show()
import matplotlib.pyplot as plt
import numpy as np


def _robust_xlim(values: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0, pad_frac: float = 0.1):
    """A percentile-based x-axis range for one curve track.

    Real well-log curves (especially resistivity/KS) can have a handful of
    extreme outlier spikes -- matplotlib's default autoscale stretches the
    axis to include them, which crushes the actual, geologically
    meaningful oscillation into a barely-visible sliver near one edge.
    Clipping the visible range to the 1st-99th percentile (with padding)
    keeps the outlier off-screen (the curve simply runs off the plot edge
    there, which is itself a visible signal that something anomalous is
    happening) while keeping the normal range readable.
    """
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return None
    lo, hi = np.percentile(finite, [low_pct, high_pct])
    if hi - lo < 1e-9:
        lo, hi = float(finite.min()), float(finite.max())
        if hi - lo < 1e-9:
            hi = lo + 1.0
    pad = (hi - lo) * pad_frac
    return lo - pad, hi + pad


def _categorical_track(ax, depth: np.ndarray, labels: np.ndarray, title: str, color_map: dict):
    """Draw a labeled color-band track (one horizontal bar of varying color
    per contiguous run of the same label)."""
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    n = len(depth)
    if n == 0:
        return
    start = 0
    for i in range(1, n + 1):
        if i == n or labels[i] != labels[start]:
            label = labels[start]
            color = color_map.get(label, "#cccccc") if label is not None else "white"
            ax.axhspan(depth[start], depth[i - 1] if i - 1 < n else depth[-1], color=color)
            if label is not None and (depth[i - 1] - depth[start]) > 0:
                ax.text(0.5, (depth[start] + depth[min(i - 1, n - 1)]) / 2, str(label),
                        ha="center", va="center", fontsize=6, rotation=90)
            start = i
    ax.set_ylim(depth.max(), depth.min())


def _normalize_labels(labels: np.ndarray) -> np.ndarray:
    """Coerce a mixed None/NaN/str/number array (as produced by round-tripping
    through Parquet, which turns ``None`` into ``float('nan')`` in an object
    column) into a consistent representation: ``None`` for missing, ``str``
    for everything else, so labels are always mutually orderable/comparable.
    """
    def _one(v):
        if v is None:
            return None
        if isinstance(v, float) and np.isnan(v):
            return None
        return str(v)

    return np.array([_one(v) for v in labels], dtype=object)


def _build_color_map(labels: np.ndarray) -> dict:
    uniques = sorted({v for v in labels if v is not None})
    cmap = plt.get_cmap("tab20")
    return {v: cmap(i % 20) for i, v in enumerate(uniques)}


def plot_well_log(
    well_id: str,
    depth: np.ndarray,
    curves: dict,
    expert_lithology: Optional[np.ndarray] = None,
    predicted_lithology: Optional[np.ndarray] = None,
    expert_zone: Optional[np.ndarray] = None,
    predicted_zone: Optional[np.ndarray] = None,
    boundary_probability: Optional[np.ndarray] = None,
    out_path: Optional[str] = None,
    figsize=(14, 10),
):
    expert_lithology = _normalize_labels(expert_lithology) if expert_lithology is not None else None
    predicted_lithology = _normalize_labels(predicted_lithology) if predicted_lithology is not None else None
    expert_zone = _normalize_labels(expert_zone) if expert_zone is not None else None
    predicted_zone = _normalize_labels(predicted_zone) if predicted_zone is not None else None

    tracks = ["GK", "KS", "PS"]
    n_categorical = sum(x is not None for x in (expert_lithology, predicted_lithology, expert_zone, predicted_zone))
    n_extra = n_categorical + (1 if boundary_probability is not None else 0)
    n_cols = len(tracks) + n_extra
    fig, axes = plt.subplots(1, n_cols, figsize=figsize, sharey=True)
    if n_cols == 1:
        axes = [axes]
    fig.suptitle(f"Well {well_id}")

    for ax, curve_name in zip(axes, tracks):
        values = curves.get(curve_name)
        if values is None:
            ax.set_visible(False)
            continue
        missing = np.isnan(values)
        ax.plot(values, depth, linewidth=0.8, color="tab:blue")
        if missing.any():
            ax.scatter(np.zeros(missing.sum()), depth[missing], marker="x", color="red", s=8,
                       label="missing")
        xlim = _robust_xlim(values)
        if xlim:
            ax.set_xlim(*xlim)
        ax.set_title(curve_name, fontsize=9)
        ax.invert_yaxis()
        ax.set_ylim(depth.max(), depth.min())

    col = len(tracks)
    litho_color_map = _build_color_map(
        np.concatenate([a for a in (expert_lithology, predicted_lithology) if a is not None])
        if (expert_lithology is not None or predicted_lithology is not None) else np.array([])
    )
    zone_color_map = _build_color_map(
        np.concatenate([a for a in (expert_zone, predicted_zone) if a is not None])
        if (expert_zone is not None or predicted_zone is not None) else np.array([])
    )

    if expert_lithology is not None:
        _categorical_track(axes[col], depth, expert_lithology, "Lithology\n(expert)", litho_color_map)
        col += 1
    if predicted_lithology is not None:
        _categorical_track(axes[col], depth, predicted_lithology, "Lithology\n(predicted)", litho_color_map)
        col += 1
    if expert_zone is not None:
        _categorical_track(axes[col], depth, expert_zone, "Zone\n(expert)", zone_color_map)
        col += 1
    if predicted_zone is not None:
        _categorical_track(axes[col], depth, predicted_zone, "Zone\n(predicted)", zone_color_map)
        col += 1
    if boundary_probability is not None:
        ax = axes[col]
        ax.plot(boundary_probability, depth, color="black", linewidth=0.8)
        ax.fill_betweenx(depth, 0, boundary_probability, alpha=0.3, color="orange")
        ax.set_title("Boundary\nprobability", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(depth.max(), depth.min())

    axes[0].set_ylabel("Depth (m)")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    return fig
