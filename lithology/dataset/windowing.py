"""Depth-window extraction and the training-time crop dataset.

Design decision (see module docstrings in ``features/engineering.py`` and
the model architecture doc): the exported dataset stores one FULL per-well
sequence, not a pre-materialized window per depth point -- pre-exploding
every overlapping ±N-point window would multiply storage by the window
length for no benefit, since a deep 1D ResNet builds its receptive field
from the whole sequence via stacked small kernels rather than needing a
literal ``Conv1d(kernel_size=2N+1)``. ``sampling.context_size`` therefore
configures two things that must agree:

  1. the ResNet1D's receptive field (see models/resnet1d.py), and
  2. the minimum training crop length used here, so every crop is at least
     large enough for the receptive field to be exercised away from the
     crop's own edges.

:func:`extract_window` is kept for diagnostics/visualization (e.g. "show
me the ±7 m window around this point") and is not on the training hot path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from lithology.features.engineering import resolve_context_points

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # torch is only required for Phase 9+ (model training)
    torch = None
    Dataset = object


def extract_window(array: np.ndarray, center_idx: int, radius: int) -> np.ndarray:
    """Return a fixed-length ``2*radius+1`` window centered at ``center_idx``,
    edge-padded (replicated) when the window runs off either end of the array.

    Works for both 1D (n,) and 2D (n, features) arrays (window taken along axis 0).
    """
    n = array.shape[0]
    lo, hi = center_idx - radius, center_idx + radius + 1
    pad_left = max(0, -lo)
    pad_right = max(0, hi - n)
    clipped = array[max(lo, 0) : min(hi, n)]
    if pad_left or pad_right:
        pad_width = [(pad_left, pad_right)] + [(0, 0)] * (array.ndim - 1)
        clipped = np.pad(clipped, pad_width, mode="edge")
    return clipped


@dataclass
class WellArrays:
    """Full per-well arrays as produced by dataset export -- what the
    training-time crop dataset slices from."""

    well_id: str
    depth: np.ndarray
    features: np.ndarray            # (n, F)
    lithology_label: np.ndarray     # (n,) int, IGNORE_INDEX where unsupervised
    zone_label: np.ndarray          # (n,) int, IGNORE_INDEX where unsupervised
    boundary_label: np.ndarray      # (n,) int
    step: Optional[float]


class CropDataset(Dataset):
    """Yields fixed-length crops from a list of :class:`WellArrays`.

    A crop is at least ``min_crop_points`` long (derived from the
    receptive-field context so the model always has room to use it), and
    never crosses a well boundary. Short wells are returned whole
    (edge-padded up to ``min_crop_points``) rather than dropped.
    """

    def __init__(self, wells: list, min_crop_points: int, stride: Optional[int] = None):
        if torch is None:
            raise ImportError("PyTorch is required for CropDataset (pip install torch).")
        self.wells = wells
        self.min_crop_points = min_crop_points
        self.stride = stride or max(min_crop_points // 2, 1)
        self._index: list[tuple[int, int]] = []  # (well_idx, start)
        for wi, w in enumerate(wells):
            n = len(w.depth)
            if n <= min_crop_points:
                self._index.append((wi, 0))
                continue
            starts = list(range(0, n - min_crop_points + 1, self.stride))
            if starts[-1] != n - min_crop_points:
                starts.append(n - min_crop_points)
            self._index.extend((wi, s) for s in starts)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        wi, start = self._index[idx]
        w = self.wells[wi]
        n = len(w.depth)
        length = min(self.min_crop_points, n)
        end = start + length

        feats = w.features[start:end]
        litho = w.lithology_label[start:end]
        zone = w.zone_label[start:end]
        boundary = w.boundary_label[start:end]

        if length < self.min_crop_points:
            pad = self.min_crop_points - length
            feats = np.pad(feats, [(0, pad), (0, 0)], mode="edge")
            litho = np.pad(litho, (0, pad), constant_values=-100)
            zone = np.pad(zone, (0, pad), constant_values=-100)
            boundary = np.pad(boundary, (0, pad), constant_values=-100)

        return {
            "well_id": w.well_id,
            "start": start,
            "features": torch.from_numpy(feats.astype(np.float32)).transpose(0, 1),  # (F, L)
            "lithology_label": torch.from_numpy(litho.astype(np.int64)),
            "zone_label": torch.from_numpy(zone.astype(np.int64)),
            "boundary_label": torch.from_numpy(boundary.astype(np.int64)),
        }


def min_crop_points_from_context(context_unit: str, context_size: float, step: float,
                                  min_multiple: int = 4) -> int:
    """A crop should span several context radii so the receptive field is
    exercised well away from the crop's own edges."""
    radius = resolve_context_points(context_unit, context_size, step)
    return max(radius * 2 + 1, radius * min_multiple)
