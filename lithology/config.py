"""Typed configuration for the whole pipeline.

Nothing in this project hard-codes a path, a context size, or a loss
weight -- every knob mentioned in the task spec lives here and is loaded
from a single YAML file (see ``configs/default.yaml``). Every dataclass
is plain-old-data so the whole config can be dumped back to YAML/JSON and
stored next to a checkpoint for reproducibility (spec section 20).
"""
from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


# --------------------------------------------------------------------------- #
# Section: data
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    las_dir: str = "data/las"
    lithology_csv: str = "data/csv/lithology.csv"
    stratigraphy_csv: str = "data/csv/stratigraphy.csv"
    output_dir: str = "outputs"


# --------------------------------------------------------------------------- #
# Section: sampling / geological context
# --------------------------------------------------------------------------- #
@dataclass
class SamplingConfig:
    # "points" or "meters". See lithology.features.engineering.resolve_context_points
    context_unit: str = "meters"
    context_size: float = 7.0

    def __post_init__(self) -> None:
        if self.context_unit not in ("points", "meters"):
            raise ValueError(
                f"sampling.context_unit must be 'points' or 'meters', got {self.context_unit!r}"
            )
        if self.context_size <= 0:
            raise ValueError("sampling.context_size must be > 0")


# --------------------------------------------------------------------------- #
# Section: interval -> point alignment semantics
# --------------------------------------------------------------------------- #
@dataclass
class AlignmentConfig:
    # "closed"    -> [top, bottom]   (bottom depth included)
    # "half_open" -> [top, bottom)   (bottom depth excluded, belongs to the
    #                 next interval instead)
    interval_semantics: str = "half_open"

    # Character(s) that mark a lithology row as coming from a physical core /
    # laboratory sample rather than a log-based (visual/geophysical) call.
    # Preserved from the raw CSV rather than assumed -- see io.lithology_csv.
    core_marker_symbols: tuple = ("*",)

    # If True, lithology_label is only supervised (non-IGNORE_INDEX) at depth
    # points whose covering interval was core/lab verified. If False, all
    # covered points are supervised regardless of source, and core-verification
    # is only kept as an auxiliary `lithology_core_verified` flag/confidence.
    require_core_verified_for_lithology: bool = False

    # Half width (in samples) of the positive region around a true boundary
    # depth for the boundary-detection head. A boundary head trained on a
    # single-sample spike is extremely class-imbalanced and brittle to a
    # +/-1 sample pick error in the expert top/bottom; widening the positive
    # band by a small tolerance is standard practice for this kind of task.
    boundary_tolerance_points: int = 1

    # Intervals thinner than this (in meters) are flagged in the quality
    # report as suspicious but are NOT silently dropped.
    min_interval_thickness_m: float = 0.0


# --------------------------------------------------------------------------- #
# Section: feature engineering
# --------------------------------------------------------------------------- #
@dataclass
class FeaturesConfig:
    use_raw: bool = True
    use_missing_mask: bool = True
    use_derivatives: bool = True
    derivative_orders: tuple = (1, 2)          # first + second derivative
    use_rolling_statistics: bool = True
    rolling_windows_m: tuple = (1.0, 3.0)      # in meters, converted via STEP
    rolling_stats: tuple = ("mean", "std", "min", "max")
    use_cross_curve: bool = True               # GK-KS, GK-PS, KS-PS relationships


# --------------------------------------------------------------------------- #
# Section: model
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    # "resnet1d" is a deep residual 1D-CNN encoder -- it already performs the
    # convolutional feature extraction, so no separate "plain CNN" module is
    # stacked in front of it (see lithology/models/resnet1d.py docstring).
    encoder: str = "resnet1d"
    base_channels: int = 64
    num_blocks: tuple = (2, 2, 2, 2)
    kernel_size: int = 7
    dropout: float = 0.1

    # Optional encoder placed after the ResNet1D to mix long-range context
    # across the window: "none", "gru", or "transformer".
    sequence_encoder: str = "gru"
    sequence_hidden_size: int = 128
    sequence_num_layers: int = 1
    sequence_num_heads: int = 4  # only used when sequence_encoder == "transformer"

    num_classes_lithology: Optional[int] = None  # resolved from data at build time
    num_classes_zone: Optional[int] = None       # resolved from data at build time


# --------------------------------------------------------------------------- #
# Section: training
# --------------------------------------------------------------------------- #
@dataclass
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    weight_lithology: float = 1.0
    weight_zone: float = 1.0
    weight_boundary: float = 1.0
    class_balanced_loss: bool = True
    seed: int = 42
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    early_stopping_patience: int = 10
    num_workers: int = 0
    # Clips the gradient norm before every optimizer step. Real well-log
    # curves (especially resistivity/KS) routinely have extreme outlier
    # spikes; without this, one bad batch can produce an exploding gradient
    # that pushes weights to NaN/Inf, permanently corrupting the model for
    # every batch after it. Standard practice for sequence models; disable
    # by setting to 0 or a negative value.
    grad_clip_norm: float = 5.0


# --------------------------------------------------------------------------- #
# Section: train/val/test split
# --------------------------------------------------------------------------- #
@dataclass
class SplitConfig:
    strategy: str = "group_by_well"
    train_frac: float = 0.7
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 42
    # Optional explicit override -- if given, takes precedence over the
    # fractional split above. This is how a documented, fixed split is
    # produced/reproduced (spec section 14: "Document the exact split").
    train_wells: Optional[list] = None
    val_wells: Optional[list] = None
    test_wells: Optional[list] = None
    # Below this many wells, k-fold grouped cross-validation is used instead
    # of a single train/val/test split.
    min_wells_for_holdout_split: int = 8
    n_folds: int = 5


# --------------------------------------------------------------------------- #
# Section: normalization
# --------------------------------------------------------------------------- #
@dataclass
class NormalizationConfig:
    method: str = "robust"  # "robust" (median/IQR) or "standard" (mean/std)
    # Statistics are ALWAYS fit on the train split only (see dataset.split);
    # this flag exists purely so it is visible/auditable in the saved config.
    fit_split: str = "train"
    # After centering/scaling, clip to +/-clip_value. Robust (median/IQR)
    # scaling reduces sensitivity to outliers but does not bound them --
    # a real resistivity (KS) spike orders of magnitude above the typical
    # range can still normalize to a huge value and overflow through
    # BatchNorm/GELU/sigmoid during training. Set to None to disable.
    clip_value: Optional[float] = 10.0


# --------------------------------------------------------------------------- #
# Section: augmentation
# --------------------------------------------------------------------------- #
@dataclass
class AugmentationConfig:
    enabled: bool = False
    gaussian_noise_std: float = 0.0       # in units of normalized curve std
    amplitude_scale_range: tuple = (1.0, 1.0)
    mask_probability: float = 0.0         # probability of masking a curve at a point
    max_depth_shift_points: int = 0       # small window jitter


# --------------------------------------------------------------------------- #
# Section: dataset export
# --------------------------------------------------------------------------- #
@dataclass
class DatasetConfig:
    format: str = "parquet"   # "parquet" or "npz"
    out_dir: str = "dataset"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    # --------------------------------------------------------------------- #
    @staticmethod
    def _merge(base: dict, override: dict) -> dict:
        out = dict(base)
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = Config._merge(out[k], v)
            else:
                out[k] = v
        return out

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        path = Path(path)
        raw = yaml.safe_load(path.read_text()) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "Config":
        defaults = dataclasses.asdict(cls())
        merged = cls._merge(defaults, raw or {})
        section_types = {
            "data": DataConfig,
            "sampling": SamplingConfig,
            "alignment": AlignmentConfig,
            "features": FeaturesConfig,
            "model": ModelConfig,
            "training": TrainingConfig,
            "split": SplitConfig,
            "normalization": NormalizationConfig,
            "augmentation": AugmentationConfig,
            "dataset": DatasetConfig,
        }
        kwargs = {}
        for section, dtype in section_types.items():
            kwargs[section] = dtype(**merged.get(section, {}))
        return cls(**kwargs)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    def copy(self) -> "Config":
        return copy.deepcopy(self)


def load_config(path: Optional[str | Path] = None) -> Config:
    """Load config from ``path``, or return pure defaults if ``path`` is None."""
    if path is None:
        return Config()
    return Config.from_yaml(path)
