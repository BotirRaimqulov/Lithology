"""Shared inference logic: run a trained checkpoint on a raw .las file that
was never part of the exported training dataset.

Used by both ``tools/predict_well.py`` (one file) and
``tools/predict_batch.py`` (every unlabeled/new file in a directory) so
the parse -> feature -> normalize -> model -> reconstruct -> save pipeline
is defined exactly once.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from lithology.config import FeaturesConfig, ModelConfig
from lithology.features.engineering import build_features
from lithology.io.curve_aliases import CurveAliasResolver
from lithology.io.las_parser import parse_las_file
from lithology.postprocess.interval_reconstruction import (
    majority_vote_smooth, merge_short_runs, reconstruct_single_task_intervals, reconstruct_well_intervals,
)
from lithology.viz.plot_log import plot_well_log

try:
    import torch
except ImportError as e:  # pragma: no cover
    raise ImportError("PyTorch is required for inference (pip install torch).") from e

from lithology.models.multitask import MultiTaskLithologyModel


class PredictionError(Exception):
    """Raised for a per-file problem (schema mismatch, missing checkpoint
    data, model divergence) that should be reported and skipped rather
    than crashing an entire batch run."""


@dataclass
class PredictionBundle:
    """Everything loaded once from a checkpoint, reused across many files."""

    model: "torch.nn.Module"
    feature_names: list
    label_maps: dict
    id_to_lithology: dict
    id_to_zone: dict
    normalization: dict
    features_config: FeaturesConfig


def load_prediction_bundle(checkpoint_path: str) -> PredictionBundle:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    feature_names = ckpt["feature_names"]
    label_maps = ckpt["label_maps"]

    normalization = ckpt.get("normalization")
    if normalization is None:
        raise PredictionError(
            f"{checkpoint_path} has no embedded normalization stats (it was trained before "
            f"that fix, or trained in cross-validation mode). Retrain with the current "
            f"tools/train.py to get a checkpoint usable on a fresh LAS file."
        )

    features_config_dict = ckpt.get("features_config")
    features_config = FeaturesConfig(**features_config_dict) if features_config_dict else FeaturesConfig()

    model = MultiTaskLithologyModel(in_features=len(feature_names), config=ModelConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return PredictionBundle(
        model=model,
        feature_names=feature_names,
        label_maps=label_maps,
        id_to_lithology={v: k for k, v in label_maps["lithology"].items()},
        id_to_zone={v: k for k, v in label_maps["zone"].items()},
        normalization=normalization,
        features_config=features_config,
    )


@dataclass
class PredictionSummary:
    well_id: str
    las_path: str
    out_dir: str
    n_points: int
    n_lithology_intervals: int
    n_zone_intervals: int
    parser_warnings: list
    missing_curves: list


def predict_las_file(las_path: str, bundle: PredictionBundle, out_dir: str) -> PredictionSummary:
    """Run the full inference pipeline on one raw .las file and write its
    interval CSVs + plot under ``out_dir``. Raises :class:`PredictionError`
    on any per-file problem (schema mismatch, divergence) instead of
    crashing -- callers processing many files should catch this per file.
    """
    las = parse_las_file(las_path, alias_resolver=CurveAliasResolver())
    missing_curves = [c for c in ("GK", "KS", "PS") if c not in las.curves]

    depth = las.curves["DEPT"]
    fb = build_features(las.curves, las.step_actual, bundle.features_config)
    if fb.names != bundle.feature_names:
        raise PredictionError(
            f"{las_path}: engineered features don't match the checkpoint's feature schema.\n"
            f"    checkpoint expects: {bundle.feature_names}\n"
            f"    got from this file: {fb.names}"
        )

    center = np.asarray(bundle.normalization["center"], dtype=np.float32)
    scale = np.asarray(bundle.normalization["scale"], dtype=np.float32)
    normalized = (fb.matrix - center) / scale
    clip_value = bundle.normalization.get("clip_value")
    if clip_value is not None:
        normalized = np.clip(normalized, -clip_value, clip_value)

    with torch.no_grad():
        x = torch.from_numpy(normalized.astype(np.float32).copy()).transpose(0, 1).unsqueeze(0)
        out = bundle.model(x)
        if not (torch.isfinite(out["lithology_logits"]).all() and torch.isfinite(out["zone_logits"]).all()
                and torch.isfinite(out["boundary_logits"]).all()):
            raise PredictionError(
                f"{las_path}: the model produced NaN/Inf predictions on this file -- the "
                f"checkpoint is diverged/corrupted."
            )
        litho_pred = out["lithology_logits"].argmax(-1).squeeze(0).numpy()
        zone_pred = out["zone_logits"].argmax(-1).squeeze(0).numpy()
        litho_conf = torch.softmax(out["lithology_logits"], dim=-1).max(-1).values.squeeze(0).numpy()
        zone_conf = torch.softmax(out["zone_logits"], dim=-1).max(-1).values.squeeze(0).numpy()
        boundary_prob = torch.sigmoid(out["boundary_logits"]).squeeze(0).numpy()

    well_id = las.well_raw or Path(las_path).stem

    combined_df = reconstruct_well_intervals(
        well_id, depth, litho_pred, litho_conf, zone_pred, zone_conf,
        boundary_prob, bundle.id_to_lithology, bundle.id_to_zone,
    )
    litho_df = reconstruct_single_task_intervals(
        well_id, depth, litho_pred, litho_conf, bundle.id_to_lithology, "lithology"
    )
    zone_df = reconstruct_single_task_intervals(
        well_id, depth, zone_pred, zone_conf, bundle.id_to_zone, "zone"
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(out_path / "combined_intervals.csv", index=False)
    litho_df.to_csv(out_path / "lithology_intervals.csv", index=False)
    zone_df.to_csv(out_path / "zone_intervals.csv", index=False)

    # Plot the SAME smoothed labels the interval tables were built from
    # (majority-vote + short-run merge) so the picture never disagrees
    # with the reported intervals.
    smoothed_litho = merge_short_runs(majority_vote_smooth(litho_pred), min_run_points=3)
    smoothed_zone = merge_short_runs(majority_vote_smooth(zone_pred), min_run_points=3)
    predicted_lithology = np.array([bundle.id_to_lithology.get(int(i)) for i in smoothed_litho], dtype=object)
    predicted_zone = np.array([bundle.id_to_zone.get(int(i)) for i in smoothed_zone], dtype=object)
    plot_well_log(
        well_id, depth, las.curves,
        predicted_lithology=predicted_lithology, predicted_zone=predicted_zone,
        boundary_probability=boundary_prob, out_path=str(out_path / "well_log.png"),
    )

    return PredictionSummary(
        well_id=well_id, las_path=str(las_path), out_dir=str(out_path),
        n_points=len(depth), n_lithology_intervals=len(litho_df), n_zone_intervals=len(zone_df),
        parser_warnings=las.warnings, missing_curves=missing_curves,
    )
