#!/usr/bin/env python3
"""Run a trained checkpoint on a SINGLE, fresh .las file (not part of the
training/export dataset) and reconstruct lithology + stratigraphic-zone
intervals from the point-wise predictions.

This is the "give it one new LAS file and see what it predicts" script --
useful for sanity-checking a trained model, and for confirming the
lithology and zone predictions are genuinely independent of each other
(each gets its own interval table, in addition to the combined one).

Usage:
    python tools/predict_well.py --las path/to/new_well.las \\
        --checkpoint outputs/experiments/<run>/checkpoint_best.pt \\
        --out-dir outputs/predictions/<well>

Writes, under --out-dir:
    combined_intervals.csv    Well/Top/Bottom/Thickness/Lithology/Zone/... (spec section 13 format)
    lithology_intervals.csv   lithology-only segmentation (its own boundaries)
    zone_intervals.csv        zone-only segmentation (its own boundaries)
    well_log.png              GK/KS/PS + predicted lithology/zone/boundary plot
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lithology.config import AlignmentConfig, FeaturesConfig
from lithology.features.engineering import build_features
from lithology.io.curve_aliases import CurveAliasResolver
from lithology.io.las_parser import parse_las_file
from lithology.postprocess.interval_reconstruction import (
    majority_vote_smooth, merge_short_runs, reconstruct_single_task_intervals, reconstruct_well_intervals,
)
from lithology.viz.plot_log import plot_well_log

try:
    import torch
except ImportError as e:
    raise ImportError("PyTorch is required (pip install torch).") from e

from lithology.config import ModelConfig
from lithology.models.multitask import MultiTaskLithologyModel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--las", required=True, help="Path to a single .las file to run inference on.")
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint_best.pt from tools/train.py.")
    parser.add_argument("--out-dir", default=None, help="Defaults to a folder next to the checkpoint.")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    feature_names = ckpt["feature_names"]
    label_maps = ckpt["label_maps"]
    id_to_litho = {v: k for k, v in label_maps["lithology"].items()}
    id_to_zone = {v: k for k, v in label_maps["zone"].items()}

    normalization = ckpt.get("normalization")
    if normalization is None:
        print("ERROR: this checkpoint has no embedded normalization stats (it was trained before "
              "that fix, or trained in cross-validation mode). Retrain with the current tools/train.py "
              "to get a checkpoint usable on a fresh LAS file.")
        return 2

    features_config_dict = ckpt.get("features_config")
    features_config = FeaturesConfig(**features_config_dict) if features_config_dict else FeaturesConfig()

    # --- parse the new LAS file -------------------------------------------
    las = parse_las_file(args.las, alias_resolver=CurveAliasResolver())
    if las.warnings:
        print(f"Parser notes for {args.las}:")
        for w in las.warnings:
            print(f"    - {w}")
    missing_curves = [c for c in ("GK", "KS", "PS") if c not in las.curves]
    if missing_curves:
        print(f"WARNING: this LAS file is missing curve(s) {missing_curves}; predictions may be unreliable.")

    depth = las.curves["DEPT"]
    fb = build_features(las.curves, las.step_actual, features_config)
    if fb.names != feature_names:
        print("ERROR: this LAS file's engineered features don't match the checkpoint's feature schema "
              "(likely a mismatched features_config). Cannot run inference safely.")
        print(f"    checkpoint expects: {feature_names}")
        print(f"    got from this file: {fb.names}")
        return 2

    center = np.asarray(normalization["center"], dtype=np.float32)
    scale = np.asarray(normalization["scale"], dtype=np.float32)
    normalized = (fb.matrix - center) / scale
    clip_value = normalization.get("clip_value")
    if clip_value is not None:
        normalized = np.clip(normalized, -clip_value, clip_value)

    # --- run the model -------------------------------------------------------
    model = MultiTaskLithologyModel(in_features=len(feature_names), config=ModelConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        x = torch.from_numpy(normalized.astype(np.float32).copy()).transpose(0, 1).unsqueeze(0)
        out = model(x)
        if not (torch.isfinite(out["lithology_logits"]).all() and torch.isfinite(out["zone_logits"]).all()
                and torch.isfinite(out["boundary_logits"]).all()):
            print("ERROR: the model produced NaN/Inf predictions on this file -- this checkpoint is "
                  "diverged/corrupted. Retrain (see training.grad_clip_norm / normalization.clip_value).")
            return 2
        litho_pred = out["lithology_logits"].argmax(-1).squeeze(0).numpy()
        zone_pred = out["zone_logits"].argmax(-1).squeeze(0).numpy()
        litho_conf = torch.softmax(out["lithology_logits"], dim=-1).max(-1).values.squeeze(0).numpy()
        zone_conf = torch.softmax(out["zone_logits"], dim=-1).max(-1).values.squeeze(0).numpy()
        boundary_prob = torch.sigmoid(out["boundary_logits"]).squeeze(0).numpy()

    # --- reconstruct intervals -------------------------------------------
    combined_df = reconstruct_well_intervals(
        las.well_raw or Path(args.las).stem, depth, litho_pred, litho_conf, zone_pred, zone_conf,
        boundary_prob, id_to_litho, id_to_zone,
    )
    litho_df = reconstruct_single_task_intervals(
        las.well_raw or Path(args.las).stem, depth, litho_pred, litho_conf, id_to_litho, "lithology"
    )
    zone_df = reconstruct_single_task_intervals(
        las.well_raw or Path(args.las).stem, depth, zone_pred, zone_conf, id_to_zone, "zone"
    )

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).parent / f"predict_{Path(args.las).stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(out_dir / "combined_intervals.csv", index=False)
    litho_df.to_csv(out_dir / "lithology_intervals.csv", index=False)
    zone_df.to_csv(out_dir / "zone_intervals.csv", index=False)

    # Plot the SAME smoothed labels the interval tables were built from
    # (majority-vote + short-run merge) -- plotting the raw point-wise
    # argmax instead would show single-sample flicker the CSVs already
    # cleaned up, making the picture disagree with the reported intervals.
    smoothed_litho = merge_short_runs(majority_vote_smooth(litho_pred), min_run_points=3)
    smoothed_zone = merge_short_runs(majority_vote_smooth(zone_pred), min_run_points=3)
    predicted_lithology = np.array([id_to_litho.get(int(i)) for i in smoothed_litho], dtype=object)
    predicted_zone = np.array([id_to_zone.get(int(i)) for i in smoothed_zone], dtype=object)
    plot_well_log(
        las.well_raw or Path(args.las).stem, depth, las.curves,
        predicted_lithology=predicted_lithology, predicted_zone=predicted_zone,
        boundary_probability=boundary_prob, out_path=str(out_dir / "well_log.png"),
    )

    print(f"\nWell: {las.well_raw}  |  {len(depth)} depth points  |  "
          f"{len(litho_df)} lithology intervals  |  {len(zone_df)} zone intervals")
    print(f"\nLithology intervals (independent segmentation):\n{litho_df.to_string(index=False)}")
    print(f"\nZone intervals (independent segmentation):\n{zone_df.to_string(index=False)}")
    print(f"\nWrote combined_intervals.csv, lithology_intervals.csv, zone_intervals.csv, "
          f"well_log.png to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
