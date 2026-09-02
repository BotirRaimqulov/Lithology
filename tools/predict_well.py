#!/usr/bin/env python3
"""Run a trained checkpoint on a SINGLE, fresh .las file (not part of the
training/export dataset) and reconstruct lithology + stratigraphic-zone
intervals from the point-wise predictions.

This is the "give it one new LAS file and see what it predicts" script --
useful for sanity-checking a trained model, and for confirming the
lithology and zone predictions are genuinely independent of each other
(each gets its own interval table, in addition to the combined one).

For processing every unlabeled .las file in a directory at once, see
tools/predict_batch.py instead.

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lithology.inference import PredictionError, load_prediction_bundle, predict_las_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--las", required=True, help="Path to a single .las file to run inference on.")
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint_best.pt from tools/train.py.")
    parser.add_argument("--out-dir", default=None, help="Defaults to a folder next to the checkpoint.")
    args = parser.parse_args()

    try:
        bundle = load_prediction_bundle(args.checkpoint)
    except PredictionError as e:
        print(f"ERROR: {e}")
        return 2

    out_dir = args.out_dir or str(Path(args.checkpoint).parent / f"predict_{Path(args.las).stem}")
    try:
        summary = predict_las_file(args.las, bundle, out_dir)
    except PredictionError as e:
        print(f"ERROR: {e}")
        return 2

    if summary.parser_warnings:
        print(f"Parser notes for {args.las}:")
        for w in summary.parser_warnings:
            print(f"    - {w}")
    if summary.missing_curves:
        print(f"WARNING: this LAS file is missing curve(s) {summary.missing_curves}; "
              f"predictions may be unreliable.")

    print(f"\nWell: {summary.well_id}  |  {summary.n_points} depth points  |  "
          f"{summary.n_lithology_intervals} lithology intervals  |  {summary.n_zone_intervals} zone intervals")
    print(f"\nWrote combined_intervals.csv, lithology_intervals.csv, zone_intervals.csv, "
          f"well_log.png to {summary.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
