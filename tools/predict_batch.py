#!/usr/bin/env python3
"""Run a trained checkpoint on every .las file that was NOT part of the
training/export dataset (no matching lithology/stratigraphy labels), or
on every .las file in a directory with --all.

This answers "we have LAS files with no expert interpretation yet -- show
me what the model thinks for all of them" without running
tools/predict_well.py once per file by hand.

Usage:
    # only wells with no CSV label match (per configs/default.yaml paths)
    python tools/predict_batch.py --checkpoint outputs/experiments/<run>/checkpoint_best.pt

    # every .las file in a given directory, labeled or not
    python tools/predict_batch.py --checkpoint outputs/experiments/<run>/checkpoint_best.pt \\
        --las-dir some/other/folder --all

Writes, under --out-dir/<well_id>/ for each well:
    combined_intervals.csv, lithology_intervals.csv, zone_intervals.csv, well_log.png
plus a top-level batch_summary.json listing what succeeded/failed and why.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lithology.config import load_config
from lithology.inference import PredictionError, load_prediction_bundle, predict_las_file
from lithology.io.las_parser import find_las_files
from lithology.quality.report import build_data_quality_report

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint_best.pt from tools/train.py.")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Used to find las_dir/lithology_csv/stratigraphy_csv (to determine which wells "
                             "already have labels) unless --all is given.")
    parser.add_argument("--las-dir", default=None, help="Override the LAS directory from --config.")
    parser.add_argument("--all", action="store_true",
                        help="Process every .las file found, including ones already in the training dataset.")
    parser.add_argument("--out-dir", default=None, help="Defaults to a folder next to the checkpoint.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path) if config_path.exists() else load_config(None)
    if args.las_dir:
        config.data.las_dir = args.las_dir

    try:
        bundle = load_prediction_bundle(args.checkpoint)
    except PredictionError as e:
        print(f"ERROR: {e}")
        return 2

    if args.all:
        las_paths = find_las_files(config.data.las_dir)
        if not las_paths:
            print(f"No .las files found in {config.data.las_dir}")
            return 2
        targets = [(p.stem, p) for p in las_paths]
    else:
        report = build_data_quality_report(config)
        if not report.availability.has_any_data:
            print(f"No LAS files found in {config.data.las_dir}")
            return 2
        unlabeled_raw_ids = report.well_match.las_without_labels if report.well_match else []
        if not unlabeled_raw_ids:
            print("Every LAS well already has matching expert labels (none are 'unlabeled'). "
                  "Use --all to process every well anyway, or tools/predict_well.py for one file.")
            return 0
        targets = [(w, Path(report.las_files_parsed[w].path)) for w in unlabeled_raw_ids]

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).parent / "predict_batch"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for well_id, las_path in tqdm(targets, desc="Predicting", unit="well"):
        well_out_dir = out_dir / well_id
        try:
            summary = predict_las_file(str(las_path), bundle, str(well_out_dir))
            results.append({"well_id": well_id, "las_path": str(las_path), "status": "ok",
                            "n_points": summary.n_points, "n_lithology_intervals": summary.n_lithology_intervals,
                            "n_zone_intervals": summary.n_zone_intervals,
                            "missing_curves": summary.missing_curves})
        except PredictionError as e:
            results.append({"well_id": well_id, "las_path": str(las_path), "status": "error", "error": str(e)})

    (out_dir / "batch_summary.json").write_text(json.dumps(results, indent=2))

    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_err = len(results) - n_ok
    print(f"\nProcessed {len(results)} well(s): {n_ok} OK, {n_err} failed.")
    if n_err:
        print("Failed wells:")
        for r in results:
            if r["status"] == "error":
                print(f"    - {r['well_id']}: {r['error']}")
    print(f"\nPer-well outputs (interval CSVs + well_log.png) and batch_summary.json written to {out_dir}")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
