#!/usr/bin/env python3
"""Phase 6-8 CLI: inspect the data, then export the reproducible on-disk
dataset (train/val/test Parquet + metadata) used by tools/train.py.

Refuses to export (non-zero exit, no files written) if the data-quality
report says the dataset is not valid for training -- see
DataQualityReport.is_valid_for_training().
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lithology.config import load_config
from lithology.dataset.export import export_dataset
from lithology.quality.report import build_data_quality_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path) if config_path.exists() else load_config(None)

    report = build_data_quality_report(config)
    print(report.to_text())

    valid, reasons = report.is_valid_for_training()
    if not report.availability.has_any_data:
        print("\nNo real dataset found -- nothing to export.")
        return 2
    if not valid:
        print("\nRefusing to export: dataset is not valid for training.")
        for r in reasons:
            print(f"  - {r}")
        return 1

    summary = export_dataset(config, report)
    print(f"\nExported {summary.n_wells_exported} wells to {summary.out_dir}")
    print(f"Split: {summary.split.as_dict()}")
    print(f"{len(summary.feature_names)} features, "
          f"{len(summary.label_maps.lithology)} lithology classes, "
          f"{len(summary.label_maps.zone)} zone classes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
