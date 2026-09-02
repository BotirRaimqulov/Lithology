#!/usr/bin/env python3
"""Phase-1/6 inspection CLI.

Run this after placing real LAS files and CSVs in the directories named by
the config (default: ``data/las``, ``data/csv/lithology.csv``,
``data/csv/stratigraphy.csv``). It NEVER fabricates statistics: if the
configured directories are empty it says so explicitly instead of printing
zeros that could be mistaken for "checked, found nothing wrong".

Usage:
    python tools/inspect_dataset.py
    python tools/inspect_dataset.py --config configs/default.yaml
    python tools/inspect_dataset.py --las-dir data/las \\
        --lithology-csv data/csv/lithology.csv \\
        --stratigraphy-csv data/csv/stratigraphy.csv
    python tools/inspect_dataset.py --save-report outputs/data_quality_report.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lithology.config import load_config
from lithology.quality.report import build_data_quality_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--las-dir", default=None, help="Override data.las_dir")
    parser.add_argument("--lithology-csv", default=None, help="Override data.lithology_csv")
    parser.add_argument("--stratigraphy-csv", default=None, help="Override data.stratigraphy_csv")
    parser.add_argument("--save-report", default=None, help="Also write the report text to this path.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path) if config_path.exists() else load_config(None)
    if not config_path.exists():
        print(f"(note: config file {config_path} not found, using built-in defaults)\n")

    if args.las_dir:
        config.data.las_dir = args.las_dir
    if args.lithology_csv:
        config.data.lithology_csv = args.lithology_csv
    if args.stratigraphy_csv:
        config.data.stratigraphy_csv = args.stratigraphy_csv

    report = build_data_quality_report(config)
    text = report.to_text()
    print(text)

    if args.save_report:
        out_path = Path(args.save_report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)
        print(f"\nReport written to {out_path}")

    valid, _ = report.is_valid_for_training()
    if not report.availability.has_any_data:
        return 2
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
