#!/usr/bin/env python3
"""Phase 12 CLI: train the multi-task model on an already-exported dataset.

Run tools/build_dataset.py first. This script never touches raw LAS/CSV
files -- it only reads the Parquet/NPZ dataset and metadata produced by
the export step, so training is fully reproducible from that artifact.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lithology.config import load_config
from lithology.training.train import train


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset-dir", default=None,
                        help="Defaults to <data.output_dir>/<dataset.out_dir>")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else Path(config.data.output_dir) / config.dataset.out_dir
    if not (dataset_dir / "metadata" / "label_maps.json").exists():
        print(f"No exported dataset found at {dataset_dir}. Run tools/build_dataset.py first.")
        return 2

    result = train(config, dataset_dir, run_name=args.run_name)
    print(f"Training complete. Best epoch {result.best_epoch}, "
          f"best val lithology macro-F1 {result.best_val_metric:.4f}")
    print(f"Run artifacts: {result.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
