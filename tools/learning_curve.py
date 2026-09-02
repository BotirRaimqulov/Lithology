#!/usr/bin/env python3
"""Empirically answer "how many wells do we need?" by training on
progressively larger subsets of the available training wells (val/test
wells held FIXED across every run for a fair comparison) and plotting
macro-F1 vs. training-well count for both lithology and zone.

If the curve is still rising steeply at your current well count, more
wells will likely help a lot. If it has flattened, you are near the
ceiling for the current model/feature setup, and further gains need to
come from data quality (more consistent labels, better per-class well
coverage) rather than raw well count -- see the per-class support counts
in tools/evaluate.py's output for which classes are actually starved.

Usage:
    python tools/learning_curve.py --config configs/default.yaml \\
        --fractions 0.2,0.4,0.6,0.8,1.0 --epochs 20

Writes, under --out-dir (default outputs/learning_curve):
    learning_curve.json    raw per-fraction results
    learning_curve.png     macro-F1 vs. #train wells, one line per task/split
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lithology.config import load_config
from lithology.dataset.export import export_dataset
from lithology.dataset.split import SplitError, split_wells
from lithology.inference import load_prediction_bundle
from lithology.quality.report import build_data_quality_report
from lithology.training.train import evaluate, load_well_arrays, resolve_device, train


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--fractions", default="0.2,0.4,0.6,0.8,1.0",
                        help="Comma-separated fractions of the available training wells to try.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training.epochs for each run (fewer = faster diagnostic; "
                             "default keeps the config's value).")
    parser.add_argument("--out-dir", default="outputs/learning_curve")
    args = parser.parse_args()

    base_config = load_config(args.config)
    report = build_data_quality_report(base_config)
    valid, reasons = report.is_valid_for_training()
    if not valid:
        print("Cannot run a learning curve: dataset is not valid for training.")
        for r in reasons:
            print(f"  - {r}")
        return 1

    all_wells = list(report.aligned_wells.keys())
    try:
        base_split = split_wells(all_wells, base_config.split)
    except SplitError as e:
        print(f"ERROR: {e}")
        return 1
    if base_split.mode != "holdout" and base_split.mode != "explicit":
        print(f"A learning curve needs a fixed holdout val/test set, but the split strategy resolved to "
              f"'{base_split.mode}' (too few wells for split.min_wells_for_holdout_split={base_config.split.min_wells_for_holdout_split}). "
              f"Lower that threshold or add more labeled wells first.")
        return 1

    train_wells_full = list(base_split.train)
    val_wells, test_wells = list(base_split.val), list(base_split.test)
    if len(train_wells_full) < 2:
        print("Not enough training wells to build a meaningful learning curve (need at least 2).")
        return 1

    fractions = sorted({float(f) for f in args.fractions.split(",")})
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for frac in fractions:
        n = max(1, round(frac * len(train_wells_full)))
        subset_train = train_wells_full[:n]

        sub_config = copy.deepcopy(base_config)
        sub_config.split.train_wells = subset_train
        sub_config.split.val_wells = val_wells
        sub_config.split.test_wells = test_wells
        if args.epochs is not None:
            sub_config.training.epochs = args.epochs
        run_dir = out_dir / f"n{n}"
        sub_config.data.output_dir = str(run_dir)

        print(f"\n=== {n}/{len(train_wells_full)} training wells (fraction {frac}) ===")
        # Restrict the report to exactly this run's wells first -- otherwise
        # split_wells()'s "never silently drop a well" safety net would
        # treat every OTHER training well as an accidental omission and
        # quietly add it back in, defeating the whole point of the subset.
        run_report = report.restricted_to(subset_train + val_wells + test_wells)
        export_dataset(sub_config, run_report)
        dataset_dir = run_dir / sub_config.dataset.out_dir
        train_result = train(sub_config, dataset_dir, run_name="run")

        if train_result.best_epoch == -1:
            print(f"  WARNING: no valid checkpoint produced for n={n}; skipping evaluation for this point.")
            results.append({"n_train_wells": n, "fraction": frac, "status": "failed_to_converge"})
            continue

        bundle = load_prediction_bundle(str(Path(train_result.run_dir) / "checkpoint_best.pt"))
        num_lithology = len(bundle.label_maps["lithology"])
        num_zone = len(bundle.label_maps["zone"])
        device = resolve_device(sub_config.training.device)
        bundle.model.to(device)

        val_arrays = load_well_arrays(dataset_dir / "val", bundle.feature_names, sub_config.dataset.format)
        test_arrays = load_well_arrays(dataset_dir / "test", bundle.feature_names, sub_config.dataset.format)
        val_metrics = evaluate(bundle.model, val_arrays, device, num_lithology, num_zone)
        test_metrics = evaluate(bundle.model, test_arrays, device, num_lithology, num_zone) if test_arrays else None

        row = {
            "n_train_wells": n, "fraction": frac, "status": "ok",
            "val_lithology_macro_f1": val_metrics["lithology"].get("macro_f1"),
            "val_zone_macro_f1": val_metrics["zone"].get("macro_f1"),
            "test_lithology_macro_f1": test_metrics["lithology"].get("macro_f1") if test_metrics else None,
            "test_zone_macro_f1": test_metrics["zone"].get("macro_f1") if test_metrics else None,
        }
        results.append(row)
        print(f"  val lithology macro-F1={row['val_lithology_macro_f1']:.3f}  "
              f"val zone macro-F1={row['val_zone_macro_f1']:.3f}")

    (out_dir / "learning_curve.json").write_text(json.dumps(results, indent=2))

    ok_results = [r for r in results if r["status"] == "ok"]
    if ok_results:
        fig, ax = plt.subplots(figsize=(8, 5))
        ns = [r["n_train_wells"] for r in ok_results]
        for key, label, style in (
            ("val_lithology_macro_f1", "Lithology (val)", "-o"),
            ("val_zone_macro_f1", "Zone (val)", "-s"),
            ("test_lithology_macro_f1", "Lithology (test)", "--o"),
            ("test_zone_macro_f1", "Zone (test)", "--s"),
        ):
            ys = [r[key] for r in ok_results]
            if all(y is not None for y in ys):
                ax.plot(ns, ys, style, label=label)
        ax.set_xlabel("Number of training wells")
        ax.set_ylabel("Macro F1")
        ax.set_title("Learning curve: does more data help?")
        ax.legend()
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(out_dir / "learning_curve.png", dpi=150)
        print(f"\nWrote learning_curve.json and learning_curve.png to {out_dir}")
    else:
        print("\nNo run converged to a usable checkpoint -- no plot produced.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
