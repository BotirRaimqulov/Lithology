#!/usr/bin/env python3
"""Phase 13-14 CLI: evaluate a trained checkpoint on a dataset split and
reconstruct continuous geological intervals from the point-wise predictions.

Writes, under <run_dir>/eval_<split>/:
  * metrics.json           -- classification + boundary metrics (spec 21)
  * intervals_<well>.csv   -- reconstructed Well/Top/Bottom/.../Confidence table (spec 13)
  * interval_quality.json  -- boundary depth error (m) + interval IoU vs. expert labels
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lithology.constants import IGNORE_INDEX
from lithology.postprocess.interval_reconstruction import (
    boundary_depth_errors, interval_iou, reconstruct_well_intervals,
)
from lithology.training.metrics import boundary_metrics, classification_metrics
from lithology.training.train import load_well_arrays

try:
    import torch
except ImportError as e:
    raise ImportError("PyTorch is required for evaluation.") from e

from lithology.models.multitask import MultiTaskLithologyModel
from lithology.config import ModelConfig


def _true_intervals_from_labels(depth: np.ndarray, label_ids: np.ndarray, id_to_name: dict) -> list:
    intervals = []
    n = len(label_ids)
    if n == 0:
        return intervals
    start = 0
    for i in range(1, n + 1):
        if i == n or label_ids[i] != label_ids[start]:
            if label_ids[start] != IGNORE_INDEX:
                intervals.append((float(depth[start]), float(depth[min(i, n - 1)]), id_to_name.get(int(label_ids[start]))))
            start = i
    return intervals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--dataset-format", default="parquet", choices=["parquet", "npz"])
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    feature_names = ckpt["feature_names"]
    label_maps = ckpt["label_maps"]
    id_to_litho = {v: k for k, v in label_maps["lithology"].items()}
    id_to_zone = {v: k for k, v in label_maps["zone"].items()}

    model_config = ModelConfig(**ckpt["model_config"])
    model = MultiTaskLithologyModel(in_features=len(feature_names), config=model_config)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dataset_dir = Path(args.dataset_dir)
    wells = load_well_arrays(dataset_dir / args.split, feature_names, args.dataset_format)
    if not wells:
        print(f"No wells found in {dataset_dir / args.split}")
        return 2

    out_dir = Path(args.checkpoint).parent / f"eval_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_litho_true, all_litho_pred, all_zone_true, all_zone_pred = [], [], [], []
    all_boundary_true, all_boundary_prob = [], []
    boundary_errors_all, iou_all = [], []

    with torch.no_grad():
        for w in wells:
            x = torch.from_numpy(w.features.astype(np.float32)).transpose(0, 1).unsqueeze(0)
            out = model(x)
            litho_pred = out["lithology_logits"].argmax(-1).squeeze(0).numpy()
            zone_pred = out["zone_logits"].argmax(-1).squeeze(0).numpy()
            boundary_prob = torch.sigmoid(out["boundary_logits"]).squeeze(0).numpy()

            all_litho_true.append(w.lithology_label); all_litho_pred.append(litho_pred)
            all_zone_true.append(w.zone_label); all_zone_pred.append(zone_pred)
            all_boundary_true.append(w.boundary_label); all_boundary_prob.append(boundary_prob)

            conf = np.ones(len(w.depth), dtype=float)
            intervals_df = reconstruct_well_intervals(
                w.well_id, w.depth, litho_pred, conf, zone_pred, conf, boundary_prob,
                id_to_litho, id_to_zone,
            )
            intervals_df.to_csv(out_dir / f"intervals_{w.well_id}.csv", index=False)

            true_litho_intervals = _true_intervals_from_labels(w.depth, w.lithology_label, id_to_litho)
            pred_litho_intervals = list(zip(intervals_df["top"], intervals_df["bottom"], intervals_df["lithology"]))
            iou_all.append(interval_iou(true_litho_intervals, pred_litho_intervals))

            true_boundaries = [d for d, b in zip(w.depth, w.boundary_label) if b == 1]
            pred_boundaries = [d for d, p in zip(w.depth, boundary_prob) if p >= 0.5]
            step = w.step or 0.1
            boundary_errors_all.append(boundary_depth_errors(true_boundaries, pred_boundaries,
                                                              max_match_distance_m=step * 5))

    num_lithology = len(label_maps["lithology"])
    num_zone = len(label_maps["zone"])
    metrics = {
        "lithology": classification_metrics(np.concatenate(all_litho_true), np.concatenate(all_litho_pred), num_lithology),
        "zone": classification_metrics(np.concatenate(all_zone_true), np.concatenate(all_zone_pred), num_zone),
        "boundary": boundary_metrics(np.concatenate(all_boundary_true), np.concatenate(all_boundary_prob)),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "interval_quality.json").write_text(json.dumps(
        {"boundary_errors_per_well": boundary_errors_all, "lithology_iou_per_well": iou_all}, indent=2))

    print(json.dumps(metrics, indent=2))
    print(f"\nWrote per-well interval tables and interval_quality.json to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
