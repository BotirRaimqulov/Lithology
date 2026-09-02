#!/usr/bin/env python3
"""Phase 7/22 CLI: plot GK/KS/PS + expert (and, optionally, predicted)
lithology/zone/boundary tracks for one well from the exported dataset.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lithology.viz.plot_log import plot_well_log


def _find_well_file(dataset_dir: Path, well_id: str) -> Path:
    for split in ("train", "val", "test", "cv_wells"):
        for ext in (".parquet", ".npz"):
            p = dataset_dir / split / f"{well_id}{ext}"
            if p.exists():
                return p
    raise FileNotFoundError(f"Well {well_id} not found under {dataset_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--well", required=True)
    parser.add_argument("--checkpoint", default=None, help="Optional: overlay model predictions.")
    parser.add_argument("--out", default=None, help="Output PNG path.")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    path = _find_well_file(dataset_dir, args.well)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.DataFrame(dict(np.load(path, allow_pickle=True)))

    depth = df["depth"].to_numpy()
    curves = {c: df[f"{c}_measured"].to_numpy() for c in ("GK", "KS", "PS")}

    predicted_lithology = predicted_zone = boundary_prob = None
    if args.checkpoint:
        import torch
        from lithology.config import ModelConfig
        from lithology.models.multitask import MultiTaskLithologyModel

        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        feature_names = ckpt["feature_names"]
        label_maps = ckpt["label_maps"]
        id_to_litho = {v: k for k, v in label_maps["lithology"].items()}
        id_to_zone = {v: k for k, v in label_maps["zone"].items()}
        model = MultiTaskLithologyModel(in_features=len(feature_names), config=ModelConfig(**ckpt["model_config"]))
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        with torch.no_grad():
            x = torch.from_numpy(df[feature_names].to_numpy(dtype=np.float32).copy()).transpose(0, 1).unsqueeze(0)
            out = model(x)
            predicted_lithology = np.array([id_to_litho.get(int(i)) for i in out["lithology_logits"].argmax(-1).squeeze(0).numpy()])
            predicted_zone = np.array([id_to_zone.get(int(i)) for i in out["zone_logits"].argmax(-1).squeeze(0).numpy()])
            boundary_prob = torch.sigmoid(out["boundary_logits"]).squeeze(0).numpy()

    out_path = args.out or f"well_{args.well}_log.png"
    plot_well_log(
        args.well, depth, curves,
        expert_lithology=df["lithology_code"].to_numpy(dtype=object),
        expert_zone=df["zone_name"].to_numpy(dtype=object),
        predicted_lithology=predicted_lithology,
        predicted_zone=predicted_zone,
        boundary_probability=boundary_prob,
        out_path=out_path,
    )
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
