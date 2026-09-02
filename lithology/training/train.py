"""Training/validation loop, driven entirely by :class:`~lithology.config.Config`.

Loads the already-exported dataset (see ``dataset/export.py`` /
``tools/build_dataset.py``) -- this module does NOT parse LAS/CSV files
itself. Every run writes its full resolved config, per-epoch metrics, and
the best checkpoint under ``<output_dir>/experiments/<run_name>/`` so a
result can always be traced back to exactly how it was produced (spec
section 20).
"""
from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from lithology.config import Config
from lithology.constants import IGNORE_INDEX
from lithology.dataset.windowing import CropDataset, WellArrays, min_crop_points_from_context
from lithology.training.metrics import boundary_metrics, classification_metrics

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError as e:  # pragma: no cover
    raise ImportError("PyTorch is required for training (pip install torch).") from e

from lithology.models.losses import MultiTaskLoss
from lithology.models.multitask import MultiTaskLithologyModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> "torch.device":
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_well_arrays(split_dir: Path, feature_names: list, fmt: str) -> list:
    wells = []
    ext = ".parquet" if fmt == "parquet" else ".npz"
    for path in sorted(split_dir.glob(f"*{ext}")):
        if fmt == "parquet":
            df = pd.read_parquet(path)
        else:
            npz = np.load(path, allow_pickle=True)
            df = pd.DataFrame({k: npz[k] for k in npz.files})
        wells.append(
            WellArrays(
                well_id=str(df["well_id"].iloc[0]),
                depth=df["depth"].to_numpy(dtype=np.float64),
                features=df[feature_names].to_numpy(dtype=np.float32),
                lithology_label=df["lithology_label_id"].to_numpy(dtype=np.int64),
                zone_label=df["zone_label_id"].to_numpy(dtype=np.int64),
                boundary_label=df["boundary_label"].to_numpy(dtype=np.int64),
                step=float(np.median(np.diff(df["depth"].to_numpy()))) if len(df) > 1 else None,
            )
        )
    return wells


def class_balanced_weights(well_arrays: list, label_attr: str, num_classes: int) -> "torch.Tensor":
    counts = np.zeros(num_classes, dtype=np.float64)
    for w in well_arrays:
        labels = getattr(w, label_attr)
        valid = labels[labels != IGNORE_INDEX]
        counts += np.bincount(valid, minlength=num_classes)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


@dataclass
class TrainResult:
    run_dir: str
    best_val_metric: float
    best_epoch: int
    history: list


def evaluate(model, wells: list, device, num_lithology: int, num_zone: int) -> dict:
    model.eval()
    all_litho_true, all_litho_pred = [], []
    all_zone_true, all_zone_pred = [], []
    all_boundary_true, all_boundary_prob = [], []
    with torch.no_grad():
        for w in wells:
            x = torch.from_numpy(w.features.astype(np.float32)).transpose(0, 1).unsqueeze(0).to(device)
            out = model(x)
            litho_pred = out["lithology_logits"].argmax(-1).squeeze(0).cpu().numpy()
            zone_pred = out["zone_logits"].argmax(-1).squeeze(0).cpu().numpy()
            boundary_prob = torch.sigmoid(out["boundary_logits"]).squeeze(0).cpu().numpy()

            all_litho_true.append(w.lithology_label)
            all_litho_pred.append(litho_pred)
            all_zone_true.append(w.zone_label)
            all_zone_pred.append(zone_pred)
            all_boundary_true.append(w.boundary_label)
            all_boundary_prob.append(boundary_prob)

    litho_metrics = classification_metrics(
        np.concatenate(all_litho_true), np.concatenate(all_litho_pred), num_lithology
    )
    zone_metrics = classification_metrics(
        np.concatenate(all_zone_true), np.concatenate(all_zone_pred), num_zone
    )
    boundary_m = boundary_metrics(np.concatenate(all_boundary_true), np.concatenate(all_boundary_prob))
    return {"lithology": litho_metrics, "zone": zone_metrics, "boundary": boundary_m}


def train(config: Config, dataset_dir: Path, run_name: Optional[str] = None) -> TrainResult:
    set_seed(config.training.seed)
    device = resolve_device(config.training.device)

    meta_dir = dataset_dir / "metadata"
    label_maps = json.loads((meta_dir / "label_maps.json").read_text())
    feature_names = json.loads((meta_dir / "feature_names.json").read_text())
    num_lithology = len(label_maps["lithology"])
    num_zone = len(label_maps["zone"])

    fmt = config.dataset.format
    train_wells = load_well_arrays(dataset_dir / "train", feature_names, fmt)
    val_wells = load_well_arrays(dataset_dir / "val", feature_names, fmt)
    if not train_wells:
        raise ValueError(f"No training wells found under {dataset_dir / 'train'}")

    step = train_wells[0].step or 0.1
    min_crop = min_crop_points_from_context(config.sampling.context_unit, config.sampling.context_size, step)
    train_dataset = CropDataset(train_wells, min_crop_points=min_crop)
    loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True,
                         num_workers=config.training.num_workers)

    model_config = copy.deepcopy(config.model)
    model_config.num_classes_lithology = num_lithology
    model_config.num_classes_zone = num_zone
    model = MultiTaskLithologyModel(in_features=len(feature_names), config=model_config).to(device)

    litho_w = class_balanced_weights(train_wells, "lithology_label", num_lithology).to(device) \
        if config.training.class_balanced_loss else None
    zone_w = class_balanced_weights(train_wells, "zone_label", num_zone).to(device) \
        if config.training.class_balanced_loss else None
    loss_fn = MultiTaskLoss(
        weight_lithology=config.training.weight_lithology,
        weight_zone=config.training.weight_zone,
        weight_boundary=config.training.weight_boundary,
        lithology_class_weights=litho_w,
        zone_class_weights=zone_w,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate,
                                  weight_decay=config.training.weight_decay)

    run_name = run_name or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(config.data.output_dir) / "experiments" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config.to_yaml(run_dir / "config_used.yaml")

    history = []
    best_val_metric = -1.0
    best_epoch = -1
    patience_left = config.training.early_stopping_patience

    for epoch in range(config.training.epochs):
        model.train()
        epoch_losses = []
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(batch["features"])
            loss, parts = loss_fn(out, batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(parts)

        avg_loss = {k: float(np.mean([p[k] for p in epoch_losses])) for k in epoch_losses[0]} if epoch_losses else {}
        val_metrics = evaluate(model, val_wells, device, num_lithology, num_zone) if val_wells else {}
        val_score = val_metrics.get("lithology", {}).get("macro_f1", 0.0) if val_wells else avg_loss.get("loss_total", 0.0) * -1

        record = {"epoch": epoch, "train_loss": avg_loss, "val_metrics": val_metrics}
        history.append(record)
        (run_dir / "training_log.json").write_text(json.dumps(history, indent=2))

        improved = val_score > best_val_metric
        if improved:
            best_val_metric = val_score
            best_epoch = epoch
            patience_left = config.training.early_stopping_patience
            torch.save(
                {"model_state": model.state_dict(), "model_config": model_config.__dict__,
                 "feature_names": feature_names, "label_maps": label_maps, "epoch": epoch},
                run_dir / "checkpoint_best.pt",
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    return TrainResult(run_dir=str(run_dir), best_val_metric=best_val_metric, best_epoch=best_epoch,
                        history=history)
