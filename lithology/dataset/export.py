"""Build the reproducible on-disk dataset: label encoding, train-only
normalization, per-well Parquet files under ``dataset/{train,val,test}``
(spec section 19), plus a ``metadata/`` folder recording everything needed
to reproduce the run (spec section 20): the exact split, label vocabulary,
normalization statistics, feature names, and the full resolved config.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from lithology.alignment.depth_alignment import AlignedWell
from lithology.config import Config
from lithology.constants import IGNORE_INDEX
from lithology.dataset.split import SplitResult, split_wells
from lithology.features.engineering import RAW_CURVES, build_features
from lithology.io.las_parser import LASFile
from lithology.quality.report import DataQualityReport


@dataclass
class LabelMaps:
    lithology: dict
    zone: dict

    def to_json(self) -> dict:
        return {"lithology": self.lithology, "zone": self.zone}


def build_label_maps(report: DataQualityReport) -> LabelMaps:
    litho_codes = sorted(report.lithology_class_distribution.keys())
    zones = sorted(report.zone_distribution.keys())
    return LabelMaps(
        lithology={code: i for i, code in enumerate(litho_codes)},
        zone={z: i for i, z in enumerate(zones)},
    )


def _encode(labels: np.ndarray, mapping: dict) -> np.ndarray:
    return np.array(
        [mapping[v] if v is not None and v in mapping else IGNORE_INDEX for v in labels], dtype=np.int64
    )


@dataclass
class NormalizationStats:
    method: str
    feature_names: list
    center: list
    scale: list

    def apply(self, matrix: np.ndarray) -> np.ndarray:
        center = np.asarray(self.center, dtype=np.float32)
        scale = np.asarray(self.scale, dtype=np.float32)
        return (matrix - center) / scale

    def to_json(self) -> dict:
        return {"method": self.method, "feature_names": self.feature_names,
                "center": self.center, "scale": self.scale}


def fit_normalization(feature_matrices: list, feature_names: list, method: str = "robust") -> NormalizationStats:
    stacked = np.concatenate(feature_matrices, axis=0) if feature_matrices else np.zeros((0, len(feature_names)))
    if method == "robust":
        center = np.median(stacked, axis=0)
        q75, q25 = np.percentile(stacked, [75, 25], axis=0)
        scale = q75 - q25
    elif method == "standard":
        center = stacked.mean(axis=0)
        scale = stacked.std(axis=0)
    else:
        raise ValueError(f"Unknown normalization method {method!r}")
    scale = np.where(scale < 1e-8, 1.0, scale)
    return NormalizationStats(method=method, feature_names=list(feature_names),
                               center=center.tolist(), scale=scale.tolist())


@dataclass
class WellFrame:
    well_id: str
    df: pd.DataFrame
    feature_names: list


def _build_well_frame(well_id: str, aligned: AlignedWell, las: LASFile, config: Config) -> WellFrame:
    fb = build_features(las.curves, las.step_actual, config.features)
    df = pd.DataFrame({"well_id": well_id, "depth": aligned.depth})
    for c in RAW_CURVES:
        df[f"{c}_measured"] = las.curves.get(c, np.full(len(aligned.depth), np.nan))
    for i, name in enumerate(fb.names):
        df[name] = fb.matrix[:, i]
    df["zone_name"] = aligned.zone_label
    df["lithology_code"] = aligned.lithology_label
    df["lithology_code_raw"] = aligned.lithology_label_raw
    df["lithology_source"] = aligned.lithology_source
    df["lithology_core_verified"] = aligned.lithology_core_verified
    df["lithology_confidence"] = aligned.lithology_confidence
    df["boundary_label"] = aligned.boundary_label
    return WellFrame(well_id=well_id, df=df, feature_names=fb.names)


@dataclass
class ExportSummary:
    out_dir: str
    split: SplitResult
    label_maps: LabelMaps
    feature_names: list
    n_wells_exported: int
    files_written: list = field(default_factory=list)


def _write_well(df: pd.DataFrame, label_maps: LabelMaps, norm: Optional[NormalizationStats],
                 feature_names: list, out_path: Path, fmt: str) -> None:
    out = df.copy()
    out["zone_label_id"] = _encode(out["zone_name"].to_numpy(dtype=object), label_maps.zone)
    out["lithology_label_id"] = _encode(out["lithology_code"].to_numpy(dtype=object), label_maps.lithology)
    if norm is not None:
        out.loc[:, feature_names] = norm.apply(out[feature_names].to_numpy(dtype=np.float32))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        out.to_parquet(out_path.with_suffix(".parquet"), index=False)
    elif fmt == "npz":
        arrays = {col: out[col].to_numpy() for col in out.columns}
        np.savez_compressed(out_path.with_suffix(".npz"), **arrays)
    else:
        raise ValueError(f"Unknown dataset format {fmt!r}")


def export_dataset(config: Config, report: DataQualityReport) -> ExportSummary:
    valid, reasons = report.is_valid_for_training()
    if not valid:
        raise ValueError(f"Refusing to export an invalid dataset: {reasons}")

    well_ids = list(report.aligned_wells.keys())
    split = split_wells(well_ids, config.split)
    label_maps = build_label_maps(report)

    def get_las(well_id: str) -> LASFile:
        las_raw = report.well_match.matched[well_id]["las"][0]
        return report.las_files_parsed[las_raw]

    frames: dict[str, WellFrame] = {
        wid: _build_well_frame(wid, report.aligned_wells[wid], get_las(wid), config) for wid in well_ids
    }
    feature_names = next(iter(frames.values())).feature_names if frames else []

    out_dir = Path(config.data.output_dir) / config.dataset.out_dir
    files_written: list[str] = []
    fmt = config.dataset.format

    if split.mode in ("holdout", "explicit"):
        norm = fit_normalization(
            [frames[w].df[feature_names].to_numpy(dtype=np.float32) for w in split.train], feature_names,
            method=config.normalization.method,
        )
        for split_name, wells in (("train", split.train), ("val", split.val), ("test", split.test)):
            for w in wells:
                path = out_dir / split_name / w
                _write_well(frames[w].df, label_maps, norm, feature_names, path, fmt)
                files_written.append(str(path.with_suffix("." + fmt if fmt != "parquet" else ".parquet")))
        normalization_to_save = norm
    else:  # cross_validation: shared raw (unnormalized) well files + per-fold stats
        for w in well_ids:
            path = out_dir / "cv_wells" / w
            _write_well(frames[w].df, label_maps, None, feature_names, path, fmt)
            files_written.append(str(path.with_suffix("." + fmt if fmt != "parquet" else ".parquet")))
        for i, fold in enumerate(split.folds):
            norm = fit_normalization(
                [frames[w].df[feature_names].to_numpy(dtype=np.float32) for w in fold["train"]],
                feature_names, method=config.normalization.method,
            )
            fold_meta_dir = out_dir / "metadata" / f"fold_{i}"
            fold_meta_dir.mkdir(parents=True, exist_ok=True)
            (fold_meta_dir / "normalization.json").write_text(json.dumps(norm.to_json(), indent=2))
        normalization_to_save = None

    meta_dir = out_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "split.json").write_text(json.dumps(split.as_dict(), indent=2))
    (meta_dir / "label_maps.json").write_text(json.dumps(label_maps.to_json(), indent=2))
    (meta_dir / "feature_names.json").write_text(json.dumps(feature_names, indent=2))
    if normalization_to_save is not None:
        (meta_dir / "normalization.json").write_text(json.dumps(normalization_to_save.to_json(), indent=2))
    config.to_yaml(meta_dir / "config_used.yaml")
    (meta_dir / "data_quality_report.txt").write_text(report.to_text())

    return ExportSummary(
        out_dir=str(out_dir), split=split, label_maps=label_maps,
        feature_names=feature_names, n_wells_exported=len(well_ids), files_written=files_written,
    )
