"""Stratigraphy CSV parser: well, zone name, top, bottom, thickness (interval-based)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from lithology.io.csv_common import (
    ROLE_BOTTOM, ROLE_THICKNESS, ROLE_TOP, ROLE_WELL, ROLE_ZONE,
    ColumnResolution, _column_preview, normalize_label_text, read_csv_robust, resolve_columns,
)


class CSVSchemaError(Exception):
    pass


@dataclass
class StratigraphyRecord:
    well_raw: str
    zone_name: str
    top: float
    bottom: float
    thickness_declared: Optional[float]
    thickness_computed: float
    row_index: int


@dataclass
class StratigraphyCSVResult:
    records: list = field(default_factory=list)
    dropped: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    columns: Optional[ColumnResolution] = None


def parse_stratigraphy_csv(path: str, thickness_tolerance: float = 1e-3) -> StratigraphyCSVResult:
    df = read_csv_robust(path)
    df.columns = [str(c).strip() for c in df.columns]
    cols = resolve_columns(list(df.columns))

    result = StratigraphyCSVResult(columns=cols)
    if cols.unresolved_columns:
        result.warnings.append(
            f"Unrecognized stratigraphy CSV columns (kept in raw file, unused): "
            f"{cols.unresolved_columns}"
        )

    required = {ROLE_WELL, ROLE_ZONE, ROLE_TOP, ROLE_BOTTOM}
    missing = required - cols.role_to_column.keys()
    if missing:
        raise CSVSchemaError(
            f"Stratigraphy CSV {path} is missing required column role(s) {missing}. "
            f"Columns present: {list(df.columns)}\n{_column_preview(df)}"
        )

    well_col = cols.role_to_column[ROLE_WELL]
    zone_col = cols.role_to_column[ROLE_ZONE]
    top_col = cols.role_to_column[ROLE_TOP]
    bottom_col = cols.role_to_column[ROLE_BOTTOM]
    thickness_col = cols.role_to_column.get(ROLE_THICKNESS)

    n_thickness_mismatch = 0
    n_zone_name_normalized = 0
    for idx, row in df.iterrows():
        well_val = row[well_col]
        if pd.isna(well_val) or str(well_val).strip() == "":
            result.dropped.append((idx, "missing well identifier"))
            continue

        zone_val = row[zone_col]
        if pd.isna(zone_val) or str(zone_val).strip() == "":
            result.dropped.append((idx, "missing zone name"))
            continue
        zone_name = normalize_label_text(zone_val)
        if zone_name != str(zone_val).strip():
            n_zone_name_normalized += 1

        top_val = pd.to_numeric(row[top_col], errors="coerce")
        bottom_val = pd.to_numeric(row[bottom_col], errors="coerce")
        if pd.isna(top_val) or pd.isna(bottom_val):
            result.dropped.append((idx, "non-numeric top/bottom"))
            continue
        top_val, bottom_val = float(top_val), float(bottom_val)

        if bottom_val < top_val:
            result.dropped.append((idx, f"bottom ({bottom_val}) < top ({top_val})"))
            continue
        if top_val < 0:
            result.dropped.append((idx, f"negative top depth ({top_val})"))
            continue

        thickness_declared = None
        if thickness_col is not None:
            t = pd.to_numeric(row[thickness_col], errors="coerce")
            thickness_declared = None if pd.isna(t) else float(t)

        thickness_computed = bottom_val - top_val
        if thickness_declared is not None and abs(thickness_declared - thickness_computed) > thickness_tolerance:
            n_thickness_mismatch += 1

        result.records.append(
            StratigraphyRecord(
                well_raw=str(well_val).strip(),
                zone_name=zone_name,
                top=top_val,
                bottom=bottom_val,
                thickness_declared=thickness_declared,
                thickness_computed=thickness_computed,
                row_index=int(idx),
            )
        )

    if result.dropped:
        result.warnings.append(
            f"Dropped {len(result.dropped)}/{len(df)} stratigraphy rows during parsing "
            f"(see `dropped` for reasons, none discarded silently)."
        )
    if n_thickness_mismatch:
        result.warnings.append(
            f"{n_thickness_mismatch} row(s) have a declared Thickness that disagrees with "
            f"(bottom - top) by more than {thickness_tolerance}; computed thickness is used "
            f"downstream, declared value is kept for reference."
        )
    if n_zone_name_normalized:
        result.warnings.append(
            f"{n_zone_name_normalized} row(s) had internal whitespace stripped from Zone Name "
            f"(e.g. 'P 3-N1/1' -> 'P3-N1/1') to merge inconsistent spellings of the same zone "
            f"into one label-vocabulary class."
        )

    return result
