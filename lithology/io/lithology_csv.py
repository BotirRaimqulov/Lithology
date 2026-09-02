"""Lithology CSV parser.

IMPORTANT geological/schema note (confirmed by the project owner, not
assumed): the lithology table is expected to be **point-based** -- one row
per physical/laboratory (core) sample at a specific measured depth (`MD`),
carrying a lithology `kod` (code), rather than a top/bottom interval that
should be spread across a whole depth range. Quoting the requirement:

    "lithology can be taken from the kernel site, specifically from the
     laboratory sample, not from beginning to end"

i.e. a single row's lithology label must NOT be assumed to hold for an
entire interval unless the CSV explicitly provides top/bottom columns for
it. This module therefore supports both shapes and records which one a
given file actually uses instead of hard-coding either:

  * point schema:    well, MD, kod[, dataset name]
  * interval schema: well, top, bottom, kod[, dataset name]

If a row's code cell carries a core/lab marker symbol (default: ``*``,
configurable via ``alignment.core_marker_symbols``), it is stripped and
recorded as ``core_verified=True`` rather than discarded -- its exact
meaning/placement is confirmed once real data is inspected, but the value
itself is never silently dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from lithology.io.csv_common import (
    ROLE_BOTTOM, ROLE_CODE, ROLE_DATASET, ROLE_MARKER, ROLE_MD, ROLE_TOP, ROLE_WELL,
    ColumnResolution, _column_preview, extract_marker, read_csv_robust, resolve_columns,
)


class CSVSchemaError(Exception):
    pass


@dataclass
class LithologyRecord:
    well_raw: str
    top: float
    bottom: float
    is_point_sample: bool     # True: MD-based row, zero-width sample point
    code_raw: str
    code: str                 # marker stripped
    core_verified: bool
    dataset_name: Optional[str]
    row_index: int


@dataclass
class LithologyCSVResult:
    records: list = field(default_factory=list)
    dropped: list = field(default_factory=list)   # (row_index, reason)
    warnings: list = field(default_factory=list)
    columns: Optional[ColumnResolution] = None
    schema: str = "unknown"   # "point" | "interval"


def parse_lithology_csv(path: str, core_marker_symbols: tuple = ("*",)) -> LithologyCSVResult:
    df = read_csv_robust(path)
    df.columns = [str(c).strip() for c in df.columns]
    cols = resolve_columns(list(df.columns))

    result = LithologyCSVResult(columns=cols)
    if cols.unresolved_columns:
        result.warnings.append(
            f"Unrecognized lithology CSV columns (kept in raw file, unused): "
            f"{cols.unresolved_columns}"
        )

    has_well = ROLE_WELL in cols.role_to_column
    has_code = ROLE_CODE in cols.role_to_column
    has_md = ROLE_MD in cols.role_to_column
    has_interval = ROLE_TOP in cols.role_to_column and ROLE_BOTTOM in cols.role_to_column

    if not has_well:
        raise CSVSchemaError(
            f"Could not detect a well-identifier column in {path}. "
            f"Columns present: {list(df.columns)}\n{_column_preview(df)}"
        )
    if not has_code:
        raise CSVSchemaError(
            f"Could not detect a lithology-code column in {path}. "
            f"Columns present: {list(df.columns)}\n"
            f"None of them matched known code/lithology aliases (kod, code, litho...). "
            f"If one of the columns below actually holds the lithology value under a "
            f"different header name (e.g. it was auto-mapped to 'zone' instead), rename "
            f"it or add the alias to lithology/io/csv_common.py:COLUMN_ALIASES[ROLE_CODE].\n"
            f"{_column_preview(df)}"
        )
    if not has_md and not has_interval:
        raise CSVSchemaError(
            f"Could not detect either an MD (point) column or top/bottom (interval) "
            f"columns in {path}. Columns present: {list(df.columns)}\n{_column_preview(df)}"
        )

    if has_interval:
        result.schema = "interval"
        if has_md:
            result.warnings.append(
                "Both MD and top/bottom columns detected; using top/bottom as the "
                "authoritative interval and ignoring MD."
            )
    else:
        result.schema = "point"

    well_col = cols.role_to_column[ROLE_WELL]
    code_col = cols.role_to_column[ROLE_CODE]
    dataset_col = cols.role_to_column.get(ROLE_DATASET)
    marker_col = cols.role_to_column.get(ROLE_MARKER)

    for idx, row in df.iterrows():
        well_val = row[well_col]
        if pd.isna(well_val) or str(well_val).strip() == "":
            result.dropped.append((idx, "missing well identifier"))
            continue
        well_raw = str(well_val).strip()

        code_cell = row[code_col]
        if pd.isna(code_cell) or str(code_cell).strip() == "":
            result.dropped.append((idx, "missing lithology code"))
            continue
        code_clean, core_from_code = extract_marker(code_cell, core_marker_symbols)
        if not code_clean:
            result.dropped.append((idx, f"lithology code empty after stripping marker ({code_cell!r})"))
            continue

        core_from_marker_col = False
        if marker_col is not None:
            marker_val = row[marker_col]
            if not pd.isna(marker_val) and str(marker_val).strip() != "":
                _, core_from_marker_col = extract_marker(marker_val, core_marker_symbols)
                # A marker column whose value doesn't literally contain the
                # symbol but is non-empty is still a hint worth keeping visible.
                core_from_marker_col = core_from_marker_col or bool(str(marker_val).strip())

        core_verified = core_from_code or core_from_marker_col
        dataset_name = str(row[dataset_col]).strip() if dataset_col else None

        if result.schema == "interval":
            top_val = pd.to_numeric(row[cols.role_to_column[ROLE_TOP]], errors="coerce")
            bottom_val = pd.to_numeric(row[cols.role_to_column[ROLE_BOTTOM]], errors="coerce")
            if pd.isna(top_val) or pd.isna(bottom_val):
                result.dropped.append((idx, "non-numeric top/bottom"))
                continue
            top_val, bottom_val = float(top_val), float(bottom_val)
            if bottom_val < top_val:
                result.dropped.append((idx, f"bottom ({bottom_val}) < top ({top_val})"))
                continue
            is_point = False
        else:
            md_val = pd.to_numeric(row[cols.role_to_column[ROLE_MD]], errors="coerce")
            if pd.isna(md_val):
                result.dropped.append((idx, "non-numeric MD"))
                continue
            top_val = bottom_val = float(md_val)
            is_point = True

        if top_val < 0:
            result.dropped.append((idx, f"negative depth ({top_val})"))
            continue

        result.records.append(
            LithologyRecord(
                well_raw=well_raw,
                top=top_val,
                bottom=bottom_val,
                is_point_sample=is_point,
                code_raw=str(code_cell),
                code=code_clean,
                core_verified=core_verified,
                dataset_name=dataset_name,
                row_index=int(idx),
            )
        )

    if result.dropped:
        result.warnings.append(
            f"Dropped {len(result.dropped)}/{len(df)} lithology rows during parsing "
            f"(see `dropped` for reasons, none discarded silently)."
        )

    n_core = sum(1 for r in result.records if r.core_verified)
    result.warnings.append(
        f"{n_core}/{len(result.records)} lithology records flagged core/lab-verified "
        f"via marker symbol(s) {core_marker_symbols}."
    )

    return result
