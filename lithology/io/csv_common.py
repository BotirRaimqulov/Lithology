"""Shared column-role detection + core/lab marker handling for the two
expert-interpretation CSVs (lithology, stratigraphy).

Column names are never assumed. Instead each expected *role* (e.g. "the
column holding the well identifier") has a table of known header aliases,
matched the same way LAS curve mnemonics are (case/whitespace/punctuation
-insensitive, Cyrillic and Latin variants both listed explicitly).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

# --------------------------------------------------------------------------- #
ROLE_WELL = "well"
ROLE_MD = "md"                 # single-depth (point) sample
ROLE_TOP = "top"
ROLE_BOTTOM = "bottom"
ROLE_THICKNESS = "thickness"
ROLE_ZONE = "zone"
ROLE_CODE = "code"             # lithology code ("kod")
ROLE_DATASET = "dataset"
ROLE_MARKER = "marker"         # explicit core/lab marker column, if any

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    ROLE_WELL: (
        "WELL", "WELLNUM", "WELLNUMBER", "WELLNO", "WELLNAME", "SKV", "SKVAZHINA",
        "СКВАЖИНА", "СКВ", "НОМЕРСКВАЖИНЫ",
    ),
    ROLE_MD: (
        "MD", "DEPTH", "MEASUREDDEPTH", "ГЛУБИНА", "ГЛ",
    ),
    ROLE_TOP: (
        "TOP", "FROM", "TOPDEPTH", "KROVLYA", "КРОВЛЯ", "ВЕРХ", "НАЧАЛО",
    ),
    ROLE_BOTTOM: (
        "BOTTOM", "TO", "BOTTOMDEPTH", "PODOSHVA", "ПОДОШВА", "НИЗ", "КОНЕЦ",
    ),
    ROLE_THICKNESS: (
        "THICKNESS", "MOSHCHNOST", "МОЩНОСТЬ", "ТОЛЩИНА",
    ),
    ROLE_ZONE: (
        "ZONENAME", "ZONE", "STRATIGRAPHICZONE", "HORIZON", "ГОРИЗОНТ", "ПЛАСТ",
        "СВИТА", "SVITA", "STRATZONE",
    ),
    ROLE_CODE: (
        "KOD", "CODE", "LITHOLOGYCODE", "LITHCODE", "КОД", "ЛИТОЛОГИЯ", "ЛИТОКОД",
        "LITHOLOGY",
    ),
    ROLE_DATASET: (
        "DATASETNAME", "DATASET", "НАБОРДАННЫХ",
    ),
    ROLE_MARKER: (
        "MARKER", "FLAG", "SOURCE", "NOTE", "CORE", "КЕРН", "ПРИМЕЧАНИЕ", "ИСТОЧНИК",
    ),
}


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).upper().strip()
    return re.sub(r"[^0-9A-ZА-ЯЁ]", "", text)


@dataclass
class ColumnResolution:
    role_to_column: dict = field(default_factory=dict)   # role -> actual df column name
    unresolved_columns: list = field(default_factory=list)  # df columns matching no role


def resolve_columns(columns: list[str]) -> ColumnResolution:
    lookup = {}
    for role, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            lookup[_normalize(alias)] = role

    role_to_column: dict[str, str] = {}
    unresolved = []
    for col in columns:
        norm = _normalize(col)
        role = lookup.get(norm)
        if role is None:
            # fuzzy containment fallback
            for norm_alias, cand_role in lookup.items():
                if len(norm_alias) >= 3 and (norm_alias in norm or norm in norm_alias):
                    role = cand_role
                    break
        if role is None:
            unresolved.append(col)
        elif role not in role_to_column:
            role_to_column[role] = col
        # if role already claimed by an earlier column, leave the extra one
        # unresolved rather than silently overwriting the mapping.
        elif role_to_column[role] != col:
            unresolved.append(col)

    return ColumnResolution(role_to_column=role_to_column, unresolved_columns=unresolved)


def _column_preview(df: pd.DataFrame, n_values: int = 8) -> str:
    """A per-column sample of unique values, for schema-detection error
    messages -- when column *names* don't match any known alias, seeing the
    actual *content* is often the only way to tell which column is which
    without guessing."""
    lines = ["Sample values per column (to help identify the right one):"]
    for col in df.columns:
        values = df[col].dropna().unique()[:n_values].tolist()
        lines.append(f"    {col!r}: {values}")
    return "\n".join(lines)


def read_csv_robust(path: str) -> pd.DataFrame:
    """Read a CSV trying a few encodings/separators commonly seen from Excel exports."""
    last_err = None
    for enc in ("utf-8-sig", "cp1251", "latin-1"):
        for sep in (None, ";", ","):
            try:
                return pd.read_csv(path, encoding=enc, sep=sep, engine="python")
            except Exception as e:  # noqa: BLE001 - genuinely need to try alternatives
                last_err = e
    raise ValueError(f"Could not read CSV {path} with any encoding/separator combination: {last_err}")


def extract_marker(raw_value: object, marker_symbols: tuple[str, ...]) -> tuple[str, bool]:
    """Strip a leading/trailing core/lab marker (e.g. '*') off a cell value.

    Returns (cleaned_value_as_str, core_verified). Never assumes the marker's
    exact position -- checks both ends of the raw string.
    """
    if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
        return "", False
    text = str(raw_value).strip()
    found = False
    for sym in marker_symbols:
        if text.startswith(sym):
            text = text[len(sym):].strip()
            found = True
        if text.endswith(sym):
            text = text[: -len(sym)].strip()
            found = True
    return text, found
