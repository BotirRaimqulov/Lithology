"""Robust-ish LAS 2.0 parser.

Deliberately hand-rolled (not delegated to a third-party LAS library) so
every parsing/cleaning decision is visible and loggable, per the project's
"log every data-cleaning decision" rule. It supports the subset of the LAS
2.0 spec actually needed here:

    ~Version
    ~Well Information   (STRT, STOP, STEP, NULL, WELL, ...)
    ~Curve Information  (defines column order for ~ASCII)
    ~Parameter          (ignored, but preserved as raw text)
    ~Ascii / ~A         (the data matrix, WRAP NO or WRAP YES)

Nothing about column names, encoding, or the NULL value is assumed to be
identical across files: each is re-derived from that file's own header,
with a documented fallback when a header field is missing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from lithology.constants import CANONICAL_CURVES, DEFAULT_LAS_NULL_VALUE
from lithology.io.curve_aliases import CurveAliasResolver, CurveMatch

_SECTION_RE = re.compile(r"^~([A-Za-z])")
# MNEM . UNIT   VALUE-OR-BLANK   [: DESCRIPTION]
_LINE_RE = re.compile(r"^\s*([^.\s]+)\s*\.(\S*)\s*([^:]*?)\s*(?::\s*(.*))?$")

_ENCODINGS_TO_TRY = ("utf-8", "cp1251", "latin-1")


class LASParseError(Exception):
    """Raised only for unrecoverable parsing failures (e.g. no DEPT curve)."""


@dataclass
class LASHeaderField:
    mnemonic: str
    unit: str
    value: str
    description: str


@dataclass
class LASFile:
    path: str
    well_raw: Optional[str]
    step_header: Optional[float]     # STEP as declared in ~Well Information
    step_actual: Optional[float]     # median of np.diff(depth) -- may disagree
    strt: Optional[float]
    stop: Optional[float]
    null_value: float
    curve_matches: list[CurveMatch]
    curves: dict                     # canonical name -> np.ndarray (float, NaN = missing)
    missing_mask: dict               # canonical name -> np.ndarray[bool] (True = missing)
    n_points: int
    warnings: list = field(default_factory=list)

    def missing_fraction(self, canonical: str) -> Optional[float]:
        mask = self.missing_mask.get(canonical)
        if mask is None or len(mask) == 0:
            return None
        return float(mask.mean())


def _read_text(path: Path) -> tuple[str, str]:
    last_err = None
    for enc in _ENCODINGS_TO_TRY:
        try:
            return path.read_text(encoding=enc), enc
        except (UnicodeDecodeError, LookupError) as e:
            last_err = e
    # Should not happen since latin-1 never raises, but keep it explicit.
    raise LASParseError(f"Could not decode {path} with any of {_ENCODINGS_TO_TRY}: {last_err}")


def _split_sections(text: str) -> dict[str, list[str]]:
    """Group non-comment lines by LAS section letter (V/W/C/P/O/A)."""
    sections: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = _SECTION_RE.match(stripped)
        if m:
            current = m.group(1).upper()
            sections.setdefault(current, [])
            continue
        if current is not None:
            sections.setdefault(current, []).append(line)
    return sections


def _parse_header_lines(lines: list[str]) -> list[LASHeaderField]:
    fields = []
    for line in lines:
        m = _LINE_RE.match(line)
        if not m:
            continue
        mnem, unit, value, desc = m.groups()
        fields.append(LASHeaderField(mnem.strip(), (unit or "").strip(),
                                      (value or "").strip(), (desc or "").strip()))
    return fields


def _find_field(fields: list[LASHeaderField], *names: str) -> Optional[LASHeaderField]:
    wanted = {n.upper() for n in names}
    for f in fields:
        if f.mnemonic.upper() in wanted:
            return f
    return None


def _to_float(text: str) -> Optional[float]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_las_file(path: str | Path, alias_resolver: Optional[CurveAliasResolver] = None) -> LASFile:
    path = Path(path)
    resolver = alias_resolver or CurveAliasResolver()
    warnings: list[str] = []

    text, encoding = _read_text(path)
    if encoding != "utf-8":
        warnings.append(f"File decoded with fallback encoding '{encoding}' (utf-8 failed).")

    sections = _split_sections(text)
    if "W" not in sections:
        raise LASParseError(f"{path}: no ~Well Information section found.")
    if "C" not in sections:
        raise LASParseError(f"{path}: no ~Curve Information section found.")
    if "A" not in sections:
        raise LASParseError(f"{path}: no ~Ascii/~A data section found.")

    well_fields = _parse_header_lines(sections["W"])
    curve_fields = _parse_header_lines(sections["C"])

    well_field = _find_field(well_fields, "WELL")
    well_raw = well_field.value if well_field and well_field.value else None
    if well_raw is None:
        warnings.append("No WELL mnemonic (or empty value) in ~Well Information section.")

    strt_field = _find_field(well_fields, "STRT")
    stop_field = _find_field(well_fields, "STOP")
    step_field = _find_field(well_fields, "STEP")
    null_field = _find_field(well_fields, "NULL")
    wrap_field = None
    if "V" in sections:
        wrap_field = _find_field(_parse_header_lines(sections["V"]), "WRAP")

    strt = _to_float(strt_field.value) if strt_field else None
    stop = _to_float(stop_field.value) if stop_field else None
    step_header = _to_float(step_field.value) if step_field else None
    null_value = _to_float(null_field.value) if null_field else None
    if null_value is None:
        warnings.append(
            f"No NULL value declared in header; falling back to default {DEFAULT_LAS_NULL_VALUE}."
        )
        null_value = DEFAULT_LAS_NULL_VALUE

    is_wrapped = bool(wrap_field and wrap_field.value.strip().upper() == "YES")

    if not curve_fields:
        raise LASParseError(f"{path}: ~Curve Information section is empty.")

    curve_matches = resolver.resolve_many([(f.mnemonic, f.description) for f in curve_fields])
    n_cols = len(curve_fields)

    # --- parse the ASCII data matrix -------------------------------------
    tokens: list[str] = []
    for line in sections["A"]:
        tokens.extend(line.split())

    rows: list[list[float]] = []
    if is_wrapped:
        buf: list[float] = []
        for tok in tokens:
            v = _to_float(tok)
            buf.append(v if v is not None else float("nan"))
            if len(buf) == n_cols:
                rows.append(buf)
                buf = []
        if buf:
            warnings.append(
                f"Trailing {len(buf)} wrapped values did not complete a full row; discarded."
            )
    else:
        if len(tokens) % n_cols != 0:
            n_complete = len(tokens) // n_cols
            dropped = len(tokens) - n_complete * n_cols
            warnings.append(
                f"ASCII data length ({len(tokens)}) is not a multiple of column count "
                f"({n_cols}); dropping trailing {dropped} malformed token(s)."
            )
            tokens = tokens[: n_complete * n_cols]
        arr = np.array([_to_float(t) if _to_float(t) is not None else float("nan") for t in tokens],
                        dtype=float)
        if n_cols:
            rows = arr.reshape(-1, n_cols).tolist()

    data = np.array(rows, dtype=float) if rows else np.zeros((0, n_cols))
    n_points = data.shape[0]

    # --- map columns -> canonical curves, NULL -> NaN, build masks -------
    curves: dict[str, np.ndarray] = {}
    missing_mask: dict[str, np.ndarray] = {}
    seen_canonical: dict[str, int] = {}
    for col_idx, match in enumerate(curve_matches):
        canonical = match.canonical
        raw_col = data[:, col_idx] if n_points else np.zeros(0)

        is_null = np.isclose(raw_col, null_value, atol=1e-6, equal_nan=False)
        is_null |= np.isclose(raw_col, DEFAULT_LAS_NULL_VALUE, atol=1e-6, equal_nan=False)
        is_null |= np.isnan(raw_col)
        cleaned = raw_col.copy()
        cleaned[is_null] = np.nan

        if canonical is None:
            # Keep unrecognized curves out of `curves` but never lose them
            # silently -- surfaced via warnings for the quality report.
            warnings.append(
                f"Unrecognized curve '{match.raw_mnemonic}' (desc={match.raw_description!r}) "
                f"could not be mapped to a canonical curve; ignored for modeling but not deleted "
                f"from the source file."
            )
            continue

        if canonical in seen_canonical:
            prev_idx = seen_canonical[canonical]
            warnings.append(
                f"Duplicate mapping to '{canonical}': column {prev_idx} "
                f"('{curve_matches[prev_idx].raw_mnemonic}') and column {col_idx} "
                f"('{match.raw_mnemonic}') both resolve to it; keeping the first occurrence."
            )
            continue

        seen_canonical[canonical] = col_idx
        curves[canonical] = cleaned
        missing_mask[canonical] = is_null

        n_missing = int(is_null.sum())
        if n_points and n_missing:
            warnings.append(
                f"Curve '{canonical}' (raw '{match.raw_mnemonic}'): "
                f"{n_missing}/{n_points} points ({n_missing / n_points:.1%}) are NULL -> NaN."
            )

    if "DEPT" not in curves:
        raise LASParseError(f"{path}: could not resolve a DEPT/depth curve from {curve_fields}.")

    missing_curves = [c for c in CANONICAL_CURVES if c not in curves]
    if missing_curves:
        warnings.append(f"File is missing curve(s) entirely: {missing_curves}.")

    depth = curves["DEPT"]
    step_actual = float(np.median(np.diff(depth))) if len(depth) > 1 else None
    if step_header is not None and step_actual is not None and not np.isclose(
        step_header, step_actual, atol=1e-6
    ):
        warnings.append(
            f"Header STEP={step_header} does not match median depth spacing "
            f"{step_actual:.6g} computed from the DEPT curve; using the computed value "
            f"for downstream alignment."
        )

    return LASFile(
        path=str(path),
        well_raw=well_raw,
        step_header=step_header,
        step_actual=step_actual,
        strt=strt,
        stop=stop,
        null_value=null_value,
        curve_matches=curve_matches,
        curves=curves,
        missing_mask=missing_mask,
        n_points=n_points,
        warnings=warnings,
    )


def find_las_files(las_dir: str | Path) -> list[Path]:
    las_dir = Path(las_dir)
    if not las_dir.exists():
        return []
    return sorted(p for p in las_dir.rglob("*") if p.suffix.lower() in (".las",))
