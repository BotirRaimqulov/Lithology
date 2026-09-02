"""Well-identifier normalization and cross-source matching.

The LAS ``WELL`` field and the well identifiers used in the two expert
CSVs are not guaranteed to use the same convention (e.g. a field-style id
like ``2-12-27`` vs. a plain sequential number like ``2006``, or the same
number spelled as ``2006``, ``2006.0``, or ``Well-2006``). This module:

1. Normalizes each raw identifier deterministically (never guesses meaning
   it can't justify -- e.g. it will NOT assume "2-12-27" and "2006" are
   the same well).
2. Matches wells ONLY on exact-after-normalization or numeric equality by
   default -- both high-confidence, unambiguous rules.
3. Surfaces everything that did *not* match, plus low-confidence
   "possible match" suggestions, for a human/report to review. Nothing is
   auto-merged based on a heuristic guess.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

_GENERIC_PREFIXES = ("WELL", "СКВАЖИНА", "СКВ", "SKV", "NUMBER", "NUM", "NO")
_PREFIX_RE = re.compile(
    r"^(?:" + "|".join(_GENERIC_PREFIXES) + r")[\s\-_#.:]*", re.IGNORECASE
)
_DASH_VARIANTS_RE = re.compile(r"[‐-―−]")  # unicode dash/minus variants


@dataclass(frozen=True)
class NormalizedWell:
    raw: str
    normalized: str                 # generic-prefix stripped, dashes unified, trimmed
    numeric_id: Optional[int]       # set only if `normalized` is purely numeric (int or int-valued float)

    @property
    def match_key(self) -> str:
        """The key actually used for equality: numeric id if available, else the
        normalized string. Two wells match iff their match_key is equal."""
        return str(self.numeric_id) if self.numeric_id is not None else self.normalized


def normalize_well_id(raw: object) -> NormalizedWell:
    raw_str = "" if raw is None else str(raw).strip()
    text = unicodedata.normalize("NFKC", raw_str)
    text = _DASH_VARIANTS_RE.sub("-", text)
    text = _PREFIX_RE.sub("", text).strip(" \t-_#.")
    text = re.sub(r"\s+", " ", text)
    normalized = text.upper()

    numeric_id: Optional[int] = None
    try:
        f = float(normalized)
        if f.is_integer():
            numeric_id = int(f)
    except ValueError:
        pass

    return NormalizedWell(raw=raw_str, normalized=normalized, numeric_id=numeric_id)


def _last_numeric_token(normalized: str) -> Optional[str]:
    tokens = re.split(r"[-_ ]+", normalized)
    for tok in reversed(tokens):
        if tok.isdigit():
            return str(int(tok))
    return None


@dataclass
class WellMatchReport:
    las_wells: dict = field(default_factory=dict)            # raw -> NormalizedWell
    lithology_wells: dict = field(default_factory=dict)
    stratigraphy_wells: dict = field(default_factory=dict)

    matched: dict = field(default_factory=dict)               # match_key -> {"las": [...], "lithology": [...], "stratigraphy": [...]}
    las_without_labels: list = field(default_factory=list)    # raw LAS wells
    labels_without_las: list = field(default_factory=list)    # (source, raw) tuples
    weak_suggestions: list = field(default_factory=list)      # (las_raw, label_source, label_raw, method)

    def summary(self) -> dict:
        return {
            "n_las_wells": len(self.las_wells),
            "n_lithology_wells": len(self.lithology_wells),
            "n_stratigraphy_wells": len(self.stratigraphy_wells),
            "n_matched": len(self.matched),
            "n_las_without_labels": len(self.las_without_labels),
            "n_labels_without_las": len(self.labels_without_las),
            "n_weak_suggestions": len(self.weak_suggestions),
        }


def build_well_match_report(
    las_wells: list, lithology_wells: list, stratigraphy_wells: list
) -> WellMatchReport:
    report = WellMatchReport()
    report.las_wells = {w: normalize_well_id(w) for w in dict.fromkeys(las_wells)}
    report.lithology_wells = {w: normalize_well_id(w) for w in dict.fromkeys(lithology_wells)}
    report.stratigraphy_wells = {w: normalize_well_id(w) for w in dict.fromkeys(stratigraphy_wells)}

    by_key: dict[str, dict[str, list[str]]] = {}
    for source, wells in (
        ("las", report.las_wells),
        ("lithology", report.lithology_wells),
        ("stratigraphy", report.stratigraphy_wells),
    ):
        for raw, nw in wells.items():
            by_key.setdefault(nw.match_key, {"las": [], "lithology": [], "stratigraphy": []})
            by_key[nw.match_key][source].append(raw)

    label_keys = set()
    for key, sources in by_key.items():
        has_las = bool(sources["las"])
        has_label = bool(sources["lithology"]) or bool(sources["stratigraphy"])
        if has_las and has_label:
            report.matched[key] = sources
        if has_label:
            label_keys.add(key)

    for raw, nw in report.las_wells.items():
        if nw.match_key not in report.matched:
            report.las_without_labels.append(raw)

    for source_name, wells in (("lithology", report.lithology_wells), ("stratigraphy", report.stratigraphy_wells)):
        for raw, nw in wells.items():
            if nw.match_key not in {k for k in report.matched}:
                report.labels_without_las.append((source_name, raw))

    # Low-confidence suggestions: match unmatched LAS wells to unmatched
    # label wells via the last numeric token (e.g. "2-12-2006" ~ "2006").
    # Reported only -- never merged into `matched`.
    unmatched_label_by_token: dict[str, list[tuple]] = {}
    for source_name, raw in report.labels_without_las:
        nw = report.lithology_wells.get(raw) if source_name == "lithology" else report.stratigraphy_wells.get(raw)
        token = _last_numeric_token(nw.normalized) if nw else None
        if token:
            unmatched_label_by_token.setdefault(token, []).append((source_name, raw))

    for las_raw in report.las_without_labels:
        nw = report.las_wells[las_raw]
        token = _last_numeric_token(nw.normalized)
        if token and token in unmatched_label_by_token:
            for source_name, label_raw in unmatched_label_by_token[token]:
                report.weak_suggestions.append((las_raw, source_name, label_raw, "last_numeric_token"))

    return report
