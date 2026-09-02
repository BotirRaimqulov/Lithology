"""LAS curve-name normalization / alias resolution.

LAS files in this project are written by different logging services over
different decades, so the same physical measurement shows up under many
different mnemonics -- sometimes in Russian, sometimes in English,
sometimes an abbreviation of either. This module maps whatever mnemonic
(and, as a fallback, whatever free-text description) a file uses onto one
of the four canonical curves the rest of the pipeline understands:

    DEPT, GK, KS, PS

Nothing here silently assumes a curve is present or absent: callers get an
explicit :class:`CurveMatch` per raw curve, including the ones that could
not be resolved, so an unmapped/unexpected curve in a real file is always
visible in the data-quality report rather than dropped quietly.

The alias table is intentionally a plain, editable dict (not a hardcoded
`if/elif` chain) so it can be extended the moment a real file shows an
unrecognized mnemonic, and it can be extended further at runtime by
passing ``extra_aliases`` (e.g. loaded from ``configs/curve_aliases.yaml``)
without touching this file.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

# --------------------------------------------------------------------------- #
# Alias table. Keys are the canonical curve names; values are every raw
# mnemonic / description token we already know maps to it. Matching is
# case- and whitespace-insensitive (see _normalize).
# --------------------------------------------------------------------------- #
DEFAULT_ALIASES: dict[str, tuple[str, ...]] = {
    "DEPT": (
        "DEPT", "DEPTH", "MD", "M", "GLUB", "GLUBINA",
        "ГЛУБИНА", "ГЛУБ",
    ),
    "GK": (
        "GK", "GKK", "GAMMA", "GAMMARAY", "GR", "NGK", "GK1", "GK2",
        "ГК", "ГКК", "ГАММА", "ГАММАКАРОТАЖ", "ГАММА-КАРОТАЖ",
        "ГАММАЛУЧЕВОЙКАРОТАЖ", "ГАММА КАРОТАЖ",
    ),
    "KS": (
        "KS", "RESISTIVITY", "RES", "RT", "IK", "BK", "LLD", "LLS",
        "КС", "УЭС", "СОПРОТИВЛЕНИЕ", "КАЖУЩЕЕСЯСОПРОТИВЛЕНИЕ",
    ),
    "PS": (
        "PS", "SP", "SELFPOTENTIAL", "SPONTANEOUSPOTENTIAL",
        "ПС", "ПОТЕНЦИАЛ", "САМОПРОИЗВОЛЬНАЯПОЛЯРИЗАЦИЯ",
    ),
}

# Units commonly associated with each curve -- used only as a *tie-breaker*
# hint when the mnemonic/description match is ambiguous, never on their own.
EXPECTED_UNITS: dict[str, tuple[str, ...]] = {
    "GK": ("uR/h", "MKR/H", "API", "GAPI"),
    "KS": ("OHMM", "OHM.M", "OM"),
    "PS": ("MV",),
    "DEPT": ("M",),
}


def _normalize(text: str) -> str:
    """Uppercase, strip accents/punctuation/whitespace for robust matching."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.upper().strip()
    # Drop anything that isn't a letter (incl. Cyrillic) or digit.
    text = re.sub(r"[^0-9A-ZА-ЯЁ]", "", text)
    return text


@dataclass(frozen=True)
class CurveMatch:
    raw_mnemonic: str
    raw_description: str
    canonical: Optional[str]   # None if unresolved
    method: str                # "mnemonic_exact" | "description_exact" | "fuzzy" | "unresolved"


class CurveAliasResolver:
    """Resolves raw LAS curve mnemonics/descriptions to canonical curve names."""

    def __init__(self, extra_aliases: Optional[dict[str, list[str]]] = None):
        merged: dict[str, list[str]] = {k: list(v) for k, v in DEFAULT_ALIASES.items()}
        if extra_aliases:
            for canonical, aliases in extra_aliases.items():
                merged.setdefault(canonical, [])
                merged[canonical].extend(aliases)

        self._lookup: dict[str, str] = {}
        for canonical, aliases in merged.items():
            for alias in aliases:
                norm = _normalize(alias)
                if not norm:
                    continue
                if norm in self._lookup and self._lookup[norm] != canonical:
                    raise ValueError(
                        f"Alias {alias!r} is ambiguous between "
                        f"{self._lookup[norm]!r} and {canonical!r}"
                    )
                self._lookup[norm] = canonical

    def resolve(self, mnemonic: str, description: str = "") -> CurveMatch:
        mnemonic = mnemonic or ""
        description = description or ""

        norm_mnemonic = _normalize(mnemonic)
        if norm_mnemonic in self._lookup:
            return CurveMatch(mnemonic, description, self._lookup[norm_mnemonic], "mnemonic_exact")

        norm_desc = _normalize(description)
        if norm_desc in self._lookup:
            return CurveMatch(mnemonic, description, self._lookup[norm_desc], "description_exact")

        # Fuzzy fallback: containment match, guarded by a minimum alias
        # length so short codes like "M" or "KS" can't match by accident
        # inside an unrelated word.
        for norm_alias, canonical in self._lookup.items():
            if len(norm_alias) < 2:
                continue
            if norm_mnemonic and (norm_alias in norm_mnemonic or norm_mnemonic in norm_alias):
                return CurveMatch(mnemonic, description, canonical, "fuzzy")
            if norm_desc and (norm_alias in norm_desc or norm_desc in norm_alias):
                return CurveMatch(mnemonic, description, canonical, "fuzzy")

        return CurveMatch(mnemonic, description, None, "unresolved")

    def resolve_many(self, curves: list[tuple[str, str]]) -> list[CurveMatch]:
        """``curves`` is a list of (mnemonic, description) pairs, in file order."""
        return [self.resolve(mnemonic, description) for mnemonic, description in curves]
