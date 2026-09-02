"""End-to-end data-quality report: LAS + CSVs -> well matching -> depth
alignment -> aggregated statistics.

This is the single place that ties every parsing/matching/alignment module
together. It is deliberately conservative: any file that fails to parse is
recorded (path + error) and skipped rather than crashing the whole run, so
one malformed LAS file cannot hide the state of the other 500. Nothing
here fabricates numbers -- if a directory is empty, the report says so
instead of showing zeros that could be mistaken for "checked, found none".
"""
from __future__ import annotations

import collections
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from lithology.alignment.depth_alignment import AlignedWell, align_well
from lithology.config import Config
from lithology.constants import BOUNDARY_POSITIVE, CANONICAL_CURVES, IGNORE_INDEX
from lithology.io.curve_aliases import CurveAliasResolver
from lithology.io.las_parser import LASFile, LASParseError, find_las_files, parse_las_file
from lithology.io.lithology_csv import CSVSchemaError as LithoSchemaError, parse_lithology_csv
from lithology.io.stratigraphy_csv import CSVSchemaError as StratSchemaError, parse_stratigraphy_csv
from lithology.wells.well_id import WellMatchReport, build_well_match_report, normalize_well_id


def _bucket_dropped_reasons(dropped: list) -> dict:
    """Group (row_index, reason) drop records by reason *shape*, collapsing
    the dynamic numbers inside a message (e.g. "bottom (5.0) < top (10.0)")
    so 500 rows dropped for the same underlying cause show up as one
    line with a count, not 500 near-duplicate messages."""
    counts: collections.Counter = collections.Counter()
    for _, reason in dropped:
        bucket = re.sub(r"[-+]?\d+\.?\d*", "#", reason)
        counts[bucket] += 1
    return dict(counts.most_common())


@dataclass
class DataAvailability:
    las_dir: str
    las_dir_exists: bool
    n_las_files_found: int
    lithology_csv: str
    lithology_csv_exists: bool
    stratigraphy_csv: str
    stratigraphy_csv_exists: bool

    @property
    def has_any_data(self) -> bool:
        return self.n_las_files_found > 0 or self.lithology_csv_exists or self.stratigraphy_csv_exists


@dataclass
class DataQualityReport:
    availability: DataAvailability
    las_parse_errors: list = field(default_factory=list)          # (path, message)
    las_files_parsed: dict = field(default_factory=dict)           # well_raw -> LASFile
    lithology_parse_error: Optional[str] = None
    stratigraphy_parse_error: Optional[str] = None
    lithology_dropped: list = field(default_factory=list)
    stratigraphy_dropped: list = field(default_factory=list)
    lithology_warnings: list = field(default_factory=list)
    stratigraphy_warnings: list = field(default_factory=list)
    well_match: Optional[WellMatchReport] = None

    total_depth_points: int = 0
    missing_count: dict = field(default_factory=dict)   # canonical curve -> count
    missing_fraction: dict = field(default_factory=dict)

    n_lithology_records: int = 0
    n_stratigraphy_records: int = 0
    n_zone_overlap_points: int = 0
    n_zone_out_of_range_intervals: int = 0
    n_lithology_interval_overlap_points: int = 0
    n_lithology_out_of_range_intervals: int = 0
    n_lithology_point_unmatched: int = 0
    n_lithology_point_conflicts: int = 0

    lithology_class_distribution: dict = field(default_factory=dict)
    zone_distribution: dict = field(default_factory=dict)
    boundary_distribution: dict = field(default_factory=dict)

    aligned_wells: dict = field(default_factory=dict)   # well_key -> AlignedWell
    warnings: list = field(default_factory=list)

    def is_valid_for_training(self) -> tuple:
        reasons = []
        if not self.availability.has_any_data:
            reasons.append("No LAS files or CSVs were found in the configured directories.")
            return False, reasons
        if not self.aligned_wells:
            reasons.append("No well could be matched between LAS files and the CSV labels.")
        if self.total_depth_points == 0:
            reasons.append("Matched wells contain zero usable depth points.")
        if not self.lithology_class_distribution:
            reasons.append("No lithology labels were assigned to any depth point.")
        if not self.zone_distribution:
            reasons.append("No stratigraphic zone labels were assigned to any depth point.")
        return (len(reasons) == 0), reasons

    def to_text(self) -> str:
        lines = []
        a = self.availability
        lines.append("=" * 78)
        lines.append("DATA QUALITY REPORT")
        lines.append("=" * 78)
        lines.append(f"LAS directory        : {a.las_dir} (exists={a.las_dir_exists})")
        lines.append(f"Lithology CSV         : {a.lithology_csv} (exists={a.lithology_csv_exists})")
        lines.append(f"Stratigraphy CSV      : {a.stratigraphy_csv} (exists={a.stratigraphy_csv_exists})")

        if not a.has_any_data:
            lines.append("")
            lines.append("No real dataset found.")
            lines.append("Pipeline infrastructure is ready.")
            lines.append("Place LAS and CSV files in the configured directories and re-run this tool.")
            return "\n".join(lines)

        lines.append("")
        lines.append(f"LAS files found       : {a.n_las_files_found}")
        lines.append(f"LAS files parsed OK   : {len(self.las_files_parsed)}")
        if self.las_parse_errors:
            lines.append(f"LAS parse FAILURES    : {len(self.las_parse_errors)}")
            for path, msg in self.las_parse_errors[:20]:
                lines.append(f"    - {path}: {msg}")

        if self.lithology_parse_error:
            lines.append(f"Lithology CSV FAILED to parse: {self.lithology_parse_error}")
        else:
            lines.append(f"Lithology records parsed  : {self.n_lithology_records} "
                          f"(dropped {len(self.lithology_dropped)})")
            for reason, count in _bucket_dropped_reasons(self.lithology_dropped).items():
                lines.append(f"        dropped x{count}: {reason}")
        if self.stratigraphy_parse_error:
            lines.append(f"Stratigraphy CSV FAILED to parse: {self.stratigraphy_parse_error}")
        else:
            lines.append(f"Stratigraphy records parsed: {self.n_stratigraphy_records} "
                          f"(dropped {len(self.stratigraphy_dropped)})")
            for reason, count in _bucket_dropped_reasons(self.stratigraphy_dropped).items():
                lines.append(f"        dropped x{count}: {reason}")

        if self.well_match is not None:
            s = self.well_match.summary()
            lines.append("")
            lines.append("-- Well matching " + "-" * 60)
            lines.append(f"LAS wells             : {s['n_las_wells']}")
            lines.append(f"Lithology wells       : {s['n_lithology_wells']}")
            lines.append(f"Stratigraphy wells    : {s['n_stratigraphy_wells']}")
            lines.append(f"Matched wells         : {s['n_matched']}")
            lines.append(f"LAS wells w/o labels  : {s['n_las_without_labels']} -> "
                         f"{self.well_match.las_without_labels}")
            lines.append(f"Labels w/o LAS well   : {s['n_labels_without_las']} -> "
                         f"{self.well_match.labels_without_las}")
            if self.well_match.weak_suggestions:
                lines.append(f"Weak match suggestions (NOT auto-applied): "
                             f"{self.well_match.weak_suggestions}")

        lines.append("")
        lines.append("-- Depth points / missing values " + "-" * 44)
        lines.append(f"Total aligned depth points: {self.total_depth_points}")
        for c in CANONICAL_CURVES:
            if c == "DEPT":
                continue
            n_missing = self.missing_count.get(c, 0)
            frac = self.missing_fraction.get(c, 0.0)
            lines.append(f"    Missing {c:<4}: {n_missing} ({frac:.1%})")

        lines.append("")
        lines.append("-- Interval diagnostics " + "-" * 53)
        lines.append(f"Zone: overlapping points          : {self.n_zone_overlap_points}")
        lines.append(f"Zone: out-of-range intervals       : {self.n_zone_out_of_range_intervals}")
        lines.append(f"Lithology (interval): overlap pts  : {self.n_lithology_interval_overlap_points}")
        lines.append(f"Lithology (interval): out-of-range : {self.n_lithology_out_of_range_intervals}")
        lines.append(f"Lithology (point/core): unmatched  : {self.n_lithology_point_unmatched}")
        lines.append(f"Lithology (point/core): conflicts  : {self.n_lithology_point_conflicts}")

        lithology_covered = sum(self.lithology_class_distribution.values())
        zone_covered = sum(self.zone_distribution.values())
        lithology_coverage = lithology_covered / self.total_depth_points if self.total_depth_points else 0.0
        zone_coverage = zone_covered / self.total_depth_points if self.total_depth_points else 0.0

        lines.append("")
        lines.append("-- Label coverage " + "-" * 59)
        lines.append(f"Depth points with a lithology label: {lithology_covered}/{self.total_depth_points} "
                     f"({lithology_coverage:.1%})")
        lines.append(f"Depth points with a zone label      : {zone_covered}/{self.total_depth_points} "
                     f"({zone_coverage:.1%})")
        if lithology_coverage < 0.5 or zone_coverage < 0.5:
            lines.append(
                "    NOTE: less than half of the depth range has a label for at least one task. "
                "This is common for well logs (core samples/mapped intervals rarely span the "
                "whole well) but means many training crops will have zero valid targets for "
                "that task -- those crops correctly contribute 0 loss for it rather than NaN "
                "(see lithology/models/losses.py), but if coverage is extremely sparse, consider "
                "a smaller training crop length so more crops land entirely inside a labeled span."
            )

        lines.append("")
        lines.append("-- Class distributions " + "-" * 54)
        lines.append(f"Lithology classes ({len(self.lithology_class_distribution)}): "
                     f"{dict(sorted(self.lithology_class_distribution.items()))}")
        lines.append(f"Zone classes ({len(self.zone_distribution)}): "
                     f"{dict(sorted(self.zone_distribution.items()))}")
        lines.append(f"Boundary distribution: {self.boundary_distribution}")

        valid, reasons = self.is_valid_for_training()
        lines.append("")
        lines.append("-- Training readiness " + "-" * 55)
        lines.append(f"Valid for training: {valid}")
        for r in reasons:
            lines.append(f"    - {r}")

        if self.warnings:
            lines.append("")
            lines.append("-- General warnings " + "-" * 57)
            for w in self.warnings[:50]:
                lines.append(f"    - {w}")

        return "\n".join(lines)


def build_data_quality_report(config: Config) -> DataQualityReport:
    las_dir = Path(config.data.las_dir)
    las_files = find_las_files(las_dir)

    availability = DataAvailability(
        las_dir=str(las_dir),
        las_dir_exists=las_dir.exists(),
        n_las_files_found=len(las_files),
        lithology_csv=str(config.data.lithology_csv),
        lithology_csv_exists=Path(config.data.lithology_csv).exists(),
        stratigraphy_csv=str(config.data.stratigraphy_csv),
        stratigraphy_csv_exists=Path(config.data.stratigraphy_csv).exists(),
    )

    report = DataQualityReport(availability=availability)
    if not availability.has_any_data:
        return report

    resolver = CurveAliasResolver()
    las_by_well_raw: dict[str, LASFile] = {}
    for path in las_files:
        try:
            las = parse_las_file(path, alias_resolver=resolver)
        except LASParseError as e:
            report.las_parse_errors.append((str(path), str(e)))
            continue
        if las.well_raw is None:
            report.las_parse_errors.append((str(path), "parsed OK but WELL id is missing/empty"))
            continue
        if las.well_raw in las_by_well_raw:
            report.warnings.append(
                f"Duplicate WELL id '{las.well_raw}' found in both "
                f"{las_by_well_raw[las.well_raw].path} and {las.path}; keeping the first."
            )
            continue
        las_by_well_raw[las.well_raw] = las
        report.warnings.extend(f"[{path.name}] {w}" for w in las.warnings)
    report.las_files_parsed = las_by_well_raw

    lithology_records, stratigraphy_records = [], []
    if availability.lithology_csv_exists:
        try:
            res = parse_lithology_csv(
                config.data.lithology_csv, core_marker_symbols=tuple(config.alignment.core_marker_symbols)
            )
            lithology_records = res.records
            report.n_lithology_records = len(res.records)
            report.lithology_dropped = res.dropped
            report.lithology_warnings = res.warnings
        except LithoSchemaError as e:
            report.lithology_parse_error = str(e)
    if availability.stratigraphy_csv_exists:
        try:
            res = parse_stratigraphy_csv(config.data.stratigraphy_csv)
            stratigraphy_records = res.records
            report.n_stratigraphy_records = len(res.records)
            report.stratigraphy_dropped = res.dropped
            report.stratigraphy_warnings = res.warnings
        except StratSchemaError as e:
            report.stratigraphy_parse_error = str(e)

    litho_by_well = collections.defaultdict(list)
    for r in lithology_records:
        litho_by_well[r.well_raw].append(r)
    strat_by_well = collections.defaultdict(list)
    for r in stratigraphy_records:
        strat_by_well[r.well_raw].append(r)

    well_match = build_well_match_report(
        las_wells=list(las_by_well_raw.keys()),
        lithology_wells=list(litho_by_well.keys()),
        stratigraphy_wells=list(strat_by_well.keys()),
    )
    report.well_match = well_match

    missing_count = {c: 0 for c in CANONICAL_CURVES if c != "DEPT"}
    litho_class_dist = collections.Counter()
    zone_dist = collections.Counter()
    boundary_dist = collections.Counter()

    for match_key, sources in well_match.matched.items():
        las_raw = sources["las"][0]
        las = las_by_well_raw[las_raw]
        depth = las.curves["DEPT"]

        zone_recs = [r for raw in sources["stratigraphy"] for r in strat_by_well[raw]]
        litho_recs = [r for raw in sources["lithology"] for r in litho_by_well[raw]]

        aligned = align_well(match_key, depth, las.step_actual, zone_recs, litho_recs, config.alignment)
        report.aligned_wells[match_key] = aligned

        n = len(depth)
        report.total_depth_points += n
        for c in missing_count:
            mask = las.missing_mask.get(c)
            missing_count[c] += int(mask.sum()) if mask is not None else n  # curve absent = fully missing

        report.n_zone_overlap_points += aligned.diagnostics.zone.n_conflicting_points
        report.n_zone_out_of_range_intervals += aligned.diagnostics.zone.n_out_of_range
        report.n_lithology_interval_overlap_points += aligned.diagnostics.lithology_interval.n_conflicting_points
        report.n_lithology_out_of_range_intervals += aligned.diagnostics.lithology_interval.n_out_of_range
        report.n_lithology_point_unmatched += aligned.diagnostics.lithology_points.n_unmatched
        report.n_lithology_point_conflicts += aligned.diagnostics.lithology_points.n_conflicting_points

        for v in aligned.lithology_label_raw:
            if v is not None:
                litho_class_dist[v] += 1
        for v in aligned.zone_label:
            if v is not None:
                zone_dist[v] += 1
        for v in aligned.boundary_label:
            key = {0: "negative", BOUNDARY_POSITIVE: "positive", IGNORE_INDEX: "ignored"}.get(int(v), str(v))
            boundary_dist[key] += 1

    report.missing_count = missing_count
    report.missing_fraction = {
        c: (missing_count[c] / report.total_depth_points if report.total_depth_points else 0.0)
        for c in missing_count
    }
    report.lithology_class_distribution = dict(litho_class_dist)
    report.zone_distribution = dict(zone_dist)
    report.boundary_distribution = dict(boundary_dist)
    report.warnings.extend(report.lithology_warnings)
    report.warnings.extend(report.stratigraphy_warnings)

    return report
