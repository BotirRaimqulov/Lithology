# Lithology — Well-Log Lithology & Stratigraphic-Zone Learning Pipeline

A production-quality pipeline for learning geological lithology and
stratigraphic zones from well-log LAS files and expert-corrected CSV
interpretation data, using a multi-task 1D-ResNet sequence model
(lithology + zone + boundary-detection heads).

## Status

**Stage 1 (this codebase): pipeline infrastructure is complete and tested.**
**Stage 2: waiting on real data.**

No real LAS files or CSVs exist in this repository yet — only the schema
described in the task spec and the screenshots the project owner shared.
Every component below was built against that schema and validated with
synthetic fixtures (see `tests/`), but **no model has been trained** and
**no real dataset statistics are reported anywhere** in this repo. Numbers
you see in commit history/tests are from synthetic smoke tests only.

To start Stage 2, drop real files into `data/las/*.las`,
`data/csv/lithology.csv`, `data/csv/stratigraphy.csv`, then run:

```bash
pip install -r requirements.txt
python tools/inspect_dataset.py
```

This prints the full Phase-1 report (curve names found, aliases resolved,
well matching, missing-value stats, interval diagnostics, class
distributions) against your real files, with **zero fabricated numbers** —
if the directories are still empty it says so explicitly instead of
printing misleading zeros.

## Pipeline stages

```
tools/inspect_dataset.py   -> data-quality report only, no files written
tools/build_dataset.py     -> exports dataset/{train,val,test}/*.parquet + metadata/
tools/train.py              -> trains the multi-task model on the exported dataset
tools/evaluate.py           -> metrics + interval reconstruction on a held-out split
tools/visualize_well.py     -> GK/KS/PS + expert/predicted lithology/zone/boundary plot
```

`build_dataset.py` refuses to export (non-zero exit) if
`DataQualityReport.is_valid_for_training()` finds the data fundamentally
broken (no matched wells, no labels assigned, etc.) — see spec section 17.

## Directory layout

```
configs/default.yaml     All tunable parameters (spec section 18). No path
                          or hyperparameter is hard-coded anywhere else.
data/las/                 Drop real .las files here (or point config elsewhere).
data/csv/                 Drop lithology.csv / stratigraphy.csv here.
lithology/                The package.
  io/                      LAS parsing, curve-alias resolution, CSV parsing.
  wells/                   Well-ID normalization & cross-source matching.
  alignment/               Interval -> depth-point label alignment.
  quality/                 End-to-end data-quality report.
  features/                Per-well feature engineering.
  dataset/                 Group-by-well split, windowing, Parquet/NPZ export.
  models/                  1D ResNet encoder, sequence encoder, multi-task heads, losses.
  training/                Train/val loop, metrics.
  postprocess/             Point predictions -> continuous interval reconstruction.
  viz/                     Well-log visualization.
tools/                     Thin CLI wrappers around the package (see above).
tests/                     pytest unit/integration tests (46 passing), synthetic fixtures only.
outputs/                   Generated: dataset export, experiment runs, reports (gitignored).
```

## Key design decisions (and why)

- **Lithology is point-based, not interval-based, by default.** The
  lithology CSV is one row per core/lab sample at a specific `MD`, not a
  top/bottom interval — per the project owner's explicit instruction, a
  single lab sample must label only its own depth, never be spread across
  a surrounding interval. If a real file instead provides `top`/`bottom`
  columns, the parser detects that schema automatically and treats it as
  an interval (see `io/lithology_csv.py`).
- **The `*` core/lab marker is preserved, not interpreted.** It is
  stripped from the code value and recorded as `lithology_core_verified`
  + `lithology_confidence`; `alignment.require_core_verified_for_lithology`
  (config) controls whether non-core-verified points are excluded from
  supervision. Its exact real-world meaning/placement will be confirmed
  once real files are inspected.
- **Overlapping/duplicate/out-of-range intervals are never silently
  resolved.** A depth point claimed by two conflicting intervals gets
  `None` (→ `IGNORE_INDEX`), not an arbitrary pick — see
  `alignment/intervals.py` and the diagnostics counters in the quality report.
- **Well-ID matching only merges on exact-after-normalization or numeric
  equality.** Field-style IDs (`2-12-27`) are never guessed to equal a
  sequential number (`2006`) — low-confidence heuristic matches are
  reported separately (`weak_suggestions`) and never auto-applied.
- **No raw depth is fed to the model** — only curve values, their missing
  masks, derivatives, rolling statistics, and cross-curve rolling
  correlation. This forces the network to learn curve *behavior* rather
  than memorizing "this depth number means this zone" (spec section 6).
- **The 1D ResNet *is* the CNN** — there's no separate plain-CNN stage in
  front of it. It never downsamples (dense per-point prediction), and its
  receptive field grows via dilated convolutions
  (`models/resnet1d.receptive_field_points`) rather than one huge kernel.
  `sampling.context_size`/`context_unit` (meters or points, spec section 7-8)
  is convertible to a point radius via `features.engineering.resolve_context_points`
  and used to size training crops (`dataset/windowing.py`) generously
  larger than the receptive field.
- **Normalization is fit on train wells only**, applied identically to
  val/test, and the fitted stats are saved to `metadata/normalization.json`.
  Label vocabularies (lithology codes, zone names), by contrast, are built
  across all wells — a categorical vocabulary isn't a leaking statistic
  the way a feature mean/std is, and it must cover val/test labels for
  them to be scoreable at all.
- **Splitting is always by well**, never by row (`dataset/split.py`). Below
  `split.min_wells_for_holdout_split` wells, grouped k-fold cross-validation
  is used instead of a single holdout split.
- **Boundary detection is a learned head**, not a GK/KS-jump rule — trained
  from expert top/bottom transitions with a configurable tolerance band
  (`alignment.boundary_tolerance_points`), and excluded (`IGNORE_INDEX`)
  wherever there is no expert coverage at all.

## Running the tests

```bash
pip install -r requirements.txt
python -m pytest tests/ -q
```

All 46 tests use synthetic fixtures generated inline — none depend on real
project data, so they exercise every parsing/alignment/model code path
today, before real files exist.

## Configuration

Everything tunable lives in `configs/default.yaml` (data paths, context
size/unit, interval semantics, feature toggles, model architecture, loss
weights, split strategy, normalization method, augmentation). See
`lithology/config.py` for the full documented schema and defaults.
