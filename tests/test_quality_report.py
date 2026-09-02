from lithology.config import Config
from lithology.quality.report import _bucket_las_file_warnings, build_data_quality_report

LAS_TEMPLATE = """~Version
VERS. 2.0
WRAP. NO

~Well Information
STRT.M 0.0
STOP.M {stop}
STEP.M 0.1
NULL. -9999
WELL. {well}

~Curve Information
DEPT.M : DEPTH
GK.uR/h : GK
KS.ohmm : KS
PS.mV : PS

~Ascii
{rows}
"""


def _write_las(dirpath, well, n=20, step=0.1):
    import numpy as np
    depth = np.round(np.arange(0, n * step, step), 3)
    rows = "\n".join(f"{d:.3f} {5+d:.3f} {20+d:.3f} {-0.1:.3f}" for d in depth)
    (dirpath / f"{well}.las").write_text(LAS_TEMPLATE.format(stop=depth[-1], well=well, rows=rows))
    return depth


def test_no_data_found_reports_clearly(tmp_path):
    cfg = Config()
    cfg.data.las_dir = str(tmp_path / "nope")
    cfg.data.lithology_csv = str(tmp_path / "nope.csv")
    cfg.data.stratigraphy_csv = str(tmp_path / "nope2.csv")
    report = build_data_quality_report(cfg)
    assert not report.availability.has_any_data
    assert "No real dataset found" in report.to_text()
    valid, reasons = report.is_valid_for_training()
    assert not valid


def test_synthetic_end_to_end_report(tmp_path):
    las_dir = tmp_path / "las"
    las_dir.mkdir()
    depth = _write_las(las_dir, "2006", n=20)

    litho_csv = tmp_path / "lithology.csv"
    litho_csv.write_text(
        "Well number,MD,kod,Dataset name\n"
        f"2006,{depth[3]},5*,Lithology\n"
        f"2006,{depth[15]},4,Lithology\n"
    )
    strat_csv = tmp_path / "stratigraphy.csv"
    strat_csv.write_text(
        "Well,Zone name,top,bottom,Thickness\n"
        f"2006,Q4,0,{depth[10]},{depth[10]}\n"
        f"2006,N1,{depth[10]},{depth[-1]},{depth[-1]-depth[10]}\n"
    )

    cfg = Config()
    cfg.data.las_dir = str(las_dir)
    cfg.data.lithology_csv = str(litho_csv)
    cfg.data.stratigraphy_csv = str(strat_csv)

    report = build_data_quality_report(cfg)
    assert report.availability.has_any_data
    assert report.well_match.summary()["n_matched"] == 1
    assert report.total_depth_points == 20
    assert report.lithology_class_distribution == {"5": 1, "4": 1}
    assert set(report.zone_distribution.keys()) == {"Q4", "N1"}
    valid, reasons = report.is_valid_for_training()
    assert valid, reasons


def test_las_warnings_are_bucketed_not_repeated_per_file():
    warnings = []
    for i in range(49):
        fn = f"{2000 + i}.las"
        if i % 4 != 0:
            warnings.append((fn, "File decoded with fallback encoding 'cp1251' (utf-8 failed)."))
        # per-curve missing-value notes are redundant with the aggregate
        # "Depth points / missing values" section and must be dropped entirely
        warnings.append((fn, f"Curve 'GK' (raw 'GK'): {20 + i}/3600 points (0.6%) are NULL -> NaN."))

    buckets = _bucket_las_file_warnings(warnings)
    assert len(buckets) == 1  # only the encoding-fallback bucket survives
    shape, entry = buckets[0]
    assert "utf-#" in shape  # hyphen preserved, not swallowed by number-collapsing
    assert entry["count"] == 36  # 49 - ceil(49/4) files hit the i % 4 != 0 branch
    assert len(entry["examples"]) <= 3


def test_restricted_to_excludes_wells_from_split_safety_net(tmp_path):
    # Regression test: tools/learning_curve.py needs to train on a strict
    # SUBSET of the available training wells while keeping val/test fixed.
    # Without restricted_to(), split_wells()'s "never silently drop a well"
    # safety net would treat every well outside the explicit train/val/test
    # lists as an accidental omission and re-add it to train, silently
    # defeating the deliberate subset.
    from lithology.config import SplitConfig
    from lithology.dataset.split import split_wells

    las_dir = tmp_path / "las"
    las_dir.mkdir()
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()

    wells = [f"W{i}" for i in range(6)]
    litho_rows = ["Well,top,bottom,kod"]
    strat_rows = ["Well,Zone Name,top,bottom,Thickness"]
    for w in wells:
        depth = _write_las(las_dir, w, n=20)
        litho_rows.append(f"{w},0,{depth[-1]},4")
        strat_rows.append(f"{w},Q4,0,{depth[-1]},{depth[-1]}")
    (csv_dir / "lithology.csv").write_text("\n".join(litho_rows))
    (csv_dir / "stratigraphy.csv").write_text("\n".join(strat_rows))

    cfg = Config()
    cfg.data.las_dir = str(las_dir)
    cfg.data.lithology_csv = str(csv_dir / "lithology.csv")
    cfg.data.stratigraphy_csv = str(csv_dir / "stratigraphy.csv")
    report = build_data_quality_report(cfg)
    assert set(report.aligned_wells.keys()) == set(wells)

    # Deliberately restrict to only 2 of the 4 "train" wells, keeping the
    # other 2 as val/test.
    subset_train, val_wells, test_wells = ["W0", "W1"], ["W2"], ["W3"]
    restricted = report.restricted_to(subset_train + val_wells + test_wells)
    assert set(restricted.aligned_wells.keys()) == {"W0", "W1", "W2", "W3"}

    split_cfg = SplitConfig(train_wells=subset_train, val_wells=val_wells, test_wells=test_wells)
    result = split_wells(list(restricted.aligned_wells.keys()), split_cfg)
    assert set(result.train) == {"W0", "W1"}  # W4/W5 (never mentioned) must NOT reappear here
