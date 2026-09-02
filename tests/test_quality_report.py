from lithology.config import Config
from lithology.quality.report import build_data_quality_report

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
