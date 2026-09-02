import pytest

from lithology.io.las_parser import LASParseError, parse_las_file

STANDARD_LAS = """~Version
VERS              .         2.0
WRAP              .         NO
DLM               .         SPACE

~Well Information
STRT               .M        5.6
STOP               .M        6.0
STEP               .M        0.1
NULL               .         -9999
WELL               .         2-12-27

~Curve Information
DEPT               .M       : DEPTH
GK                 .uR/h    : GK
KS                 .ohmm    : KS
PS                 .mV      : PS

~Ascii
5.6        -9999      -9999      -0.1479
5.7        10.2       15.5      -0.1648
5.8        -9999      15.9      -0.1941
5.9        11.1       16.0      -0.2001
6.0        11.5       -9999      -0.2100
"""

RUSSIAN_ALIAS_LAS = """~Version
VERS. 2.0
WRAP. NO

~Well Information
STRT.M 0.0
STOP.M 0.4
STEP.M 0.1
NULL. -9999
WELL. 2006

~Curve Information
DEPT.M : DEPTH
ГК.uR/h : GAMMA
КС.ohmm : RESISTIVITY
ПС.mV : SP

~Ascii
0.0 5.0 20.0 -0.1
0.1 5.1 20.1 -0.1
0.2 -9999 20.2 -0.1
0.3 5.3 20.3 -0.1
0.4 5.4 20.4 -0.1
"""


def test_parse_standard_las(tmp_path):
    p = tmp_path / "well.las"
    p.write_text(STANDARD_LAS)
    las = parse_las_file(p)

    assert las.well_raw == "2-12-27"
    assert las.step_header == pytest.approx(0.1)
    assert set(las.curves.keys()) == {"DEPT", "GK", "KS", "PS"}
    assert las.n_points == 5

    import numpy as np
    assert np.isnan(las.curves["GK"][0])   # -9999 -> NaN
    assert not np.isnan(las.curves["GK"][1])
    assert las.missing_mask["GK"].sum() == 2
    assert las.missing_mask["KS"].sum() == 2


def test_parse_russian_alias_and_compact_header(tmp_path):
    p = tmp_path / "well.las"
    p.write_text(RUSSIAN_ALIAS_LAS)
    las = parse_las_file(p)

    assert las.well_raw == "2006"
    assert set(las.curves.keys()) == {"DEPT", "GK", "KS", "PS"}
    assert las.missing_mask["GK"].sum() == 1


def test_missing_curve_section_raises(tmp_path):
    p = tmp_path / "bad.las"
    p.write_text("~Well Information\nWELL. X\n")
    with pytest.raises(LASParseError):
        parse_las_file(p)


def test_missing_dept_curve_raises(tmp_path):
    text = """~Well Information
STRT.M 0
STOP.M 1
STEP.M 0.1
NULL. -9999
WELL. X

~Curve Information
GK.uR/h : GK

~Ascii
5.0
5.1
"""
    p = tmp_path / "nodepth.las"
    p.write_text(text)
    with pytest.raises(LASParseError):
        parse_las_file(p)
