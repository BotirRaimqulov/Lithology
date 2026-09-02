from lithology.io.lithology_csv import parse_lithology_csv
from lithology.io.stratigraphy_csv import parse_stratigraphy_csv


def test_lithology_point_schema_with_marker(tmp_path):
    p = tmp_path / "lithology.csv"
    p.write_text(
        "Well number,MD,kod,Dataset name\n"
        "2006,305,5*,Lithology\n"
        "2006,308.6,4,Lithology\n"
        "2007,308,5,Lithology\n"
        "2007,,4,Lithology\n"          # dropped: missing MD
        "2007,327,,Lithology\n"        # dropped: missing code
    )
    res = parse_lithology_csv(str(p))
    assert res.schema == "point"
    assert len(res.records) == 3
    assert len(res.dropped) == 2

    r0 = res.records[0]
    assert r0.is_point_sample is True
    assert r0.top == r0.bottom == 305.0
    assert r0.code == "5"
    assert r0.core_verified is True

    r1 = res.records[1]
    assert r1.core_verified is False


def test_lithology_interval_schema(tmp_path):
    p = tmp_path / "lithology.csv"
    p.write_text(
        "Well,top,bottom,kod\n"
        "2006,300,310,5\n"
        "2006,320,310,6\n"   # dropped: bottom < top
    )
    res = parse_lithology_csv(str(p))
    assert res.schema == "interval"
    assert len(res.records) == 1
    assert res.records[0].is_point_sample is False


def test_stratigraphy_parsing_and_thickness_mismatch(tmp_path):
    p = tmp_path / "strat.csv"
    p.write_text(
        "Well,Zone name,top,bottom,Thickness\n"
        "2006,Q4,0,7,7\n"
        "2006,N1/2-3,7,57,999\n"     # declared thickness disagrees with computed
        "2006,BAD,10,5,-5\n"          # dropped: bottom < top
    )
    res = parse_stratigraphy_csv(str(p))
    assert len(res.records) == 2
    assert len(res.dropped) == 1
    assert res.records[1].thickness_declared == 999
    assert res.records[1].thickness_computed == 50
    assert any("disagrees" in w for w in res.warnings)


def test_inconsistent_zone_name_spacing_merges_into_one_class(tmp_path):
    # Real data mixes "P 3-N1/1" and "P3-N1/1" for the same geological unit
    # (an interior-space data-entry inconsistency, not a real distinction).
    p = tmp_path / "strat.csv"
    p.write_text(
        "Well,Zone name,top,bottom\n"
        "2006,P 3-N1/1,0,10\n"
        "2006,P3-N1/1,10,20\n"
    )
    res = parse_stratigraphy_csv(str(p))
    zone_names = {r.zone_name for r in res.records}
    assert zone_names == {"P3-N1/1"}
    assert any("whitespace stripped" in w for w in res.warnings)
