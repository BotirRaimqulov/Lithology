from lithology.wells.well_id import build_well_match_report, normalize_well_id


def test_normalize_variants():
    assert normalize_well_id("2006.0").numeric_id == 2006
    assert normalize_well_id("Well-2006").numeric_id == 2006
    assert normalize_well_id("2006").match_key == normalize_well_id("Well-2006").match_key
    assert normalize_well_id("2-12-27").numeric_id is None  # never guessed as numeric


def test_matching_never_merges_different_field_style_ids():
    report = build_well_match_report(
        las_wells=["2-12-27"], lithology_wells=["2006"], stratigraphy_wells=["2006"],
    )
    assert report.matched == {}
    assert report.las_without_labels == ["2-12-27"]
    assert ("lithology", "2006") in report.labels_without_las
    assert ("stratigraphy", "2006") in report.labels_without_las


def test_matching_handles_numeric_and_prefix_variants():
    report = build_well_match_report(
        las_wells=["2006", "Well-2007", "2041.0"],
        lithology_wells=["2006", "2008"],
        stratigraphy_wells=["2006", "2007", "2041"],
    )
    assert set(report.matched.keys()) == {"2006", "2007", "2041"}
    assert report.las_without_labels == []
    assert report.labels_without_las == [("lithology", "2008")]


def test_weak_suggestions_are_never_auto_merged():
    report = build_well_match_report(
        las_wells=["A-1-2008"], lithology_wells=["2008"], stratigraphy_wells=[],
    )
    assert report.matched == {}
    assert len(report.weak_suggestions) == 1
    assert report.weak_suggestions[0][:3] == ("A-1-2008", "lithology", "2008")
