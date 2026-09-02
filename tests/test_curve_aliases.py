from lithology.io.curve_aliases import CurveAliasResolver


def test_exact_english_mnemonics():
    r = CurveAliasResolver()
    assert r.resolve("GK").canonical == "GK"
    assert r.resolve("KS").canonical == "KS"
    assert r.resolve("PS").canonical == "PS"
    assert r.resolve("DEPT").canonical == "DEPT"


def test_russian_aliases():
    r = CurveAliasResolver()
    assert r.resolve("ГК").canonical == "GK"
    assert r.resolve("КС").canonical == "KS"
    assert r.resolve("ПС").canonical == "PS"


def test_description_fallback():
    r = CurveAliasResolver()
    m = r.resolve("GK1", description="Gamma Ray")
    assert m.canonical == "GK"


def test_unresolved_curve_is_reported_not_guessed():
    r = CurveAliasResolver()
    m = r.resolve("XYZ123", description="Unrelated tool reading")
    assert m.canonical is None
    assert m.method == "unresolved"


def test_extra_aliases_extend_defaults():
    r = CurveAliasResolver(extra_aliases={"GK": ["CUSTOMGK"]})
    assert r.resolve("CustomGK").canonical == "GK"
    # defaults still work
    assert r.resolve("GK").canonical == "GK"
