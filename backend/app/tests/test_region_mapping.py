from app.services.data_ingest.region_mapping import (
    ALL_BUNDESLAENDER,
    CAPITAL_TO_CODE,
    REGIONAL_NEIGHBORS,
    dwd_region_to_codes,
    normalize_state_code,
)


def test_sixteen_bundeslaender():
    assert len(ALL_BUNDESLAENDER) == 16
    assert set(ALL_BUNDESLAENDER) == set(REGIONAL_NEIGHBORS.keys())
    assert set(CAPITAL_TO_CODE.values()) == set(ALL_BUNDESLAENDER)


def test_dwd_region_grouped_label_maps_to_both_states():
    assert set(dwd_region_to_codes("Niedersachsen und Bremen")) == {"NI", "HB"}
    assert set(dwd_region_to_codes("Brandenburg und Berlin")) == {"BB", "BE"}
    assert set(dwd_region_to_codes("Schleswig-Holstein und Hamburg")) == {"SH", "HH"}
    assert set(dwd_region_to_codes("Rheinland-Pfalz und Saarland")) == {"RP", "SL"}


def test_dwd_region_single_label_maps_to_single_state():
    assert dwd_region_to_codes("Bayern") == ("BY",)
    assert dwd_region_to_codes("Sachsen") == ("SN",)
    assert dwd_region_to_codes("Sachsen-Anhalt") == ("ST",)
    assert dwd_region_to_codes("Hessen") == ("HE",)
    assert dwd_region_to_codes("Thüringen") == ("TH",)


def test_dwd_region_handles_punctuation_variants():
    assert set(dwd_region_to_codes("Niedersachsen/Bremen")) == {"NI", "HB"}
    assert set(dwd_region_to_codes(" niedersachsen  UND bremen ")) == {"NI", "HB"}
    assert dwd_region_to_codes("") == ()
    assert dwd_region_to_codes(None) == ()
    assert dwd_region_to_codes("Österreich") == ()


def test_neighbors_are_symmetric():
    """If A is a neighbor of B, B must be a neighbor of A — else we silently
    drop lead/lag coupling when the model looks it up in the wrong direction.
    """
    for state, neighbors in REGIONAL_NEIGHBORS.items():
        for neighbor in neighbors:
            assert state in REGIONAL_NEIGHBORS[neighbor], (
                f"{neighbor} lists {state}? {state in REGIONAL_NEIGHBORS[neighbor]}; "
                f"{state} lists {neighbor}? {neighbor in neighbors}"
            )


def test_normalize_state_code_by_code_and_name():
    assert normalize_state_code("NW") == "NW"
    assert normalize_state_code("Nordrhein-Westfalen") == "NW"
    assert normalize_state_code("does not exist") is None
    assert normalize_state_code(None) is None
