"""Centralized mapping between DWD pollen regions, 16 Bundesländer, and neighbors.

Single source of truth used by the ingest services and, later, the ML feature
builders. Keep this module self-contained — no DB, no network.
"""

from __future__ import annotations

from typing import Final

ALL_BUNDESLAENDER: Final[tuple[str, ...]] = (
    "BW", "BY", "BE", "BB", "HB", "HH", "HE", "MV",
    "NI", "NW", "RP", "SL", "SN", "ST", "SH", "TH",
)

BUNDESLAND_NAMES: Final[dict[str, str]] = {
    "BW": "Baden-Württemberg",
    "BY": "Bayern",
    "BE": "Berlin",
    "BB": "Brandenburg",
    "HB": "Bremen",
    "HH": "Hamburg",
    "HE": "Hessen",
    "MV": "Mecklenburg-Vorpommern",
    "NI": "Niedersachsen",
    "NW": "Nordrhein-Westfalen",
    "RP": "Rheinland-Pfalz",
    "SL": "Saarland",
    "SN": "Sachsen",
    "ST": "Sachsen-Anhalt",
    "SH": "Schleswig-Holstein",
    "TH": "Thüringen",
}

STATE_NAME_TO_CODE: Final[dict[str, str]] = {name: code for code, name in BUNDESLAND_NAMES.items()}

# DWD publishes pollen index on 8 super-regions that sometimes group two
# Bundesländer (Niedersachsen/Bremen, Brandenburg/Berlin, RP/SL, SH/HH).
# A DWD payload entry for a grouped region fans out to each constituent
# Bundesland code; the index value is identical for each member state.
DWD_POLLEN_REGION_TO_CODES: Final[dict[str, tuple[str, ...]]] = {
    "baden-württemberg": ("BW",),
    "bayern": ("BY",),
    "brandenburg und berlin": ("BB", "BE"),
    "mecklenburg-vorpommern": ("MV",),
    "niedersachsen und bremen": ("NI", "HB"),
    "nordrhein-westfalen": ("NW",),
    "rheinland-pfalz und saarland": ("RP", "SL"),
    "sachsen-anhalt": ("ST",),
    "sachsen": ("SN",),
    "schleswig-holstein und hamburg": ("SH", "HH"),
    "thüringen": ("TH",),
    "hessen": ("HE",),
}

# Capital city → Bundesland. Used by the weather ingester to stamp
# region_code on each observation and by feature builders to fetch the
# "capital weather" per state as a proxy for regional conditions.
CAPITAL_TO_CODE: Final[dict[str, str]] = {
    "Stuttgart": "BW",
    "München": "BY",
    "Berlin": "BE",
    "Potsdam": "BB",
    "Bremen": "HB",
    "Hamburg": "HH",
    "Wiesbaden": "HE",
    "Schwerin": "MV",
    "Hannover": "NI",
    "Düsseldorf": "NW",
    "Mainz": "RP",
    "Saarbrücken": "SL",
    "Dresden": "SN",
    "Magdeburg": "ST",
    "Kiel": "SH",
    "Erfurt": "TH",
}

# Geographic neighbors (shared land border). Drives lead/lag features
# across adjacent Bundesländer and the "upstream weather" logic where a
# pollen front blows in from neighboring states.
REGIONAL_NEIGHBORS: Final[dict[str, tuple[str, ...]]] = {
    "BW": ("BY", "HE", "RP"),
    "BY": ("BW", "HE", "TH", "SN"),
    "BE": ("BB",),
    "BB": ("BE", "MV", "NI", "SN", "ST"),
    "HB": ("NI",),
    "HH": ("NI", "SH"),
    "HE": ("BW", "BY", "NI", "NW", "RP", "TH"),
    "MV": ("BB", "NI", "SH"),
    "NI": ("BB", "HB", "HH", "HE", "MV", "NW", "SH", "ST", "TH"),
    "NW": ("HE", "NI", "RP"),
    "RP": ("BW", "HE", "NW", "SL"),
    "SL": ("RP",),
    "SN": ("BB", "BY", "ST", "TH"),
    "ST": ("BB", "NI", "SN", "TH"),
    "SH": ("HH", "MV", "NI"),
    "TH": ("BY", "HE", "NI", "SN", "ST"),
}


def dwd_region_to_codes(region_name: str) -> tuple[str, ...]:
    """Normalize a DWD pollen region label and return its Bundesland codes.

    Returns an empty tuple if the label does not match any known region. The
    match is case-insensitive and does not rely on exact punctuation, so
    variants like "Niedersachsen/Bremen" and "Niedersachsen und Bremen" both
    resolve correctly.
    """
    normalized = (region_name or "").strip().lower()
    if not normalized:
        return ()
    normalized = normalized.replace("/", " und ").replace(",", " und ")
    normalized = " ".join(normalized.split())
    for key, codes in DWD_POLLEN_REGION_TO_CODES.items():
        if key in normalized:
            return codes
    # Some payloads emit single-state labels that we haven't enumerated above.
    for code, name in BUNDESLAND_NAMES.items():
        if name.lower() in normalized:
            return (code,)
    return ()


def normalize_state_code(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = str(value).strip()
    if cleaned in BUNDESLAND_NAMES:
        return cleaned
    return STATE_NAME_TO_CODE.get(cleaned)
