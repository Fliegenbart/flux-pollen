"""Customer-upload schema for outcome data.

Kept deliberately small: one CSV layout, one set of accepted metrics,
one validation pass. Expansion to additional customers or new metric
families happens by adding rows here, not by branching the code.

Current onboarding target: Hexal (Lorano OTC). The file format below
matches what IQVIA OTCInside-style sell-out exports look like after
you drop the regional-hierarchy columns and keep only the
Bundesland-per-week lines. That is the same extract a Hexal analyst
can produce from their standing subscription without requiring
engineering support on their side — a critical adoption detail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from app.services.data_ingest.region_mapping import ALL_BUNDESLAENDER

# ---- Brand/product taxonomy ------------------------------------------------

HEXAL_BRAND: Final[str] = "hexal"
LORANO_PRODUCT: Final[str] = "lorano_5mg_20stk"

# Every metric we accept maps to a single Bundesland-per-week row.
# ``unit`` is carried through so downstream charts can label axes correctly;
# ``group`` lets the frontend show one tab per family without guessing.
@dataclass(frozen=True)
class MetricDefinition:
    key: str
    label: str
    unit: str
    group: str
    description: str


METRICS: Final[dict[str, MetricDefinition]] = {
    "sell_out_units": MetricDefinition(
        key="sell_out_units",
        label="Sell-Out Einheiten",
        unit="Packungen",
        group="commercial",
        description="Apotheken-Abverkauf in Packungen, Bundesland × Woche.",
    ),
    "sell_out_revenue_eur": MetricDefinition(
        key="sell_out_revenue_eur",
        label="Sell-Out Umsatz",
        unit="€",
        group="commercial",
        description="Apotheken-Abverkauf-Umsatz brutto in Euro.",
    ),
    "tv_grp": MetricDefinition(
        key="tv_grp",
        label="TV-GRPs",
        unit="GRP",
        group="media",
        description="TV-Gross-Rating-Points pro Woche und Region.",
    ),
    "search_brand_clicks": MetricDefinition(
        key="search_brand_clicks",
        label="Markensuche-Klicks",
        unit="Clicks",
        group="media",
        description="Google-Ads- oder SEO-Klicks auf Marken- und Produktterms.",
    ),
    "search_brand_impressions": MetricDefinition(
        key="search_brand_impressions",
        label="Markensuche-Impressionen",
        unit="Impressionen",
        group="media",
        description="Ads-Impressionen auf Marken-Keywords.",
    ),
}

# ---- CSV contract ---------------------------------------------------------

# The uploader hands us a single long-format CSV. One row = one
# (brand, product, region, week_start, metric, value). We do not try
# to auto-detect "wide" formats — forcing the extract into long form is
# the cheapest way to keep validation unambiguous and schema changes
# backwards-compatible.
REQUIRED_CSV_COLUMNS: Final[tuple[str, ...]] = (
    "brand",
    "product",
    "region_code",
    "week_start",
    "metric",
    "value",
)
OPTIONAL_CSV_COLUMNS: Final[tuple[str, ...]] = ("channel", "campaign_id", "unit")

SUPPORTED_METRICS: Final[frozenset[str]] = frozenset(METRICS.keys())
SUPPORTED_REGIONS: Final[frozenset[str]] = frozenset(ALL_BUNDESLAENDER)
