"""Response schemas for the pollen API."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class PollenCurrentResponse(BaseModel):
    pollen_type: str
    region_code: str
    datum: datetime
    concentration: float | None = Field(
        default=None,
        description="Daily mean concentration in grains/m³ (from pollen_observations).",
    )
    pollen_index: float | None = Field(
        default=None,
        description="DWD danger index 0–3 (from pollen_data) when ePIN data is absent.",
    )
    available_time: datetime | None = None
    source: str = Field(description="'ePIN' or 'DWD'.")

    model_config = ConfigDict(from_attributes=True)


class PollenForecastPoint(BaseModel):
    target_date: datetime
    horizon_days: int
    predicted_concentration: float
    lower_bound: float
    upper_bound: float
    confidence_label: str


class PollenForecastResponse(BaseModel):
    pollen_type: str
    region_code: str
    forecast_date: datetime
    horizon_days: int
    model_version: str
    trained_at: str
    forecast: PollenForecastPoint


class RegionalRankingEntry(BaseModel):
    region_code: str
    region_name: str
    predicted_concentration: float
    lower_bound: float
    upper_bound: float
    rank: int


class RegionalRankingResponse(BaseModel):
    pollen_type: str
    horizon_days: int
    forecast_date: datetime
    entries: list[RegionalRankingEntry]
