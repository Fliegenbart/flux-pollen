"""Prometheus metrics for PollenCast backend."""

from prometheus_client import Counter, Histogram, Info

app_info = Info("pollencast", "PollenCast Forecast Engine")

http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

ingestion_records_total = Counter(
    "ingestion_records_total",
    "Total records ingested by source",
    ["source"],
)

ingestion_errors_total = Counter(
    "ingestion_errors_total",
    "Total ingestion errors by source",
    ["source"],
)

forecast_runs_total = Counter(
    "forecast_runs_total",
    "Total forecast runs by pollen_type and region",
    ["pollen_type", "region", "status"],
)
