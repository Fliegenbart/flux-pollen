# Changelog

Alle relevanten Änderungen am PollenCast-Projekt werden hier dokumentiert. Format orientiert sich an [Keep a Changelog](https://keepachangelog.com/de/1.1.0/).

## [Unreleased]

### Added
- Phase 3 — Datenquellen-Ingestion auf Produktniveau:
  - `region_mapping.py` mit zentraler DWD-Region → Bundesland-Abbildung (inkl. 1:n-Splits für "Niedersachsen und Bremen" etc.), Hauptstadt → Bundesland-Lookup und symmetrischer Nachbarschafts-Topologie für regionale Lead/Lag-Features.
  - `pollen_service.py` gehärtet: tenacity-Retry mit Exponential-Backoff, dialect-aware Upsert (PostgreSQL `ON CONFLICT` / SQLite-Fallback), Point-in-Time-`available_time` bei jedem Ingest, robuster Parser für DWD-Index-Tokens (`0`, `1-2`, `keine`, …), optionaler `import_from_file()` für Fixtures und archivierte Snapshots.
  - `weather_service.py` via BrightSky (kostenlos, kein API-Key, DWD-Backend): Current-Observation + 7-Tage-Backfill + MOSMIX-8-Tage-Forecast, stempelt `region_code` pro Hauptstadt, idempotenter Upsert nach `(city, datum, data_type[, forecast_run_id])`, tenacity-Retry.
  - `holidays_service.py` via schulferien-api.de v2: idempotentes Upsert pro `(bundesland, ferien_typ, jahr, start_datum)`, Revision-Updates auf `end_datum`.
  - CLI-Runner: `scripts/run_pollen_ingest.py`, `scripts/run_weather_ingest.py`, `scripts/run_holidays_ingest.py`. Jeweils mit JSON-Output und Exitcodes.
- Tests (19 grün via `pytest`): Region-Mapping (inkl. Symmetrie-Check der Nachbarschaften), Pollen-Idempotenz + Fixture-basierte End-to-End-Ingestion, Holidays-Service-Idempotenz, Security-Hashing. In-Memory-SQLite-Fixture in `conftest.py`.
- Initiales Repo-Scaffolding: FastAPI-Backend-Skelett, schlankes SQLAlchemy-Modell (`pollen_data`, `pollen_forecast`, `weather_data`, `school_holidays`, `backtest_run`, `backtest_point`, `forecast_accuracy_log`, `source_nowcast_snapshot`, `user`, `audit_log`, `upload_history`), initiale Alembic-Migration `0001_initial_schema`.
- Core-Infrastruktur aus ViralFlux übernommen und bereinigt: Config, Security/JWT, Logging, Rate Limiting, Metrics, Audit, Auth, M2M-Auth.

### Known Limitations
- Der DWD-Alert-Endpunkt `s31fg.json` liefert nur die **aktuellen** Indexwerte (heute + morgen + übermorgen); er ist keine Historie-API. Eine mehrjährige Backtest-Historie entsteht entweder (a) durch tägliche Ingestion über mehrere Saisons, oder (b) durch einen separaten Ingest archivierter Snapshots via `PollenService.import_from_file(...)`. Eine Archiv-Bootstrap-Quelle ist für Phase 4 offen — vor dem ersten belastbaren Backtest muss das geklärt sein.
