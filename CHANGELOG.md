# Changelog

Alle relevanten Änderungen am PollenCast-Projekt werden hier dokumentiert. Format orientiert sich an [Keep a Changelog](https://keepachangelog.com/de/1.1.0/).

## [Unreleased]

### Added
- **Phase 4a — ML-Fundament (Feature-Engineering, Baselines, Ridge-+-GBM-Forecast, Walk-Forward-Backtester).** Erste Version der ML-Pipeline ist lauffähig, inkl. probabilistischer Prognose und harter Baselines:
  - `app/services/ml/metrics.py` — MAE, RMSE, Pinball-Loss, Interval-Score, **Weighted Interval Score (WIS)** nach Bracher et al. 2021 (die Metrik, die RKI/CDC/Hubverse für Respiratory-Forecasts verwenden), Coverage.
  - `app/services/ml/baselines.py` — PersistenceBaseline und SeasonalNaiveBaseline mit empirisch kalibrierten Quantil-Intervallen (q10/q90).
  - `app/services/ml/feature_engineering.py` — daily panel aus `pollen_observations` (Bayern, ePIN-Tagesmittel) + `weather_data` + `school_holidays` mit vier Feature-Familien: Target-Lags/Rollfenster (1/2/3/5/7 Tage; 3/7-Tage-Rolling-Mean + Slope), Wetter (Temperatur, Luftfeuchtigkeit, Wind, Regen, Lags 0/1/3), Kalender (zyklisch kodierter Tag-des-Jahres als sin/cos, Weekend-Flag, Holiday-Fraktion), Cross-Pollen-Lags entlang der bio-chronologischen Kette (Hasel→Erle→Birke→Gräser, 7/14-Tage-Lags).
  - `app/services/ml/forecast_service.py` — Ridge (standardisiert) für den Median + zwei `GradientBoostingRegressor(loss="quantile")` für lower/upper Bound. Bounds werden auf Monotonie und Nicht-Negativität normalisiert. Direct-Horizon-Strategie (kein iteriertes Forecasting).
  - `app/services/ml/backtester.py` — Walk-Forward mit Train-Grow-Window und 1-Tages-Schrittweite, pro Fold: Modell + Persistence + Seasonal-Naive → WIS80, MAE, RMSE, Pinball@0.1/0.5/0.9, Coverage80. `persist_backtest_run()` schreibt `BacktestRun` + `BacktestPoint` und die relativen WIS-Verbesserungen vs. Baseline.
  - `scripts/run_backtest.py` — CLI, persistiert den Run automatisch in die DB.
  - 16 neue Tests (`test_ml_metrics.py`, `test_ml_baselines.py`, `test_ml_backtest_e2e.py`): Metrik-Identitäten (Pinball@0.5 = ½·MAE, WIS-Degenerationen), Baseline-Semantik, End-to-End-Backtest auf 2 Jahren synthetischer Birkenpollen-Saison mit realistischer Wetter-Kopplung. Gesamt-Testsuite: **41 grün in 19,7 s**.
  - Sanity-Check auf synthetischen Daten: Ridge-+-GBM **reduziert WIS80 gegenüber Persistence um 52,8 %** und gegenüber Seasonal-Naive um 59,8 % (43 Folds, Horizon 7 Tage). Coverage80 liegt bei 0,88 (Ziel 0,80) — leicht überkonservativ.
- **Phase 3b — ePIN Bayern als primäre Historie-Quelle.** Nach einer Recherche über alle naheliegenden Pollen-Archivoptionen (DWD-OpenData, DWD CDC, DWD-Pollenflugstatistik-Tool, Wayback Machine, PID Stiftung) war das Ergebnis eindeutig: DWD OpenData hat nur aktuelle Tagesdaten, das DWD-Statistik-Tool liefert nur PNG-Bitmaps, Wayback hatte in 24 Monaten nur ~7 Snapshots, PID kostet über 100 € pro Pollenart × Woche × Station. **ePIN (Bayern)** ist die einzige offene, strukturierte, historisch zurückreichende Quelle und wird zur primären Datenbasis für den ersten belastbaren Backtest.
  - Neue Tabelle `pollen_observations` (station-level, 3-Stunden-Auflösung, Konzentration in Pollen/m³) parallel zu `pollen_data` (DWD-Regional-Index 0–3). Unique-Constraint auf `(station_id, pollen_type, from_time)`.
  - Alembic-Migration `0002_pollen_observations`.
  - Neuer Service `epin_service.py` mit tenacity-Retry, dialect-aware Upsert, `run_full_import()` für fenstergebundene Tageslieferungen und `backfill_range()` für chunked Mehrjahres-Bootstraps. `import_from_file()` für Fixtures und archivierte Snapshots.
  - `region_mapping.py` erweitert um alle 12 ePIN-Stationen (8 Automaten + 4 Hirst-Fallen) und ein Mapping der ePIN-Pollen-Wissenschaftsnamen (Corylus, Alnus, Fraxinus, Betula, Poaceae, Secale, Artemisia, Ambrosia) auf unsere kanonischen Token.
  - CLI `scripts/run_epin_ingest.py` mit `--backfill`-Modus für historische Ingestion.
  - 6 neue Tests in `test_epin_ingest.py` gegen einen realen 48h-Ausschnitt um den Beginn der 2024er-Birkenblüte (DEMUNC, DEHOF × Betula, Poaceae, Alnus). Testsuite-Total: **25 Tests grün in 1,3s**.
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
- Der DWD-Alert-Endpunkt `s31fg.json` liefert nur die **aktuellen** Indexwerte (heute + morgen + übermorgen); er ist keine Historie-API. Für die 11 nicht-bayerischen DWD-Regionen baut sich Historie über tägliche Ingestion auf. Für den Backtest (Phase 4/5) liegt der Fokus deshalb vorerst auf Bayern via ePIN.
- Die Pollendaten aus `pollen_data` (DWD-Index, 0–3, regional) und `pollen_observations` (ePIN-Konzentration, count/m³, station-level) sind bewusst in getrennten Tabellen, weil Einheit und Granularität unterschiedlich sind. Der Feature-Builder entscheidet je nach Bundesland, welche Quelle er konsumiert; eine verlustbehaftete Umrechnung Index↔Konzentration wurde absichtlich nicht eingebaut.
