# Changelog

Alle relevanten Änderungen am PollenCast-Projekt werden hier dokumentiert. Format orientiert sich an [Keep a Changelog](https://keepachangelog.com/de/1.1.0/).

## [Unreleased]

### Added
- Initiales Repo-Scaffolding: FastAPI-Backend-Skelett, schlankes SQLAlchemy-Modell (`pollen_data`, `pollen_forecast`, `weather_data`, `school_holidays`, `backtest_run`, `backtest_point`, `forecast_accuracy_log`, `source_nowcast_snapshot`, `user`, `audit_log`, `upload_history`), initiale Alembic-Migration `0001_initial_schema`.
- Core-Infrastruktur aus ViralFlux übernommen und bereinigt: Config, Security/JWT, Logging, Rate Limiting, Metrics, Audit, Auth, M2M-Auth.
