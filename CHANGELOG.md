# Changelog

Alle relevanten Änderungen am PollenCast-Projekt werden hier dokumentiert. Format orientiert sich an [Keep a Changelog](https://keepachangelog.com/de/1.1.0/).

## [Unreleased]

### Added
- **Phase 4e — Stacked+CP-Modelle als API-Default.** Der Trainings-CLI und das Model-Registry sind jetzt für alle drei Forecaster-Varianten (single-stage, stacked, stacked+CP) einheitlich nutzbar; die API spielt ab jetzt die kalibrierten Stacked-Artefakte aus.
  - `scripts/run_train.py` bekommt `--stacked` und `--calibrate` (kombinierbar, analog zu `run_backtest.py`). Das Model-Version-Tag wandert automatisch: `pollencast-ridge-gbm-v0` → `…-v0` ohne CP, `…-cp80-v0` mit CP; stacked analog.
  - `ModelArtifact.service` und `build_metadata`/`load_artifact` haben den Type-Hint auf `object` gelockert, damit beliebige ForecasterProtocol-konforme Objekte (plain, stacked, conformal-wrapped) als Artefakt persistiert werden können. joblib erhält den konkreten Typ zur Laufzeit.
  - Alle 6 Bayern-Modelle (Birke/Gräser/Erle × h7/h14) retrainiert mit `--stacked --calibrate` auf 1 900+ Samples, Trainingsfenster 2021-01-01 bis 2026-04-17.
  - Smoke-Test gegen `localhost:8001` bestätigt, dass die API jetzt `pollencast-stacked-hw-ridge-xgb-cp80-v0` ausspielt und plausible, breiten-korrekte Bänder liefert — z. B. Birke h7 am 2026-04-09 für 2026-04-16: predicted 59 Pollen/m³, Band 2–1337, `confidence_label=low` (Peak-Phase, hohe Unsicherheit). Erle h7 zeigt 15 [6, 48] — spätes Saisonende, enges Band.

- **Phase 4d — Split-Conformal-Kalibrierung.** Der Coverage-Overshoot der Stacked-Quantile (0,58–0,66 bei 0,80-Zielband) wird mit einem transparenten, theoretisch abgesicherten Post-Processing-Wrapper korrigiert.
  - `app/services/ml/conformal_calibrator.py` — generische `ConformalCalibratedForecaster`-Klasse, umschließt jeden `ForecasterProtocol`. Teilt die Trainingsreihe zeitgeordnet in Fit-Split (80 %) und Calibration-Split (20 %), berechnet die Nicht-Konformitätsscores `max(lower − y, y − upper, 0)` und deren `⌈(n+1)α⌉/n`-Quantil als Band-Erweiterung, fit'tet die Basis anschließend auf der vollen Historie. Funktioniert für den einstufigen `ForecastService` genauso wie für `StackedForecastService`.
  - `scripts/run_backtest.py --calibrate` aktiviert den Wrapper; kombinierbar mit `--stacked`. Modell-Version-Tag wandert zu `pollencast-stacked-hw-ridge-xgb-cp80-v0`.
  - 5 neue Tests in `test_ml_conformal.py` (monoton widening, Kalibrier-Summary, Shape-Stabilität, Parameter-Validierung, Insufficient-Training-Refusal). Gesamt-Testsuite: **56 grün in 25,2 s**.
  - **Coverage auf 5 Jahren Real-ePIN-Bayern-Daten, 220 Folds je Scope:**

  | Pollen × H | single | stacked | **stacked+CP** | Ziel |
  |---|---:|---:|---:|---:|
  | Birke 7    | 0.90 | 0.58 | **0.83** | 0.80 |
  | Birke 14   | 0.91 | 0.58 | **0.82** | 0.80 |
  | Gräser 7   | 0.91 | 0.62 | **0.80** | 0.80 |
  | Gräser 14  | 0.86 | 0.66 | **0.82** | 0.80 |
  | Erle 7     | 0.85 | 0.65 | **0.86** | 0.80 |

  - **MAE bleibt unverändert** (CP ist punktschätzungsneutral). **WIS80 steigt nur um 1–11 %** (Kosten der erlaubten Band-Erweiterung). Die WIS-Verbesserung vs. Baselines bleibt stark: **+40 bis +55 % vs. Persistence**, **+30 bis +56 % vs. Seasonal-Naive** — bei jetzt *ehrlich* kalibriertem 80-%-Band.
  - Die Kombination **Stacking + Conformal** ist das vorzeigbare Default-Modell. Einstufiger ForecastService bleibt als Fallback für Scopes mit wenig Trainingsdaten bestehen.

- **Phase 4b — Stacking (Holt-Winters + Ridge → XGBoost-Meta-Learner).** Die ursprünglich als "Nachkomma-Optimierung" klassifizierten Stacking-Verbesserungen erweisen sich auf realen Daten als der entscheidende Hebel.
  - `app/services/ml/holt_winters_forecaster.py` — statsmodels-basierte HW-Komponente mit `fit(X, y) / predict(X)`-Interface. Saisonal (additiv, Periode 365) wenn ≥2 volle Zyklen vorhanden, sonst trend-only-Fallback.
  - `app/services/ml/stacked_forecast_service.py` — Drop-in-Ersatz für `ForecastService` mit identischer Output-Signatur. Time-split-Stacking: Base-Estimatoren Ridge + HW werden auf den ersten 70 % der Trainingsreihe gefittet, der XGBoost-Meta-Learner (3 Modelle für q=0.1/0.5/0.9, Ziel `reg:quantileerror`) lernt auf den Base-Predictions + Originalfeatures der restlichen 30 %. Nach Meta-Training werden die Base-Estimatoren auf der vollen Historie refit'et, damit Inferenz-Zeit die maximale Information nutzt. HW-Meta-Predictions via einem einzigen Multi-Step-Forecast (Kosten: ein HW-Fit pro Fold statt ein Fit pro Meta-Row).
  - Backtester hat jetzt einen `forecaster_factory`-Hook; `scripts/run_backtest.py --stacked` wählt den Stack.
  - 4 neue Tests in `test_ml_stacked.py` (HW-Round-Trip, Stacked-Round-Trip, Small-Training-Refusal, Feature-Lock-At-Fit-Time). Gesamt-Testsuite: **51 grün in 24,7 s**.
  - **Vergleich auf 5 Jahren ePIN-Bayern-Daten, 220 Folds je Scope:**

  | Pollen × H | Modell | MAE | WIS80 | Cov80 | ΔPers | ΔSeas |
  |---|---|---:|---:|---:|---:|---:|
  | Birke 7   | single  | 48.9 | 29.4 | 0.90 | +16.6 % | +20.3 % |
  | Birke 7   | **stacked** | **22.7** | **19.4** | 0.58 | **+44.9 %** | **+47.3 %** |
  | Birke 14  | single  | 46.2 | 27.9 | 0.91 | +36.4 % | +25.1 % |
  | Birke 14  | **stacked** | **29.0** | **22.0** | 0.58 | **+49.8 %** | **+40.9 %** |
  | Gräser 7  | single  | 11.3 |  7.1 | 0.91 | +32.5 % | +26.2 % |
  | Gräser 7  | **stacked** |  **7.9** |  **5.8** | 0.62 | **+45.0 %** | **+40.0 %** |
  | Gräser 14 | single  | 15.0 |  8.3 | 0.86 | +44.1 % | +15.5 % |
  | Gräser 14 | **stacked** |  **9.2** |  **6.5** | 0.66 | **+56.2 %** | **+33.9 %** |
  | Erle 7    | single  | 31.4 | 19.3 | 0.85 | +39.2 % | +40.3 % |
  | Erle 7    | **stacked** | **19.0** | **13.9** | 0.65 | **+56.2 %** | **+57.0 %** |

  - **MAE reduziert sich um 30–54 %**, **WIS um 19–34 %** gegenüber dem einstufigen Ridge+GBM. Vs. Persistence liegt die WIS-Verbesserung jetzt bei +45 bis +56 %, vs. Seasonal-Naive bei +34 bis +57 %. Coverage80 schrumpft allerdings von ~0,90 (überbreit) auf 0,58–0,66 (leicht zu eng) — das ist der klassische Kompromiss tighter WIS ↔ Coverage. Phase 4c/5 wird das via Isotonic-Kalibrierung oder Conformal-Prediction auf den 80-%-Zielwert heben.

### Added
- **Phase 4c — Modell-Artefakte und öffentliche Forecast-API.** Die trainierten Modelle sind jetzt als Filesystem-Artefakte persistiert und werden vom FastAPI-Backend als REST-Endpunkte ausgespielt. Milestone-1-DoD-Punkte 6 und 7 erfüllt.
  - `app/services/ml/model_registry.py` — joblib-basierte Speicherung unter `backend/app/ml_models/<pollen>/<region>/h<horizon>/` plus Metadaten-JSON (Feature-Spalten-Order, Trainings-Fenster, Modell-Version, Trainingszeitpunkt, Feature-Config). Die API liest die Metadaten und verweigert die Auslieferung, wenn die aktuellen Feature-Spalten vom Trainings-Snapshot abweichen — Feature-Drift wird laut, nicht leise, erkannt.
  - `scripts/run_train.py` — CLI, trainiert ForecastService auf voller Historie und speichert Artefakt.
  - Neuer Router `app/api/pollen.py` mit drei Endpunkten:
    - `GET /api/v1/pollen/current?region=…&pollen_type=…` — liefert den jüngsten Ist-Wert. ePIN-Konzentration bevorzugt, DWD-Index als Fallback.
    - `GET /api/v1/pollen/forecast?region=…&pollen_type=…&horizon_days=7|14` — lädt das passende Modell-Artefakt, baut den Feature-Vektor aus den letzten 2 Jahren und gibt Punktprognose + 80 %-Band zurück.
    - `GET /api/v1/pollen/forecast/regional?pollen_type=…&horizon_days=7|14` — rankt alle Regionen mit einem trainierten Modell nach erwarteter Konzentration.
  - Schema in `app/schemas/pollen.py`, Router eingehängt in `main.py`.
  - 6 neue API-Tests via `fastapi.testclient.TestClient`: /current-Response-Shape, 503 ohne Modell, Bounds-Monotonie nach Training, Regional-Ranking-Order, 400 auf unbekannten Horizon, 404 auf unbekannte Region. Gesamt-Testsuite: **47 grün in 20,4 s**.
  - Smoke-Test gegen echtes Backend auf Port 8001 mit allen 6 Bayern-Modellen (Birke/Gräser/Erle × h7/h14):
    - `/api/v1/pollen/current?region=BY&pollen_type=birke` → **1462 Pollen/m³** am 2026-04-16 (München, Mitte-April-Birkenpeak — plausibel).
    - `/api/v1/pollen/forecast?region=BY&pollen_type=birke&horizon_days=7` → 483.6 [8.8, 686.3], `confidence_label=low`.
    - `/api/v1/pollen/forecast/regional?pollen_type=birke&horizon_days=7` → Ranking mit 1 Eintrag (BY) — wird mit weiteren Region-Modellen automatisch wachsen.
  - `.gitignore` erweitert um `backend/app/ml_models/**/metadata.json`, damit Modell-Artefakte lokal reproduzierbar sind, aber nicht im Repo landen.
- **Phase 4-real — erster Echtdaten-Backtest auf 5 Jahren ePIN Bayern.** Lokale SQLite-DB (Override via `DATABASE_URL`) mit 867 170 Pollen-Observations, 30 928 Wetter-Zeilen (BrightSky, 16 Hauptstädte), 496 Ferien-Einträgen 2022–2026. Fenster 2021-01-01 bis 2026-04-17, Horizonte 7 und 14, Walk-Forward mit `min_train_days=365` und `step_days=7`, ~220 Folds pro Run.

  | Pollen | Horizon | Folds | MAE | WIS80 | Cov80 | ΔWIS vs Persistence | ΔWIS vs Seasonal-Naive |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | Birke | 7  | 221 | 48.9 | 29.4 | 0.90 | **−16.6 %** | −20.3 % |
  | Birke | 14 | 220 | 46.2 | 27.9 | 0.91 | **−36.4 %** | −25.1 % |
  | Gräser | 7  | 221 | 11.3 |  7.1 | 0.91 | **−32.5 %** | −26.2 % |
  | Gräser | 14 | 220 | 15.0 |  8.3 | 0.86 | **−44.1 %** | −15.5 % |
  | Erle | 7  | 221 | 31.4 | 19.3 | 0.85 | **−39.2 %** | −40.3 % |

  **Ehrliche Befunde:** Das Ridge-Median-Modell ist auf **MAE** oft nur knapp oder gar nicht besser als die Baselines — Pollen haben starke t-7-Autokorrelation, die ein einfacher Lag bereits gut erfasst. Auf **WIS80** gewinnt das Modell trotzdem durchgängig klar (−16 bis −44 %), weil die pinball-regressierten Quantil-Bänder schmaler kalibriert sind als die aus der empirischen Residual-Verteilung der Baseline. **Coverage80** liegt systematisch über dem 80%-Ziel (0.85–0.91) — die Bänder sind leicht zu breit. Das ist das richtige Vorzeichen für Phase 4b (Holt-Winters + Prophet + XGBoost-Stacking): Der Median-Fit braucht mehr Tiefe, die Unsicherheitskalibrierung muss nachgeschärft werden.
  - `DATABASE_URL`-Override in `config.py` und `db/session.py` (StaticPool für SQLite, QueuePool für Postgres).
  - Neue CLI `scripts/summarize_backtests.py` als Report-Formatter.
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
