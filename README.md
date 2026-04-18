# PollenCast

**PollenCast** ist eine regionale Pollenflug-Prognose-Engine für den deutschen OTC-Allergiemarkt. Sie liefert für jedes Bundesland und jede relevante Pollenart einen 7- bis 14-Tage-Forecast mit Quantil-Unsicherheitsbändern — basierend auf DWD-Daten, Wettermodellen und Kalender-Features, evaluiert über mehrjähriges Walk-Forward-Backtesting gegen Persistence- und Seasonal-Naive-Baselines.

Die Methodik entspricht dem, was in Medikamentenaufsicht, Rückversicherung und Krankenhaus-Surveillance als State-of-the-Art gilt: Point-in-Time-Feature-Engineering, XGBoost-Stacking über Holt-Winters / Ridge / Prophet, probabilistische Prognose mit Quantil-Regression.

## Zielanwendungen

- **Media-Aktivierung** für Antihistaminika-Marken (Lorano, Cetirizin, Aerius, Reactine, Allegra): regional gesteuerte TV-/Digital-Budgets statt Saison-Pauschalen.
- **Bestandsdisposition** in Apotheken-Großhandel und bei Filialisten: 7- bis 14-Tage-Vorlauf für Nachbestellungen.
- **Search- und Social-Bid-Modifikation** für Online-Apotheken: Pollen-Signal als exogener Regressor für Allergie-Keywords.
- **Demand-Forecasting** für Luftreinigungs- und HEPA-Filter-Hersteller.

## Architektur in einem Satz

Python/FastAPI-Backend mit PostgreSQL, tägliche Ingestion von DWD-Pollenflug + Wetter + Kalenderfeatures, Forecast-Stacking pro (Pollenart × Bundesland × Horizon), Walk-Forward-Backtests mit WIS / MAE / PIT-Kalibrierung.

## Quickstart (Lokal)

```bash
cp .env.example .env
# Placeholder in .env ersetzen: SECRET_KEY, ADMIN_PASSWORD

docker compose up -d db
cd backend
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload
```

Healthcheck: `curl http://localhost:8000/health/live`

## Datenquellen

| Quelle | Frequenz | Verwendung |
|---|---|---|
| DWD OpenData Pollenflug (`s31fg.json`) | täglich | Zielvariable (Index 0–3) für 8 Pollenarten × 8 DWD-Regionen |
| OpenWeather / DWD | täglich | Temperatur, Niederschlag, Feuchtigkeit, Wind — Features |
| `ferien-api.de` / statische Tabellen | jährlich | Kalendereffekte |

## Pollenarten und Horizonte

Primärziele im ersten Milestone: **Gräser**, **Birke**, **Erle**.
Erweiterung: Hasel, Esche, Beifuß, Ambrosia, Roggen.

Horizonte: 7 Tage (primär), 14 Tage (optional). Längere Horizonte sind wetterbedingt nicht seriös.

## Evidenz-Vorgabe

Kein Verkauf ohne belastbares Walk-Forward-Backtesting. Metriken (WIS, MAE, PIT) werden pro (Pollenart × Bundesland × Horizon) in `backtest_run` / `backtest_point` persistiert und über `/api/v1/backtest/summary` exponiert. Vergleich gegen Persistence- und Seasonal-Naive-Baselines ist Pflicht, nicht Option.

## Projekt-Status

Frühe Phase. Siehe [CHANGELOG.md](./CHANGELOG.md) für den tatsächlichen Umsetzungsstand.
