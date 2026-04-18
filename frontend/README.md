# PollenCast Frontend

Leichtgewichtiges statisches Dashboard für den PollenCast-Forecast.

- Keine Build-Chain, kein Framework — `public/index.html` + `public/snapshot.json`.
- Tailwind + Chart.js via CDN.
- Der Snapshot wird aus der lokalen SQLite-Datenbank per
  `backend/scripts` ad-hoc regeneriert und hier eingecheckt.

## Lokal ansehen

```bash
cd frontend/public
python -m http.server 8765
# open http://localhost:8765
```

## Snapshot neu bauen

Aus dem Backend-Kontext mit gefüllter `pollencast_local.db`:

```bash
cd backend
set -a; source .env.local; set +a
python -c "from scripts.export_snapshot import main; main()"  # (kommt)
```

Bis zu dem Zeitpunkt wird der Snapshot aus einem Ad-hoc-Python-Skript
generiert, das in den Backend-Commits dokumentiert ist.

## Vercel

`vercel.json` definiert das Ausgangsverzeichnis `public/`.
