"""Pretty-print every persisted BacktestRun and its improvements.

Handy for the "how did that real-data run actually look" moment after
running ``run_backtest.py``. Does nothing fancy — reads from the DB and
formats a table. Output is deliberately grep-friendly.

Usage:
    python -m scripts.summarize_backtests
    python -m scripts.summarize_backtests --pollen birke
    python -m scripts.summarize_backtests --region BY --horizon 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.db.session import SessionLocal
from app.models.database import BacktestRun


def _fmt(value) -> str:
    if value is None:
        return "—"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if f == 0.0:
        return "0.00"
    if abs(f) >= 1000:
        return f"{f:,.0f}"
    return f"{f:.2f}"


def _pct(value) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:+.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pollen")
    parser.add_argument("--region")
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    db = SessionLocal()
    try:
        query = db.query(BacktestRun).order_by(BacktestRun.created_at.desc())
        if args.pollen:
            query = query.filter(BacktestRun.pollen_type == args.pollen)
        if args.region:
            query = query.filter(BacktestRun.region_code == args.region)
        if args.horizon:
            query = query.filter(BacktestRun.horizon_days == args.horizon)
        runs = query.limit(args.limit).all()
    finally:
        db.close()

    if not runs:
        print("No backtest runs found for the given filter.")
        return 1

    header = (
        f"{'pollen_type':<10} {'region':<6} {'h':>3} {'folds':>5}"
        f"  {'MAE':>8}  {'WIS80':>8}  {'cov80':>6}"
        f"  {'Δ vs pers':>10}  {'Δ vs seas':>10}"
        f"  {'model_version':<28} {'run_id':<12}"
    )
    print(header)
    print("-" * len(header))

    for run in runs:
        metrics = run.metrics or {}
        improvements = run.improvement_vs_baselines or {}
        row = (
            f"{run.pollen_type:<10} "
            f"{(run.region_code or ''):<6} "
            f"{run.horizon_days:>3} "
            f"{(run.chart_points or 0):>5}  "
            f"{_fmt(metrics.get('mae')):>8}  "
            f"{_fmt(metrics.get('wis80')):>8}  "
            f"{_fmt(metrics.get('coverage80')):>6}  "
            f"{_pct(improvements.get('persistence')):>10}  "
            f"{_pct(improvements.get('seasonal_naive')):>10}  "
            f"{(run.model_version or ''):<28} "
            f"{run.run_id:<12}"
        )
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
