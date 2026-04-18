"""Filesystem-backed model registry.

Trained ForecastService instances are persisted as joblib blobs next to
a metadata JSON so a later API call can load the right artefact without
re-reading the DB or rebuilding features. The layout is

    backend/app/ml_models/<pollen_type>/<region_code>/h<horizon>/
        model.joblib
        metadata.json

Metadata records everything downstream code needs to stay honest: the
exact feature column order used at fit time, the training window, and
the feature-build config. When the API loads a model it checks the
column order against the current panel before calling ``predict`` — a
mismatch means our feature engineering has drifted and we should abort
rather than quietly serve a broken forecast.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

import joblib

from app.services.ml.feature_engineering import FeatureBuildConfig
from app.services.ml.forecast_service import ForecastService

logger = logging.getLogger(__name__)

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "ml_models"

__all__ = [
    "ModelArtifact",
    "ArtifactMetadata",
    "save_artifact",
    "load_artifact",
    "artifact_path",
    "list_artifacts",
]


@dataclass(frozen=True)
class ArtifactMetadata:
    pollen_type: str
    region_code: str
    horizon_days: int
    feature_columns: list[str]
    model_version: str
    trained_at: str
    training_window_start: str
    training_window_end: str
    train_n_samples: int
    feature_config: dict

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ModelArtifact:
    # ``service`` can be any fit/predict object satisfying ForecasterProtocol:
    # the plain ForecastService, the StackedForecastService, or either wrapped
    # in a ConformalCalibratedForecaster. joblib preserves the concrete type.
    service: object
    metadata: ArtifactMetadata


def artifact_path(
    *,
    pollen_type: str,
    region_code: str,
    horizon_days: int,
    root: Path | str | None = None,
) -> Path:
    base = Path(root) if root else DEFAULT_ROOT
    return base / pollen_type / region_code / f"h{int(horizon_days)}"


def save_artifact(
    artifact: ModelArtifact,
    *,
    root: Path | str | None = None,
) -> Path:
    meta = artifact.metadata
    path = artifact_path(
        pollen_type=meta.pollen_type,
        region_code=meta.region_code,
        horizon_days=meta.horizon_days,
        root=root,
    )
    path.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifact.service, path / "model.joblib")
    (path / "metadata.json").write_text(
        json.dumps(meta.to_dict(), indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("Saved model artefact to %s", path)
    return path


def load_artifact(
    *,
    pollen_type: str,
    region_code: str,
    horizon_days: int,
    root: Path | str | None = None,
) -> ModelArtifact:
    path = artifact_path(
        pollen_type=pollen_type,
        region_code=region_code,
        horizon_days=horizon_days,
        root=root,
    )
    model_file = path / "model.joblib"
    metadata_file = path / "metadata.json"
    if not model_file.exists() or not metadata_file.exists():
        raise FileNotFoundError(
            f"No trained artefact at {path}. Run scripts/run_train.py first."
        )

    service = joblib.load(model_file)
    payload = json.loads(metadata_file.read_text(encoding="utf-8"))
    metadata = ArtifactMetadata(
        pollen_type=payload["pollen_type"],
        region_code=payload["region_code"],
        horizon_days=int(payload["horizon_days"]),
        feature_columns=list(payload["feature_columns"]),
        model_version=payload["model_version"],
        trained_at=payload["trained_at"],
        training_window_start=payload["training_window_start"],
        training_window_end=payload["training_window_end"],
        train_n_samples=int(payload["train_n_samples"]),
        feature_config=dict(payload["feature_config"]),
    )
    return ModelArtifact(service=service, metadata=metadata)


def list_artifacts(root: Path | str | None = None) -> list[ArtifactMetadata]:
    base = Path(root) if root else DEFAULT_ROOT
    if not base.exists():
        return []
    items: list[ArtifactMetadata] = []
    for meta_file in base.glob("*/*/h*/metadata.json"):
        try:
            payload = json.loads(meta_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        items.append(
            ArtifactMetadata(
                pollen_type=payload["pollen_type"],
                region_code=payload["region_code"],
                horizon_days=int(payload["horizon_days"]),
                feature_columns=list(payload.get("feature_columns") or []),
                model_version=payload.get("model_version", ""),
                trained_at=payload.get("trained_at", ""),
                training_window_start=payload.get("training_window_start", ""),
                training_window_end=payload.get("training_window_end", ""),
                train_n_samples=int(payload.get("train_n_samples", 0)),
                feature_config=dict(payload.get("feature_config") or {}),
            )
        )
    return items


def build_metadata(
    *,
    service: object,
    feature_config: FeatureBuildConfig,
    horizon_days: int,
    feature_columns: Iterable[str],
    train_n_samples: int,
    model_version: str,
) -> ArtifactMetadata:
    return ArtifactMetadata(
        pollen_type=feature_config.pollen_type,
        region_code=feature_config.region_code,
        horizon_days=int(horizon_days),
        feature_columns=list(feature_columns),
        model_version=model_version,
        trained_at=datetime.utcnow().isoformat(),
        training_window_start=feature_config.start_date.isoformat(),
        training_window_end=feature_config.end_date.isoformat(),
        train_n_samples=int(train_n_samples),
        feature_config={
            "target_lags": list(feature_config.target_lags),
            "target_rolling_windows": list(feature_config.target_rolling_windows),
            "weather_lags": list(feature_config.weather_lags),
            "cross_pollen_lags": list(feature_config.cross_pollen_lags),
        },
    )
