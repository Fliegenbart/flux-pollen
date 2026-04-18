import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_default_env_file() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / ".env"
        if candidate.exists():
            return candidate
    return Path(__file__).resolve().parents[3] / ".env"


_DEFAULT_ENV_FILE = _find_default_env_file()
load_dotenv(_DEFAULT_ENV_FILE, override=False)


class Settings(BaseSettings):
    # App
    APP_NAME: str = "PollenCast"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"

    # Database
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432

    @property
    def DATABASE_URL(self) -> str:
        override = os.getenv("DATABASE_URL")
        if override:
            return override
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # Data sources
    DWD_POLLEN_URL: str = "https://opendata.dwd.de/climate_environment/health/alerts/s31fg.json"
    OPENWEATHER_API_KEY: str | None = None

    # Security
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    ADMIN_EMAIL: str | None = None
    ADMIN_PASSWORD: str | None = None
    AUTH_USER_REGISTRY_JSON: str | None = None
    M2M_SECRET_KEY: str | None = None

    # API surface
    API_DOCS_ENABLED: bool | None = None
    PUBLIC_HEALTH_DETAILS_ENABLED: bool | None = None
    PUBLIC_METRICS_ENABLED: bool | None = None
    METRICS_AUTH_TOKEN: str | None = None

    # CORS
    ALLOWED_ORIGINS: str = "http://localhost:3000"

    @property
    def CORS_ORIGINS(self) -> list[str]:
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin.strip()]

    # ML defaults
    FORECAST_HORIZON_DAYS: int = 7
    FORECAST_SECONDARY_HORIZON_DAYS: int = 14
    CONFIDENCE_LEVEL: float = 0.80

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"

    # Rate limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60

    @property
    def EFFECTIVE_API_DOCS_ENABLED(self) -> bool:
        if self.API_DOCS_ENABLED is not None:
            return bool(self.API_DOCS_ENABLED)
        return self.ENVIRONMENT != "production"

    @property
    def EFFECTIVE_PUBLIC_HEALTH_DETAILS_ENABLED(self) -> bool:
        if self.PUBLIC_HEALTH_DETAILS_ENABLED is not None:
            return bool(self.PUBLIC_HEALTH_DETAILS_ENABLED)
        return self.ENVIRONMENT != "production"

    @property
    def EFFECTIVE_PUBLIC_METRICS_ENABLED(self) -> bool:
        if self.PUBLIC_METRICS_ENABLED is not None:
            return bool(self.PUBLIC_METRICS_ENABLED)
        return self.ENVIRONMENT != "production"

    model_config = SettingsConfigDict(
        env_file=_DEFAULT_ENV_FILE,
        case_sensitive=True,
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
