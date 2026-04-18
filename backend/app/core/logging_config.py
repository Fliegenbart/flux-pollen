"""Structured JSON logging configuration."""

import logging
import os
import sys
import uuid
from contextvars import ContextVar

from pythonjsonlogger import jsonlogger

correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")


class CorrelationFilter(logging.Filter):
    def __init__(self, *, service_name: str, environment: str, app_version: str) -> None:
        super().__init__()
        self.service_name = service_name
        self.environment = environment
        self.app_version = app_version

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = correlation_id.get("")  # type: ignore[attr-defined]
        record.service = self.service_name  # type: ignore[attr-defined]
        record.environment = self.environment  # type: ignore[attr-defined]
        record.app_version = self.app_version  # type: ignore[attr-defined]
        record.process_id = os.getpid()  # type: ignore[attr-defined]
        return True


def generate_correlation_id() -> str:
    return uuid.uuid4().hex[:12]


def setup_logging(
    level: str = "INFO",
    json_format: bool = True,
    *,
    service_name: str = "pollencast-backend",
    environment: str = "development",
    app_version: str = "unknown",
) -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)

    if json_format:
        formatter = jsonlogger.JsonFormatter(
            fmt=(
                "%(asctime)s %(name)s %(levelname)s %(message)s %(correlation_id)s "
                "%(service)s %(environment)s %(app_version)s %(process_id)s"
            ),
            rename_fields={"asctime": "timestamp", "levelname": "level"},
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s (%(correlation_id)s) "
            "[%(service)s|%(environment)s|%(app_version)s] %(message)s"
        )

    handler.setFormatter(formatter)
    handler.addFilter(
        CorrelationFilter(
            service_name=service_name,
            environment=environment,
            app_version=app_version,
        )
    )
    root.addHandler(handler)

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def log_event(logger: logging.Logger, event: str, *, level: int = logging.INFO, **fields) -> None:
    logger.log(level, event, extra={"event": event, **fields})
