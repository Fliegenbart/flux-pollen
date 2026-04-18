"""PollenCast FastAPI entry-point."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api import auth, health, outcome, pollen
from app.core.config import get_settings
from app.core.logging_config import correlation_id, log_event, setup_logging
from app.core.metrics import app_info
from app.core.rate_limit import limiter

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()

    setup_logging(
        level=settings.LOG_LEVEL,
        json_format=settings.LOG_FORMAT == "json",
        service_name="pollencast-api",
        environment=settings.ENVIRONMENT,
        app_version=settings.APP_VERSION,
    )

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Regionale Pollenflug-Prognosen mit Quantil-Unsicherheitsbändern.",
        docs_url="/docs" if settings.EFFECTIVE_API_DOCS_ENABLED else None,
        redoc_url="/redoc" if settings.EFFECTIVE_API_DOCS_ENABLED else None,
        openapi_url="/openapi.json" if settings.EFFECTIVE_API_DOCS_ENABLED else None,
    )

    app_info.info({"version": settings.APP_VERSION, "environment": settings.ENVIRONMENT})

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        cid = correlation_id.get("")
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "detail": "Ungültige Eingabedaten",
                "fields": [
                    {"field": ".".join(str(x) for x in e["loc"]), "message": e["msg"]}
                    for e in exc.errors()
                ],
                "correlation_id": cid or None,
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        cid = correlation_id.get("")
        log_event(
            logger,
            "unhandled_exception",
            level=logging.ERROR,
            path=str(request.url.path),
            method=request.method,
            correlation_id=cid or None,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "detail": "Ein interner Fehler ist aufgetreten.",
                "correlation_id": cid or None,
            },
        )

    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.state.limiter = limiter

    app.include_router(health.router, tags=["health"])
    app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
    app.include_router(pollen.router, prefix="/api/v1/pollen", tags=["pollen"])
    app.include_router(outcome.router, prefix="/api/v1/outcome", tags=["outcome"])

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
