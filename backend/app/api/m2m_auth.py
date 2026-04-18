from __future__ import annotations

import logging
import os
import secrets

from fastapi import Header, HTTPException, status

logger = logging.getLogger(__name__)


def verify_m2m_api_key(x_api_key: str | None = Header(None, alias="X-API-Key")) -> None:
    """API-key guard for machine-to-machine integrations."""
    expected = os.getenv("M2M_SECRET_KEY", "")
    if not expected:
        logger.error("M2M_SECRET_KEY is empty; refusing machine-to-machine request.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="M2M auth is not configured.",
        )

    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
        )

    if not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
