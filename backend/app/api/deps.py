from datetime import UTC, datetime

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from jose import ExpiredSignatureError, JWTError, jwt
from sqlalchemy.orm import Session

from app.core.security import ALGORITHM, SECRET_KEY
from app.db.session import get_db
from app.models.database import AuditLog
from app.schemas.token import TokenPayload

AUTH_COOKIE_NAME = "pollencast_session"
AUTH_SESSION_ENTITY = "auth_session"
AUTH_LOGOUT_ACTION = "logout"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


def _supports_audit_queries(db: Session | None) -> bool:
    return db is not None and hasattr(db, "query")


def _is_revoked_session(db: Session, sid: str | None) -> bool:
    normalized_sid = str(sid or "").strip()
    if not normalized_sid or not _supports_audit_queries(db):
        return False
    return (
        db.query(AuditLog.id)
        .filter(
            AuditLog.entity_type == AUTH_SESSION_ENTITY,
            AuditLog.action == AUTH_LOGOUT_ACTION,
            AuditLog.reason == normalized_sid,
        )
        .first()
        is not None
    )


def _legacy_token_is_invalidated(db: Session, subject: str | None, issued_at: int | None) -> bool:
    normalized_subject = str(subject or "").strip()
    if not normalized_subject or issued_at is None or not _supports_audit_queries(db):
        return False

    latest_logout = (
        db.query(AuditLog.timestamp)
        .filter(
            AuditLog.entity_type == AUTH_SESSION_ENTITY,
            AuditLog.action == AUTH_LOGOUT_ACTION,
            AuditLog.user == normalized_subject,
        )
        .order_by(AuditLog.timestamp.desc())
        .first()
    )
    if not latest_logout or not latest_logout[0]:
        return False

    issued_at_dt = datetime.fromtimestamp(int(issued_at), tz=UTC).replace(tzinfo=None)
    return issued_at_dt <= latest_logout[0]


async def get_current_user(
    request: Request,
    token: str | None = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> dict:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    auth_token = token or request.cookies.get(AUTH_COOKIE_NAME)
    if not auth_token:
        raise credentials_exception

    try:
        payload = jwt.decode(
            auth_token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
            options={"require_exp": True},
        )
        token_payload = TokenPayload.model_validate(payload)
        if not token_payload.sub:
            raise credentials_exception
        if _is_revoked_session(db, token_payload.sid):
            raise credentials_exception
        if not token_payload.sid and _legacy_token_is_invalidated(
            db, token_payload.sub, token_payload.iat
        ):
            raise credentials_exception
        return token_payload.model_dump()
    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except JWTError:
        raise credentials_exception


async def get_optional_current_user(
    request: Request,
    token: str | None = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> dict | None:
    try:
        return await get_current_user(request, token, db)
    except HTTPException as exc:
        if exc.status_code == status.HTTP_401_UNAUTHORIZED:
            return None
        raise


async def get_current_admin(current_user: dict = Depends(get_current_user)) -> dict:
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough privileges",
        )
    return current_user
