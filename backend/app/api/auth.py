import json
import logging
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.api.deps import AUTH_COOKIE_NAME, get_current_user, get_optional_current_user
from app.core.config import get_settings
from app.core.rate_limit import limiter
from app.core.security import create_access_token, get_password_hash, verify_password
from app.core.time import utc_now
from app.db.session import get_db
from app.models.database import AuditLog
from app.schemas.token import SessionState

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()

_SUPPORTED_AUTH_ROLES = {"admin", "analyst", "operator", "user", "viewer"}
_WEAK_PASSWORDS = {
    "admin", "password", "123456", "test", "changeme",
    "letmein", "welcome", "monkey", "qwerty", "abc123",
}
_FAILED_LOGIN_WINDOW_SECONDS = 15 * 60
_MAX_FAILED_LOGIN_ATTEMPTS = 5
_ACCESS_TOKEN_LIFETIME_MINUTES = 60
_COOKIE_SAMESITE = "lax"
_AUTH_ENTITY_TYPE = "auth_session"
_LOGIN_SUCCESS_ACTION = "login_success"
_LOGIN_FAILED_ACTION = "login_failed"
_LOGOUT_ACTION = "logout"


def _normalized_username(username: str) -> str:
    return str(username or "").strip().lower()


def _cookie_secure(settings_obj=settings) -> bool:
    return str(getattr(settings_obj, "ENVIRONMENT", "development") or "").lower() in {
        "production", "staging",
    }


def _validate_password_strength(password: str, *, source_label: str) -> str:
    normalized_password = str(password or "")
    if len(normalized_password) < 12:
        raise RuntimeError(f"FATAL: {source_label} must be at least 12 characters long.")
    if normalized_password.lower() in _WEAK_PASSWORDS:
        raise RuntimeError(
            f"FATAL: {source_label} is a known weak/default password. Set a strong password."
        )
    return normalized_password


def _canonical_role(role: str | None) -> str:
    normalized_role = str(role or "user").strip().lower()
    if normalized_role not in _SUPPORTED_AUTH_ROLES:
        supported = ", ".join(sorted(_SUPPORTED_AUTH_ROLES))
        raise RuntimeError(
            f"FATAL: Unsupported auth role '{normalized_role}'. Supported roles: {supported}."
        )
    return normalized_role


def _register_auth_user(
    users: dict[str, dict[str, str]],
    *,
    subject: str,
    password: str,
    role: str,
    source_label: str,
) -> None:
    normalized_subject = _normalized_username(subject)
    if not normalized_subject:
        raise RuntimeError(f"FATAL: {source_label} is missing a usable email/username.")
    if normalized_subject in users:
        raise RuntimeError(
            f"FATAL: Duplicate auth user '{normalized_subject}' in auth configuration."
        )
    normalized_password = _validate_password_strength(password, source_label=source_label)
    users[normalized_subject] = {
        "subject": normalized_subject,
        "password": get_password_hash(normalized_password),
        "role": _canonical_role(role),
    }


def _registry_entries(raw_registry: str) -> list[dict]:
    try:
        parsed = json.loads(raw_registry)
    except json.JSONDecodeError as exc:
        raise RuntimeError("FATAL: AUTH_USER_REGISTRY_JSON is invalid JSON.") from exc

    if isinstance(parsed, dict):
        entries = parsed.get("users")
    else:
        entries = parsed
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("FATAL: AUTH_USER_REGISTRY_JSON must contain a non-empty user list.")
    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeError("FATAL: AUTH_USER_REGISTRY_JSON entries must be JSON objects.")
    return entries


def _build_auth_users(settings_obj=settings) -> dict[str, dict[str, str]]:
    users: dict[str, dict[str, str]] = {}

    raw_registry = str(getattr(settings_obj, "AUTH_USER_REGISTRY_JSON", "") or "").strip()
    if raw_registry:
        for index, entry in enumerate(_registry_entries(raw_registry), start=1):
            subject = (
                entry.get("email")
                or entry.get("username")
                or entry.get("subject")
                or ""
            )
            password = str(entry.get("password") or "")
            role = str(entry.get("role") or "user")
            _register_auth_user(
                users,
                subject=subject,
                password=password,
                role=role,
                source_label=f"AUTH_USER_REGISTRY_JSON entry #{index}",
            )
        return users

    admin_email = str(getattr(settings_obj, "ADMIN_EMAIL", "") or "").strip()
    admin_password = str(getattr(settings_obj, "ADMIN_PASSWORD", "") or "").strip()
    if not admin_email or not admin_password:
        raise RuntimeError(
            "FATAL: Configure AUTH_USER_REGISTRY_JSON or set ADMIN_EMAIL and ADMIN_PASSWORD."
        )

    _register_auth_user(
        users,
        subject=admin_email,
        password=admin_password,
        role="admin",
        source_label="ADMIN_PASSWORD",
    )
    return users


_AUTH_USERS = _build_auth_users(settings)


def _client_host(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _record_auth_event(
    db: Session,
    *,
    user: str,
    ip_address: str,
    action: str,
    reason: str | None = None,
) -> None:
    db.add(
        AuditLog(
            user=user,
            action=action,
            entity_type=_AUTH_ENTITY_TYPE,
            reason=reason,
            ip_address=ip_address,
        )
    )
    db.commit()


def _failed_attempt_count(db: Session, username: str, ip_address: str, now_dt) -> int:
    window_start = now_dt - timedelta(seconds=_FAILED_LOGIN_WINDOW_SECONDS)
    latest_success = (
        db.query(func.max(AuditLog.timestamp))
        .filter(
            AuditLog.entity_type == _AUTH_ENTITY_TYPE,
            AuditLog.action == _LOGIN_SUCCESS_ACTION,
            AuditLog.user == username,
            AuditLog.ip_address == ip_address,
        )
        .scalar()
    )
    effective_window_start = max(window_start, latest_success) if latest_success else window_start
    return int(
        db.query(func.count(AuditLog.id))
        .filter(
            AuditLog.entity_type == _AUTH_ENTITY_TYPE,
            AuditLog.action == _LOGIN_FAILED_ACTION,
            AuditLog.user == username,
            AuditLog.ip_address == ip_address,
            AuditLog.timestamp >= effective_window_start,
        )
        .scalar()
        or 0
    )


def _is_locked_out(db: Session, username: str, ip_address: str, now_dt) -> bool:
    return _failed_attempt_count(db, username, ip_address, now_dt) >= _MAX_FAILED_LOGIN_ATTEMPTS


def _set_auth_cookie(response: Response, token: str, remember_me: bool) -> None:
    max_age = _ACCESS_TOKEN_LIFETIME_MINUTES * 60 if remember_me else None
    response.set_cookie(
        key=AUTH_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=_cookie_secure(),
        samesite=_COOKIE_SAMESITE,
        max_age=max_age,
        path="/",
    )


def _clear_auth_cookie(response: Response) -> None:
    response.delete_cookie(
        key=AUTH_COOKIE_NAME,
        httponly=True,
        secure=_cookie_secure(),
        samesite=_COOKIE_SAMESITE,
        path="/",
    )


@router.post("/login", response_model=SessionState)
@limiter.limit("10/minute")
async def login(
    request: Request,
    response: Response,
    remember_me: bool = True,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    now_dt = utc_now()
    username = _normalized_username(form_data.username)
    client_host = _client_host(request)

    if _is_locked_out(db, username, client_host, now_dt):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed login attempts. Please try again later.",
        )

    user = _AUTH_USERS.get(username)
    if not user or not verify_password(form_data.password, user["password"]):
        _record_auth_event(db, user=username, ip_address=client_host, action=_LOGIN_FAILED_ACTION)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        data={"sub": user["subject"], "role": user["role"]},
        expires_delta=timedelta(minutes=_ACCESS_TOKEN_LIFETIME_MINUTES),
    )
    _set_auth_cookie(response, access_token, remember_me)
    _record_auth_event(db, user=username, ip_address=client_host, action=_LOGIN_SUCCESS_ACTION)
    return SessionState(authenticated=True, subject=user["subject"], role=user["role"])


@router.get("/session", response_model=SessionState)
async def get_session(current_user: dict | None = Depends(get_optional_current_user)):
    if not current_user:
        return SessionState(authenticated=False, subject=None, role=None)
    return SessionState(
        authenticated=True,
        subject=current_user.get("sub"),
        role=current_user.get("role"),
    )


@router.post("/logout", response_model=SessionState)
async def logout(
    request: Request,
    response: Response,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    session_id = str(current_user.get("sid") or "").strip()
    if session_id:
        _record_auth_event(
            db,
            user=_normalized_username(current_user.get("sub") or ""),
            ip_address=_client_host(request),
            action=_LOGOUT_ACTION,
            reason=session_id,
        )
    _clear_auth_cookie(response)
    return SessionState(authenticated=False)
