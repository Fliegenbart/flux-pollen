from pydantic import BaseModel


class SessionState(BaseModel):
    authenticated: bool
    subject: str | None = None
    role: str | None = None


class TokenPayload(BaseModel):
    sub: str | None = None
    role: str | None = None
    sid: str | None = None
    iat: int | None = None
