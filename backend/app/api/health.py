from fastapi import APIRouter

from app.db.session import check_db_connection

router = APIRouter()


@router.get("/health/live")
async def live():
    return {"status": "ok"}


@router.get("/health/ready")
async def ready():
    db_ok = await check_db_connection()
    return {"status": "ok" if db_ok else "degraded", "database": "up" if db_ok else "down"}
