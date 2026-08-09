from fastapi import APIRouter, HTTPException, status
from redis.asyncio import Redis
from sqlalchemy import text

from app.adapters.db.session import engine
from app.core.config import get_settings

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def ready() -> dict[str, object]:
    settings = get_settings()
    checks: dict[str, str] = {}
    try:
        if settings.persistence_backend.lower() == "postgres":
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            checks["postgres"] = "ok"
        if settings.cache_backend.lower() == "redis":
            redis = Redis.from_url(settings.redis_url)
            try:
                await redis.ping()
            finally:
                await redis.aclose()
            checks["redis"] = "ok"
        if settings.rag_backend.lower() == "milvus":
            from pymilvus import connections, utility

            connections.connect(alias="ready", host=settings.milvus_host, port=settings.milvus_port)
            try:
                checks["milvus"] = "ok" if utility.has_collection(settings.milvus_collection, using="ready") else "empty"
            finally:
                connections.disconnect(alias="ready")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "not_ready", "checks": checks, "error": exc.__class__.__name__},
        ) from exc
    return {"status": "ready", "checks": checks}
