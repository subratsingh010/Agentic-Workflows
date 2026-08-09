from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.health import router as health_router
from app.api.routes import router as api_router
from app.adapters.db.session import create_schema
from app.adapters.observability.setup import setup_observability
from app.core.config import get_settings
from app.core.logging import configure_logging


settings = get_settings()
configure_logging(settings.log_level)
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("startup", app=settings.app_name, env=settings.app_env)
    if settings.persistence_backend.lower() == "postgres":
        await create_schema()
    yield
    logger.info("shutdown", app=settings.app_name)


app = FastAPI(
    title="Employee Support AI Agent",
    version="0.1.0",
    description="Single-agent employee support API with policy RAG and typed mock tools.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(health_router)
app.include_router(api_router, prefix="/api/v1")
setup_observability(app, settings)

