from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field, model_validator

from app.adapters.auth.keycloak import AuthenticationError, KeycloakJWTAuthenticator
from app.adapters.cache.memory import RateLimitExceeded
from app.agent.graph import AuthorizationError, SingleEmployeeSupportAgent
from app.api.dependencies import get_agent
from app.application.operations import (
    ingest_seed_corpus,
    knowledge_status,
    rebuild_seed_corpus,
    run_policy_eval,
    settings_with_overrides,
)
from app.core.config import get_settings
from app.domain.models import ActorContext, ChatRequest, ChatResponse

router = APIRouter()


def ops_error(exc: Exception) -> HTTPException:
    message = str(exc) or exc.__class__.__name__
    lower = message.lower()
    if "sentence-transformers" in lower or "sentence_transformers" in lower or "baai/" in lower:
        message = (
            "Embedding model is not ready in the backend container. "
            "Rebuild the backend image and allow first BGE model download. "
            f"Detail: {message}"
        )
    elif "milvus" in lower or "connect" in lower or "connection" in lower:
        message = f"Milvus is not ready or not reachable. Start Docker services, then retry. Detail: {message}"
    return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=message)


class OpsRagConfig(BaseModel):
    chunk_size: int = Field(default=120, ge=20, le=2000)
    chunk_overlap: int = Field(default=20, ge=0, le=1000)
    retrieval_mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    fusion_strategy: Literal["weighted", "rrf"] = "weighted"
    reranker_provider: Literal["score", "heuristic_cross_encoder"] = "score"
    milvus_native_hybrid: bool = True
    top_k: int = Field(default=5, ge=1, le=20)
    dense_weight: float = Field(default=0.6, ge=0, le=1)
    sparse_weight: float = Field(default=0.4, ge=0, le=1)
    candidate_multiplier: int = Field(default=4, ge=1, le=20)
    embedding_provider: Literal["sentence_transformers", "fastembed"] | None = None
    embedding_model: str | None = None
    embedding_device: str | None = None
    embedding_dim: int | None = Field(default=None, ge=16, le=4096)

    @model_validator(mode="after")
    def validate_values(self) -> "OpsRagConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.fusion_strategy == "weighted" and self.dense_weight + self.sparse_weight <= 0:
            raise ValueError("at least one retrieval weight must be greater than zero")
        return self

    def to_settings_overrides(self) -> dict:
        values = {
            "rag_chunk_size": self.chunk_size,
            "rag_chunk_overlap": self.chunk_overlap,
            "rag_retrieval_mode": self.retrieval_mode,
            "rag_fusion_strategy": self.fusion_strategy,
            "rag_reranker_provider": self.reranker_provider,
            "rag_top_k": self.top_k,
            "rag_dense_weight": self.dense_weight,
            "rag_sparse_weight": self.sparse_weight,
            "rag_candidate_multiplier": self.candidate_multiplier,
            "milvus_native_hybrid": self.milvus_native_hybrid,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "embedding_device": self.embedding_device,
            "milvus_vector_dim": self.embedding_dim,
        }
        return {key: value for key, value in values.items() if value is not None}


def bearer_token(authorization: str = Header(default="")) -> str:
    prefix = "Bearer "
    if not authorization.startswith(prefix):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing bearer token")
    return authorization.removeprefix(prefix)


async def ops_actor(token: str = Depends(bearer_token)) -> ActorContext:
    settings = get_settings()
    authenticator = KeycloakJWTAuthenticator(
        settings.keycloak_issuer,
        settings.keycloak_audience,
        settings.jwks_cache_seconds,
        allow_dev_token=settings.allow_dev_token,
    )
    try:
        actor = await authenticator.authenticate(token)
    except AuthenticationError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    if not actor.roles.intersection({"admin", "platform_admin", "hr_admin"}):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="admin role required")
    return actor


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    token: str = Depends(bearer_token),
    agent: SingleEmployeeSupportAgent = Depends(get_agent),
) -> ChatResponse:
    try:
        return await agent.run(request, token)
    except AuthenticationError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    except AuthorizationError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except RateLimitExceeded as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc


@router.get("/ops/knowledge")
async def ops_knowledge(actor: ActorContext = Depends(ops_actor)) -> dict:
    _ = actor
    return knowledge_status(get_settings())


@router.post("/ops/ingest")
async def ops_ingest(config: OpsRagConfig | None = None, actor: ActorContext = Depends(ops_actor)) -> dict:
    _ = actor
    settings = settings_with_overrides(get_settings(), config.to_settings_overrides() if config else None)
    try:
        return ingest_seed_corpus(settings)
    except Exception as exc:
        raise ops_error(exc) from exc


@router.post("/ops/rebuild")
async def ops_rebuild(config: OpsRagConfig, actor: ActorContext = Depends(ops_actor)) -> dict:
    _ = actor
    settings = settings_with_overrides(get_settings(), config.to_settings_overrides())
    try:
        rebuild = rebuild_seed_corpus(settings)
        ingest = ingest_seed_corpus(settings)
        get_agent.cache_clear()
        return {"rebuild": rebuild, "ingest": ingest}
    except Exception as exc:
        raise ops_error(exc) from exc


@router.post("/ops/eval")
async def ops_eval(config: OpsRagConfig | None = None, top_k: int | None = None, actor: ActorContext = Depends(ops_actor)) -> dict:
    _ = actor
    settings = settings_with_overrides(get_settings(), config.to_settings_overrides() if config else None)
    effective_top_k = config.top_k if config else top_k
    try:
        return await run_policy_eval(settings, top_k=effective_top_k)
    except Exception as exc:
        raise ops_error(exc) from exc
