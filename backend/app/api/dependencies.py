from functools import lru_cache

from redis.asyncio import Redis

from app.adapters.auth.keycloak import KeycloakJWTAuthenticator
from app.adapters.cache.memory import InMemoryIdempotencyStore, InMemoryRateLimiter
from app.adapters.cache.redis_store import RedisIdempotencyStore, RedisRateLimiter
from app.adapters.db.memory import InMemoryAuditRepository, InMemoryConversationRepository
from app.adapters.db.repositories import (
    SessionFactoryAuditRepository,
    SessionFactoryConversationRepository,
)
from app.adapters.db.session import SessionLocal
from app.adapters.rag.embeddings import get_embedding_model
from app.adapters.rag.hybrid import (
    HeuristicCrossEncoderReranker,
    InMemoryHybridRetriever,
    MilvusHybridRetriever,
    NativeMilvusHybridRetriever,
    ScoreReranker,
)
from app.adapters.rag.llm import MockLLMGenerator, OllamaLLMGenerator, OpenAILLMGenerator
from app.adapters.tools.mock_jira import MockJiraClient
from app.adapters.tools.mock_leave import MockLeaveClient
from app.agent.graph import SingleEmployeeSupportAgent
from app.application.guardrails import FaithfulnessPolicy
from app.application.policies import PolicyAuthorizer
from app.core.config import get_settings


def _build_llm():
    settings = get_settings()
    provider = settings.llm_provider.lower()
    if provider == "openai":
        return OpenAILLMGenerator(
            model=settings.openai_model,
            timeout_seconds=settings.openai_timeout_seconds,
        )
    if provider == "ollama":
        return OllamaLLMGenerator(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            timeout_seconds=settings.ollama_timeout_seconds,
        )
    return MockLLMGenerator()


def _build_repositories():
    settings = get_settings()
    if settings.persistence_backend.lower() == "postgres":
        return (
            SessionFactoryConversationRepository(SessionLocal),
            SessionFactoryAuditRepository(SessionLocal),
        )
    return InMemoryConversationRepository(), InMemoryAuditRepository()


def _build_cache():
    settings = get_settings()
    if settings.cache_backend.lower() == "redis":
        redis = Redis.from_url(settings.redis_url, decode_responses=False)
        return (
            RedisIdempotencyStore(redis),
            RedisRateLimiter(redis, settings.rate_limit_per_minute),
        )
    return (
        InMemoryIdempotencyStore(),
        InMemoryRateLimiter(settings.rate_limit_per_minute),
    )




def _build_embedding_model(settings):
    return get_embedding_model(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
        provider=settings.embedding_provider,
    )

def _build_retriever():
    settings = get_settings()
    common = {
        "dense_weight": settings.rag_dense_weight,
        "sparse_weight": settings.rag_sparse_weight,
        "retrieval_mode": settings.rag_retrieval_mode.lower(),
        "fusion_strategy": settings.rag_fusion_strategy.lower(),
        "candidate_multiplier": settings.rag_candidate_multiplier,
        "rrf_k": settings.rag_rrf_k,
        "bm25_k1": settings.rag_bm25_k1,
        "bm25_b": settings.rag_bm25_b,
        "embedding_model": _build_embedding_model(settings),
    }
    if settings.rag_backend.lower() == "milvus":
        retriever_cls = NativeMilvusHybridRetriever if settings.milvus_native_hybrid else MilvusHybridRetriever
        return retriever_cls(
            host=settings.milvus_host,
            port=settings.milvus_port,
            collection_name=settings.milvus_collection,
            vector_dim=settings.milvus_vector_dim,
            **common,
        )
    return InMemoryHybridRetriever(**common)


def _build_reranker():
    settings = get_settings()
    if not settings.rag_rerank_enabled:
        return ScoreReranker()
    if settings.rag_reranker_provider.lower() in {"heuristic", "cross_encoder", "heuristic_cross_encoder"}:
        return HeuristicCrossEncoderReranker()
    return ScoreReranker()


@lru_cache
def get_agent() -> SingleEmployeeSupportAgent:
    settings = get_settings()
    conversations, audit = _build_repositories()
    idempotency, rate_limiter = _build_cache()
    return SingleEmployeeSupportAgent(
        authenticator=KeycloakJWTAuthenticator(
            settings.keycloak_issuer,
            settings.keycloak_audience,
            settings.jwks_cache_seconds,
            allow_dev_token=settings.allow_dev_token,
        ),
        authorizer=PolicyAuthorizer(),
        conversations=conversations,
        audit=audit,
        idempotency=idempotency,
        rate_limiter=rate_limiter,
        retriever=_build_retriever(),
        reranker=_build_reranker(),
        llm=_build_llm(),
        jira=MockJiraClient(),
        leave=MockLeaveClient(),
        top_k=settings.rag_top_k,
        faithfulness_policy=FaithfulnessPolicy(
            min_score=settings.grounding_min_score,
            min_citations=settings.grounding_min_citations,
        ),
        confirmation_ttl_seconds=settings.confirmation_ttl_seconds,
    )
