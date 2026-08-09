from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "employee-support-agent"
    app_env: str = "local"
    database_url: str = "sqlite+aiosqlite:///./employee_support.db"
    redis_url: str = "redis://localhost:6379/0"
    persistence_backend: str = "memory"
    cache_backend: str = "memory"
    rag_backend: str = "memory"
    rabbitmq_url: str = "amqp://app:app@localhost:5672//"
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "employee_policy_chunks"
    milvus_vector_dim: int = 384
    milvus_native_hybrid: bool = True
    embedding_provider: str = "sentence_transformers"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "cpu"
    keycloak_issuer: str = "http://localhost:8080/realms/employee-support"
    keycloak_audience: str = "employee-support-api"
    jwks_cache_seconds: int = 300
    llm_provider: str = "mock"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    ollama_timeout_seconds: float = 30
    openai_model: str = "gpt-4o-mini"
    openai_timeout_seconds: float = 30
    rag_top_k: int = 5
    rag_dense_weight: float = 0.6
    rag_sparse_weight: float = 0.4
    rag_rerank_enabled: bool = True
    rag_retrieval_mode: str = "hybrid"
    rag_fusion_strategy: str = "weighted"
    rag_candidate_multiplier: int = 4
    rag_rrf_k: int = 60
    rag_bm25_k1: float = 1.5
    rag_bm25_b: float = 0.75
    rag_reranker_provider: str = "score"
    rag_chunk_size: int = 120
    rag_chunk_overlap: int = 20
    grounding_min_score: float = 0.35
    grounding_min_citations: int = 1
    policy_source_dir: str = "data/policies/source"
    rate_limit_per_minute: int = 60
    otel_exporter_otlp_endpoint: str | None = None
    phoenix_collector_endpoint: str | None = None
    log_level: str = Field(default="INFO")


@lru_cache
def get_settings() -> Settings:
    return Settings()

