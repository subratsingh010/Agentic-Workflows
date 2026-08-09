import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("PERSISTENCE_BACKEND", "memory")
os.environ.setdefault("CACHE_BACKEND", "memory")
os.environ.setdefault("RAG_BACKEND", "memory")
os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("EMBEDDING_PROVIDER", "sentence_transformers")
os.environ.setdefault("MILVUS_VECTOR_DIM", "384")
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = ""
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = ""

import pytest

from app.adapters.cache.memory import InMemoryIdempotencyStore, InMemoryRateLimiter
from app.adapters.db.memory import InMemoryAuditRepository, InMemoryConversationRepository
from app.adapters.rag.hybrid import InMemoryHybridRetriever, ScoreReranker
from app.adapters.rag.llm import MockLLMGenerator
from app.adapters.tools.mock_jira import MockJiraClient
from app.adapters.tools.mock_leave import MockLeaveClient
from app.agent.graph import SingleEmployeeSupportAgent
from app.application.policies import PolicyAuthorizer
from app.application.ports import Authenticator
from app.domain.models import ActorContext
from tests.support_embeddings import SupportEmbeddingModel


class TestAuthenticator(Authenticator):
    async def authenticate(self, token: str) -> ActorContext:
        if token == "forbidden":
            return ActorContext(subject="guest", employee_id="E0000", roles={"guest"})
        return ActorContext(
            subject="user-1",
            employee_id="E1001",
            email="user@example.com",
            department="engineering",
            country="US",
            roles={"employee", "manager"},
        )


@pytest.fixture
def conversations() -> InMemoryConversationRepository:
    return InMemoryConversationRepository()


@pytest.fixture
def agent(conversations: InMemoryConversationRepository) -> SingleEmployeeSupportAgent:
    return SingleEmployeeSupportAgent(
        authenticator=TestAuthenticator(),
        authorizer=PolicyAuthorizer(),
        conversations=conversations,
        audit=InMemoryAuditRepository(),
        idempotency=InMemoryIdempotencyStore(),
        rate_limiter=InMemoryRateLimiter(100),
        retriever=InMemoryHybridRetriever(embedding_model=SupportEmbeddingModel()),
        reranker=ScoreReranker(),
        llm=MockLLMGenerator(),
        jira=MockJiraClient(),
        leave=MockLeaveClient(),
        top_k=5,
    )

