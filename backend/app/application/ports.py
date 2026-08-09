from abc import ABC, abstractmethod
from typing import Any

from app.domain.models import (
    ActorContext,
    ChatRequest,
    ChatResponse,
    Intent,
    JiraTimeLogRequest,
    JiraTimeLogResponse,
    LeaveApplicationRequest,
    LeaveApplicationResponse,
    PolicyChunk,
    RetrievedChunk,
)


class Authenticator(ABC):
    @abstractmethod
    async def authenticate(self, token: str) -> ActorContext:
        raise NotImplementedError


class Authorizer(ABC):
    @abstractmethod
    async def can_access_intent(self, actor: ActorContext, intent: Intent) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def can_access_policy_chunk(self, actor: ActorContext, chunk: PolicyChunk) -> bool:
        raise NotImplementedError


class ConversationRepository(ABC):
    @abstractmethod
    async def append_message(
        self, thread_id: str, role: str, content: str, metadata: dict[str, Any]
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def save_checkpoint(self, thread_id: str, state: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def load_checkpoint(self, thread_id: str) -> dict[str, Any] | None:
        raise NotImplementedError


class AuditRepository(ABC):
    @abstractmethod
    async def record(
        self, actor: ActorContext | None, action: str, outcome: str, metadata: dict[str, Any]
    ) -> None:
        raise NotImplementedError


class IdempotencyStore(ABC):
    @abstractmethod
    async def get(self, key: str) -> ChatResponse | None:
        raise NotImplementedError

    @abstractmethod
    async def put(self, key: str, response: ChatResponse, ttl_seconds: int) -> None:
        raise NotImplementedError


class RateLimiter(ABC):
    @abstractmethod
    async def check(self, key: str) -> None:
        raise NotImplementedError


class PolicyRetriever(ABC):
    @abstractmethod
    async def retrieve(self, query: str, actor: ActorContext, top_k: int) -> list[RetrievedChunk]:
        raise NotImplementedError


class Reranker(ABC):
    @abstractmethod
    async def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        raise NotImplementedError


class LLMGenerator(ABC):
    @abstractmethod
    async def answer_policy(self, query: str, chunks: list[RetrievedChunk]) -> str:
        raise NotImplementedError

    @abstractmethod
    async def classify_intent(self, message: str) -> Intent:
        raise NotImplementedError


class JiraClient(ABC):
    @abstractmethod
    async def get_time_logs(self, request: JiraTimeLogRequest) -> JiraTimeLogResponse:
        raise NotImplementedError


class LeaveClient(ABC):
    @abstractmethod
    async def apply_leave(self, request: LeaveApplicationRequest) -> LeaveApplicationResponse:
        raise NotImplementedError


class EmployeeSupportAgent(ABC):
    @abstractmethod
    async def run(self, request: ChatRequest, bearer_token: str) -> ChatResponse:
        raise NotImplementedError
