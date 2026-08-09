from typing import Any, TypedDict

from app.domain.models import (
    ActorContext,
    ChatRequest,
    ChatResponse,
    Citation,
    GroundingReport,
    Intent,
    RetrievedChunk,
)


class AgentState(TypedDict, total=False):
    request: ChatRequest
    token: str
    actor: ActorContext
    thread_id: str
    intent: Intent
    missing_fields: list[str]
    retrieved_chunks: list[RetrievedChunk]
    reranked_chunks: list[RetrievedChunk]
    answer: str
    tool_payload: dict[str, Any]
    tool_result: dict[str, Any]
    citations: list[Citation]
    answer_grounding: GroundingReport
    requires_confirmation: bool
    idempotent_response: ChatResponse
    response: ChatResponse

