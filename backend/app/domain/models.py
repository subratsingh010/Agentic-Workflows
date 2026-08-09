from datetime import date, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Intent(StrEnum):
    POLICY_QA = "policy_qa"
    JIRA_TIME_LOG_LOOKUP = "jira_time_log_lookup"
    LEAVE_APPLICATION = "leave_application"
    SMALL_TALK = "small_talk"
    UNSUPPORTED_TOOL = "unsupported_tool"
    UNKNOWN = "unknown"


class ActorContext(BaseModel):
    subject: str
    employee_id: str
    email: str | None = None
    department: str | None = None
    country: str | None = None
    roles: set[str] = Field(default_factory=set)
    attributes: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    thread_id: str | None = None
    idempotency_key: str | None = Field(default=None, max_length=128)
    confirm: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    document_id: str
    title: str
    chunk_id: str
    score: float
    excerpt: str


class ChatResponse(BaseModel):
    thread_id: str
    message_id: UUID = Field(default_factory=uuid4)
    intent: Intent
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    requires_confirmation: bool = False
    missing_fields: list[str] = Field(default_factory=list)
    tool_result: dict[str, Any] | None = None


class PolicyChunk(BaseModel):
    document_id: str
    title: str
    chunk_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(PolicyChunk):
    dense_score: float = 0
    sparse_score: float = 0
    score: float = 0


class JiraTimeLogRequest(BaseModel):
    employee_id: str
    start_date: date
    end_date: date


class JiraTimeLogEntry(BaseModel):
    issue_key: str
    summary: str
    hours: float
    work_date: date


class JiraTimeLogResponse(BaseModel):
    employee_id: str
    entries: list[JiraTimeLogEntry]
    total_hours: float


class LeaveApplicationRequest(BaseModel):
    employee_id: str
    leave_type: Literal["vacation", "sick", "personal"]
    start_date: date
    end_date: date
    reason: str | None = Field(default=None, max_length=500)


class LeaveApplicationResponse(BaseModel):
    request_id: str
    status: Literal["submitted"]
    submitted_at: datetime
