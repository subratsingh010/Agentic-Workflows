from datetime import datetime
from uuid import uuid4

from sqlalchemy import JSON, DateTime, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ConversationMessageRow(Base):
    __tablename__ = "conversation_messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    thread_id: Mapped[str] = mapped_column(String(128), index=True)
    role: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text)
    extra: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class CheckpointRow(Base):
    __tablename__ = "agent_checkpoints"

    thread_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    state: Mapped[dict] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AuditEventRow(Base):
    __tablename__ = "audit_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    actor_subject: Mapped[str | None] = mapped_column(String(128), nullable=True)
    action: Mapped[str] = mapped_column(String(128))
    outcome: Mapped[str] = mapped_column(String(64))
    extra: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

