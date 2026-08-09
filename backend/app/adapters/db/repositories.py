from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.db.models import AuditEventRow, CheckpointRow, ConversationMessageRow
from app.application.ports import AuditRepository, ConversationRepository
from app.domain.models import ActorContext


class SqlConversationRepository(ConversationRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def append_message(self, thread_id: str, role: str, content: str, metadata: dict[str, Any]) -> None:
        self._session.add(
            ConversationMessageRow(thread_id=thread_id, role=role, content=content, extra=metadata)
        )
        await self._session.commit()

    async def save_checkpoint(self, thread_id: str, state: dict[str, Any]) -> None:
        stmt = insert(CheckpointRow).values(thread_id=thread_id, state=state)
        stmt = stmt.on_conflict_do_update(
            index_elements=[CheckpointRow.thread_id],
            set_={"state": state},
        )
        await self._session.execute(stmt)
        await self._session.commit()

    async def load_checkpoint(self, thread_id: str) -> dict[str, Any] | None:
        result = await self._session.execute(
            select(CheckpointRow.state).where(CheckpointRow.thread_id == thread_id)
        )
        return result.scalar_one_or_none()


class SqlAuditRepository(AuditRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def record(
        self, actor: ActorContext | None, action: str, outcome: str, metadata: dict[str, Any]
    ) -> None:
        self._session.add(
            AuditEventRow(
                actor_subject=actor.subject if actor else None,
                action=action,
                outcome=outcome,
                extra=metadata,
            )
        )
        await self._session.commit()



class SessionFactoryConversationRepository(ConversationRepository):
    def __init__(self, session_factory) -> None:
        self._session_factory = session_factory

    async def append_message(self, thread_id: str, role: str, content: str, metadata: dict[str, Any]) -> None:
        async with self._session_factory() as session:
            session.add(
                ConversationMessageRow(thread_id=thread_id, role=role, content=content, extra=metadata)
            )
            await session.commit()

    async def save_checkpoint(self, thread_id: str, state: dict[str, Any]) -> None:
        async with self._session_factory() as session:
            stmt = insert(CheckpointRow).values(thread_id=thread_id, state=state)
            stmt = stmt.on_conflict_do_update(
                index_elements=[CheckpointRow.thread_id],
                set_={"state": state},
            )
            await session.execute(stmt)
            await session.commit()

    async def load_checkpoint(self, thread_id: str) -> dict[str, Any] | None:
        async with self._session_factory() as session:
            result = await session.execute(
                select(CheckpointRow.state).where(CheckpointRow.thread_id == thread_id)
            )
            return result.scalar_one_or_none()


class SessionFactoryAuditRepository(AuditRepository):
    def __init__(self, session_factory) -> None:
        self._session_factory = session_factory

    async def record(
        self, actor: ActorContext | None, action: str, outcome: str, metadata: dict[str, Any]
    ) -> None:
        async with self._session_factory() as session:
            session.add(
                AuditEventRow(
                    actor_subject=actor.subject if actor else None,
                    action=action,
                    outcome=outcome,
                    extra=metadata,
                )
            )
            await session.commit()
