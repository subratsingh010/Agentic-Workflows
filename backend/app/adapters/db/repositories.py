from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.db.models import AuditEventRow, CheckpointRow, ConversationMessageRow, PendingActionRow
from app.application.ports import AuditRepository, ConversationRepository
from app.domain.models import ActorContext, Intent, PendingAction


def _utc_naive(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _pending_from_row(row: PendingActionRow) -> PendingAction:
    return PendingAction(
        token=row.token,
        thread_id=row.thread_id,
        actor_subject=row.actor_subject,
        intent=Intent(row.intent),
        tool_payload=row.tool_payload,
        message=row.message,
        expires_at=row.expires_at.replace(tzinfo=timezone.utc) if row.expires_at.tzinfo is None else row.expires_at,
    )


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

    async def save_pending_action(self, action: PendingAction) -> None:
        values = {
            "token": action.token,
            "thread_id": action.thread_id,
            "actor_subject": action.actor_subject,
            "intent": action.intent.value,
            "tool_payload": action.tool_payload,
            "message": action.message,
            "expires_at": _utc_naive(action.expires_at),
        }
        stmt = insert(PendingActionRow).values(**values)
        stmt = stmt.on_conflict_do_update(
            index_elements=[PendingActionRow.token],
            set_=values,
        )
        await self._session.execute(stmt)
        await self._session.commit()

    async def load_pending_action(self, token: str) -> PendingAction | None:
        result = await self._session.execute(
            select(PendingActionRow).where(PendingActionRow.token == token)
        )
        row = result.scalar_one_or_none()
        if row is None:
            return None
        if row.expires_at <= datetime.utcnow():
            await self.delete_pending_action(token)
            return None
        return _pending_from_row(row)

    async def delete_pending_action(self, token: str) -> None:
        await self._session.execute(delete(PendingActionRow).where(PendingActionRow.token == token))
        await self._session.commit()


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

    async def save_pending_action(self, action: PendingAction) -> None:
        async with self._session_factory() as session:
            values = {
                "token": action.token,
                "thread_id": action.thread_id,
                "actor_subject": action.actor_subject,
                "intent": action.intent.value,
                "tool_payload": action.tool_payload,
                "message": action.message,
                "expires_at": _utc_naive(action.expires_at),
            }
            stmt = insert(PendingActionRow).values(**values)
            stmt = stmt.on_conflict_do_update(
                index_elements=[PendingActionRow.token],
                set_=values,
            )
            await session.execute(stmt)
            await session.commit()

    async def load_pending_action(self, token: str) -> PendingAction | None:
        async with self._session_factory() as session:
            result = await session.execute(
                select(PendingActionRow).where(PendingActionRow.token == token)
            )
            row = result.scalar_one_or_none()
            if row is None:
                return None
            if row.expires_at <= datetime.utcnow():
                await session.execute(delete(PendingActionRow).where(PendingActionRow.token == token))
                await session.commit()
                return None
            return _pending_from_row(row)

    async def delete_pending_action(self, token: str) -> None:
        async with self._session_factory() as session:
            await session.execute(delete(PendingActionRow).where(PendingActionRow.token == token))
            await session.commit()


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
