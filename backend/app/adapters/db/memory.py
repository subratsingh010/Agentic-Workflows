from datetime import datetime, timezone
from typing import Any

from app.application.ports import AuditRepository, ConversationRepository
from app.domain.models import ActorContext, PendingAction


class InMemoryConversationRepository(ConversationRepository):
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.checkpoints: dict[str, dict[str, Any]] = {}
        self.pending_actions: dict[str, PendingAction] = {}

    async def append_message(
        self, thread_id: str, role: str, content: str, metadata: dict[str, Any]
    ) -> None:
        self.messages.append(
            {"thread_id": thread_id, "role": role, "content": content, "metadata": metadata}
        )

    async def save_checkpoint(self, thread_id: str, state: dict[str, Any]) -> None:
        self.checkpoints[thread_id] = state

    async def load_checkpoint(self, thread_id: str) -> dict[str, Any] | None:
        return self.checkpoints.get(thread_id)

    async def save_pending_action(self, action: PendingAction) -> None:
        self.pending_actions[action.token] = action

    async def load_pending_action(self, token: str) -> PendingAction | None:
        action = self.pending_actions.get(token)
        if action is None:
            return None
        expires_at = action.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at <= datetime.now(timezone.utc):
            self.pending_actions.pop(token, None)
            return None
        return action

    async def delete_pending_action(self, token: str) -> None:
        self.pending_actions.pop(token, None)


class InMemoryAuditRepository(AuditRepository):
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def record(
        self, actor: ActorContext | None, action: str, outcome: str, metadata: dict[str, Any]
    ) -> None:
        self.events.append(
            {
                "actor_subject": actor.subject if actor else None,
                "action": action,
                "outcome": outcome,
                "metadata": metadata,
            }
        )

