import pytest

from app.agent.graph import AuthorizationError
from app.domain.models import ChatRequest


@pytest.mark.asyncio
async def test_actor_without_employee_role_is_denied(agent):
    with pytest.raises(AuthorizationError):
        await agent.run(ChatRequest(message="Show my Jira time logs from 2026-08-01 to 2026-08-02"), "forbidden")

