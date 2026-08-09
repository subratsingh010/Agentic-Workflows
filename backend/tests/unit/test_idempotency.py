import pytest

from app.domain.models import ChatRequest


@pytest.mark.asyncio
async def test_idempotency_returns_same_response(agent):
    request = ChatRequest(
        message="Show my Jira time logs from 2026-08-01 to 2026-08-02",
        idempotency_key="jira-1",
    )
    first = await agent.run(request, "ok")
    second = await agent.run(request, "ok")

    assert first.message_id == second.message_id
    assert first.tool_result == second.tool_result

