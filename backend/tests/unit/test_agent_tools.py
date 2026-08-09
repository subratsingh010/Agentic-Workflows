import pytest

from app.domain.models import ChatRequest, Intent


@pytest.mark.asyncio
async def test_jira_time_log_lookup_uses_typed_tool(agent):
    response = await agent.run(
        ChatRequest(message="Show my Jira time logs from 2026-08-01 to 2026-08-02"),
        "ok",
    )

    assert response.intent == Intent.JIRA_TIME_LOG_LOOKUP
    assert response.tool_result
    assert response.tool_result["total_hours"] == 7.5


@pytest.mark.asyncio
async def test_leave_application_requires_confirmation(agent):
    response = await agent.run(
        ChatRequest(message="Apply vacation leave from 2026-09-01 to 2026-09-03"),
        "ok",
    )

    assert response.intent == Intent.LEAVE_APPLICATION
    assert response.requires_confirmation is True
    assert response.tool_result is None


@pytest.mark.asyncio
async def test_confirmed_leave_application_executes_tool(agent):
    response = await agent.run(
        ChatRequest(
            message="Apply vacation leave from 2026-09-01 to 2026-09-03",
            confirm=True,
            idempotency_key="leave-1",
        ),
        "ok",
    )

    assert response.requires_confirmation is False
    assert response.tool_result
    assert response.tool_result["status"] == "submitted"

