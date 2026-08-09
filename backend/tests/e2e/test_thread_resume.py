import pytest

from app.domain.models import ChatRequest


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_thread_id_is_reused(agent, conversations):
    first = await agent.run(ChatRequest(message="What is the leave policy?", thread_id="thread-1"), "ok")
    second = await agent.run(
        ChatRequest(message="Show my Jira time logs from 2026-08-01 to 2026-08-02", thread_id="thread-1"),
        "ok",
    )

    assert first.thread_id == second.thread_id == "thread-1"
    assert conversations.checkpoints["thread-1"]["thread_id"] == "thread-1"

