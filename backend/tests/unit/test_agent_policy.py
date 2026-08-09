import pytest

from app.domain.models import ChatRequest, Intent


@pytest.mark.asyncio
async def test_policy_qa_returns_citations(agent):
    response = await agent.run(ChatRequest(message="What is the leave policy?"), "ok")

    assert response.intent == Intent.POLICY_QA
    assert response.citations
    assert "policy" in response.answer.lower()



@pytest.mark.asyncio
async def test_greeting_returns_helpful_response(agent):
    response = await agent.run(ChatRequest(message="hi?"), "ok")

    assert response.intent == Intent.SMALL_TALK
    assert "policy" in response.answer.lower()
    assert "jira" in response.answer.lower()


@pytest.mark.asyncio
async def test_how_are_you_returns_small_talk(agent):
    response = await agent.run(ChatRequest(message="how are u"), "ok")

    assert response.intent == Intent.SMALL_TALK
    assert "policy" in response.answer.lower()


@pytest.mark.asyncio
async def test_unsupported_jira_ticket_creation_is_explained(agent):
    response = await agent.run(ChatRequest(message="can u create jira ticket?"), "ok")

    assert response.intent == Intent.UNSUPPORTED_TOOL
    assert "cannot create jira tickets" in response.answer.lower()
