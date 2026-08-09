import pytest

from app.adapters.rag.llm import MockLLMGenerator
from app.application.guardrails import FaithfulnessPolicy
from app.domain.models import ChatRequest, Intent


@pytest.mark.asyncio
async def test_policy_qa_returns_citations(agent):
    response = await agent.run(ChatRequest(message="What is the leave policy?"), "ok")

    assert response.intent == Intent.POLICY_QA
    assert response.citations
    assert response.answer_grounding.grounded
    assert response.answer_grounding.guardrail_action == "pass"
    assert response.citations[0].evidence.support_score >= 0
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


class HallucinatingPolicyLLM(MockLLMGenerator):
    async def answer_policy(self, query, chunks):
        return "Employees receive unlimited submarine commuting benefits from the moon office."


@pytest.mark.asyncio
async def test_policy_qa_blocks_ungrounded_answer(agent):
    agent.llm = HallucinatingPolicyLLM()
    agent.faithfulness_policy = FaithfulnessPolicy(min_score=0.8, min_citations=1)

    response = await agent.run(ChatRequest(message="What is the leave policy?"), "ok")

    assert response.intent == Intent.POLICY_QA
    assert not response.answer_grounding.grounded
    assert response.answer_grounding.guardrail_action == "blocked"
    assert not response.citations
    assert "not have enough grounded policy context" in response.answer.lower()
