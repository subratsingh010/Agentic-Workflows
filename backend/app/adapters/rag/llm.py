import httpx

from app.application.ports import LLMGenerator
from app.domain.models import Intent, RetrievedChunk


class RuleBasedIntentClassifier:
    async def classify_intent(self, message: str) -> Intent:
        lowered = message.lower().strip(" ?.!")
        if lowered in {"hi", "hello", "hey", "good morning", "good afternoon", "good evening", "how are u", "how are you", "how r u"}:
            return Intent.SMALL_TALK
        if "jira" in lowered and any(action in lowered for action in ("create", "open", "raise", "file")) and "ticket" in lowered:
            return Intent.UNSUPPORTED_TOOL
        if "policy" in lowered or "eligible" in lowered or "how do i" in lowered:
            if "apply" not in lowered and "submit" not in lowered:
                return Intent.POLICY_QA
        if "time" in lowered or "jira" in lowered or "log" in lowered:
            return Intent.JIRA_TIME_LOG_LOOKUP
        if "leave" in lowered or "vacation" in lowered or "sick" in lowered:
            return Intent.LEAVE_APPLICATION
        return Intent.UNKNOWN


class MockLLMGenerator(LLMGenerator):
    def __init__(self) -> None:
        self._classifier = RuleBasedIntentClassifier()

    async def classify_intent(self, message: str) -> Intent:
        return await self._classifier.classify_intent(message)

    async def answer_policy(self, query: str, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "I could not find an accessible policy source for that question."
        return "Based on the accessible policy sources: " + " ".join(chunk.text for chunk in chunks[:2])


class OllamaLLMGenerator(LLMGenerator):
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1",
        timeout_seconds: float = 30,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._classifier = RuleBasedIntentClassifier()
        self._fallback = MockLLMGenerator()

    async def classify_intent(self, message: str) -> Intent:
        # Identity, permissions, and protected action routing remain backend controlled.
        return await self._classifier.classify_intent(message)

    async def answer_policy(self, query: str, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "I could not find an accessible policy source for that question."
        context = "\n\n".join(
            f"Source {index + 1}: {chunk.title} ({chunk.chunk_id})\n{chunk.text}"
            for index, chunk in enumerate(chunks[:5])
        )
        prompt = (
            "You are an enterprise employee support assistant. Answer only from the provided "
            "policy context. If the context is insufficient, say so. Keep the answer concise "
            "and do not make access-control, identity, or permission decisions.\n\n"
            f"Policy context:\n{context}\n\nEmployee question:\n{query}\n\nAnswer:"
        )
        try:
            async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                response = await client.post(
                    f"{self._base_url}/api/generate",
                    json={"model": self._model, "prompt": prompt, "stream": False},
                )
                response.raise_for_status()
        except httpx.HTTPError:
            return await self._fallback.answer_policy(query, chunks)
        data = response.json()
        answer = str(data.get("response", "")).strip()
        return answer or await self._fallback.answer_policy(query, chunks)


class OpenAILLMGenerator(LLMGenerator):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        timeout_seconds: float = 30,
    ) -> None:
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._classifier = RuleBasedIntentClassifier()
        self._fallback = MockLLMGenerator()

    async def classify_intent(self, message: str) -> Intent:
        # Identity, permissions, and protected action routing remain backend controlled.
        return await self._classifier.classify_intent(message)

    async def answer_policy(self, query: str, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "I could not find an accessible policy source for that question."
        context = "\n\n".join(
            f"Source {index + 1}: {chunk.title} ({chunk.chunk_id})\n{chunk.text}"
            for index, chunk in enumerate(chunks[:5])
        )
        instructions = (
            "You are an enterprise employee support assistant. Answer only from the provided "
            "policy context. If the context is insufficient, say so. Keep the answer concise. "
            "Do not make access-control, identity, or permission decisions."
        )
        user_input = f"Policy context:\n{context}\n\nEmployee question:\n{query}"
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(timeout=self._timeout_seconds)
            response = await client.responses.create(
                model=self._model,
                instructions=instructions,
                input=user_input,
            )
        except Exception:
            return await self._fallback.answer_policy(query, chunks)
        answer = str(getattr(response, "output_text", "")).strip()
        return answer or await self._fallback.answer_policy(query, chunks)
