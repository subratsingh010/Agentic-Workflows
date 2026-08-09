import time

from app.application.ports import IdempotencyStore, RateLimiter
from app.domain.models import ChatResponse


class RateLimitExceeded(RuntimeError):
    pass


class InMemoryIdempotencyStore(IdempotencyStore):
    def __init__(self) -> None:
        self._items: dict[str, tuple[float, ChatResponse]] = {}

    async def get(self, key: str) -> ChatResponse | None:
        item = self._items.get(key)
        if not item:
            return None
        expires_at, response = item
        if expires_at < time.time():
            self._items.pop(key, None)
            return None
        return response

    async def put(self, key: str, response: ChatResponse, ttl_seconds: int) -> None:
        self._items[key] = (time.time() + ttl_seconds, response)


class InMemoryRateLimiter(RateLimiter):
    def __init__(self, limit_per_minute: int) -> None:
        self._limit = limit_per_minute
        self._hits: dict[str, list[float]] = {}

    async def check(self, key: str) -> None:
        now = time.time()
        window_start = now - 60
        hits = [hit for hit in self._hits.get(key, []) if hit >= window_start]
        if len(hits) >= self._limit:
            raise RateLimitExceeded("rate limit exceeded")
        hits.append(now)
        self._hits[key] = hits

