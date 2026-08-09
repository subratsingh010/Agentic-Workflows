import time

from app.application.ports import IdempotencyStore, RateLimiter
from app.domain.models import ChatResponse


class RateLimitExceeded(RuntimeError):
    pass


class InMemoryIdempotencyStore(IdempotencyStore):
    def __init__(self) -> None:
        self._items: dict[str, tuple[float, ChatResponse]] = {}
        self._reservations: dict[str, float] = {}

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

    async def reserve(self, key: str, ttl_seconds: int) -> bool:
        now = time.time()
        expires_at = self._reservations.get(key)
        if expires_at and expires_at > now:
            return False
        self._reservations[key] = now + ttl_seconds
        return True

    async def release(self, key: str) -> None:
        self._reservations.pop(key, None)


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

