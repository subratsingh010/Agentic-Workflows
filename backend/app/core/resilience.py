import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from time import monotonic
from typing import TypeVar

from tenacity import retry, stop_after_attempt, wait_exponential

T = TypeVar("T")


class CircuitOpenError(RuntimeError):
    pass


@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    reset_seconds: float = 30
    failures: int = 0
    opened_at: float | None = None

    async def call(self, fn: Callable[[], Awaitable[T]]) -> T:
        if self.opened_at and monotonic() - self.opened_at < self.reset_seconds:
            raise CircuitOpenError("circuit breaker is open")
        try:
            result = await fn()
        except Exception:
            self.failures += 1
            if self.failures >= self.failure_threshold:
                self.opened_at = monotonic()
            raise
        self.failures = 0
        self.opened_at = None
        return result


class ConcurrencyLimiter:
    def __init__(self, limit: int) -> None:
        self._semaphore = asyncio.Semaphore(limit)

    async def run(self, fn: Callable[[], Awaitable[T]]) -> T:
        async with self._semaphore:
            return await fn()


async def with_timeout(fn: Callable[[], Awaitable[T]], timeout_seconds: float) -> T:
    return await asyncio.wait_for(fn(), timeout=timeout_seconds)


retry_external = retry(wait=wait_exponential(multiplier=0.2, min=0.2, max=2), stop=stop_after_attempt(3))

