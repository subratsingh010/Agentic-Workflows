import json
import time

from redis.asyncio import Redis

from app.adapters.cache.memory import RateLimitExceeded
from app.application.ports import IdempotencyStore, RateLimiter
from app.domain.models import ChatResponse


class RedisIdempotencyStore(IdempotencyStore):
    def __init__(self, redis: Redis, prefix: str = "idempotency") -> None:
        self._redis = redis
        self._prefix = prefix

    async def get(self, key: str) -> ChatResponse | None:
        value = await self._redis.get(f"{self._prefix}:{key}")
        if value is None:
            return None
        raw = value.decode("utf-8") if isinstance(value, bytes) else value
        return ChatResponse.model_validate_json(raw)

    async def put(self, key: str, response: ChatResponse, ttl_seconds: int) -> None:
        await self._redis.set(
            f"{self._prefix}:{key}",
            response.model_dump_json(),
            ex=ttl_seconds,
        )

    async def reserve(self, key: str, ttl_seconds: int) -> bool:
        return bool(await self._redis.set(f"{self._prefix}:lock:{key}", "1", ex=ttl_seconds, nx=True))

    async def release(self, key: str) -> None:
        await self._redis.delete(f"{self._prefix}:lock:{key}")


class RedisRateLimiter(RateLimiter):
    def __init__(self, redis: Redis, limit_per_minute: int, prefix: str = "rate-limit") -> None:
        self._redis = redis
        self._limit = limit_per_minute
        self._prefix = prefix

    async def check(self, key: str) -> None:
        now_ms = int(time.time() * 1000)
        window_start_ms = now_ms - 60_000
        redis_key = f"{self._prefix}:{key}"
        async with self._redis.pipeline(transaction=True) as pipe:
            await pipe.zremrangebyscore(redis_key, 0, window_start_ms)
            await pipe.zadd(redis_key, {str(now_ms): now_ms})
            await pipe.zcard(redis_key)
            await pipe.expire(redis_key, 60)
            results = await pipe.execute()
        hit_count = int(results[2])
        if hit_count > self._limit:
            raise RateLimitExceeded("rate limit exceeded")
