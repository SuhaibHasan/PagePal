from __future__ import annotations

import time
import uuid

import redis

RATE_LIMIT_MAX_REQUESTS = 10
RATE_LIMIT_WINDOW_SECONDS = 60


def is_within_rate_limit(redis_client: redis.Redis, session_id: str) -> bool:
    # Sliding window log: a sorted set per session scored by request time, so
    # the window slides continuously rather than resetting on fixed boundaries.
    key = f"psb:ratelimit:{session_id}"
    now = time.time()

    redis_client.zremrangebyscore(key, 0, now - RATE_LIMIT_WINDOW_SECONDS)
    if redis_client.zcard(key) >= RATE_LIMIT_MAX_REQUESTS:
        return False

    # member must be unique per request - timestamp alone can collide under load
    redis_client.zadd(key, {f"{now}-{uuid.uuid4().hex}": now})
    redis_client.expire(key, RATE_LIMIT_WINDOW_SECONDS)
    return True
