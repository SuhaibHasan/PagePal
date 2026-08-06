import time

from api.rate_limit import RATE_LIMIT_MAX_REQUESTS, is_within_rate_limit


class _FakeRedis:
    """Minimal in-memory stand-in for the sorted-set calls the rate limiter uses."""

    def __init__(self) -> None:
        self._sets: dict[str, dict[str, float]] = {}

    def zremrangebyscore(self, key: str, min_score: float, max_score: float) -> None:
        members = self._sets.get(key, {})
        self._sets[key] = {m: s for m, s in members.items() if not (min_score <= s <= max_score)}

    def zcard(self, key: str) -> int:
        return len(self._sets.get(key, {}))

    def zadd(self, key: str, mapping: dict[str, float]) -> None:
        self._sets.setdefault(key, {}).update(mapping)

    def expire(self, key: str, seconds: int) -> None:
        pass


def test_allows_requests_under_the_limit():
    redis_client = _FakeRedis()

    results = [is_within_rate_limit(redis_client, "session-1") for _ in range(RATE_LIMIT_MAX_REQUESTS)]

    assert all(results)


def test_rejects_the_request_beyond_the_limit():
    redis_client = _FakeRedis()
    for _ in range(RATE_LIMIT_MAX_REQUESTS):
        is_within_rate_limit(redis_client, "session-1")

    assert is_within_rate_limit(redis_client, "session-1") is False


def test_sessions_are_rate_limited_independently():
    redis_client = _FakeRedis()
    for _ in range(RATE_LIMIT_MAX_REQUESTS):
        is_within_rate_limit(redis_client, "session-1")

    assert is_within_rate_limit(redis_client, "session-2") is True


def test_requests_outside_the_window_are_evicted_and_free_up_capacity():
    redis_client = _FakeRedis()
    key = "psb:ratelimit:session-1"
    old_time = time.time() - 61
    redis_client.zadd(key, {f"old-{i}": old_time for i in range(RATE_LIMIT_MAX_REQUESTS)})

    assert is_within_rate_limit(redis_client, "session-1") is True
