from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

from wiki.schema import WikiEntry


def make_entry(**overrides) -> WikiEntry:
    now = datetime.now(UTC)
    defaults = {
        "title": "payment-service ERR-503 under connection pool exhaustion",
        "summary": "Payments fail when postgres-primary's connection pool is exhausted.",
        "confidence": 0.9,
        "created_at": now,
        "last_updated": now,
        "last_validated": now,
    }
    defaults.update(overrides)
    return WikiEntry(**defaults)


def test_id_is_auto_generated_from_title_and_sorted_tags():
    entry = make_entry(tags=["postgres", "payment-service"])

    expected = hashlib.sha256(f"{entry.title}|payment-service,postgres".encode()).hexdigest()
    assert entry.id == expected


def test_id_generation_is_order_independent_for_tags():
    a = make_entry(tags=["postgres", "payment-service"])
    b = make_entry(tags=["payment-service", "postgres"])

    assert a.id == b.id


def test_explicit_id_is_not_overwritten():
    entry = make_entry(id="explicit-id-123")

    assert entry.id == "explicit-id-123"


def test_is_stale_true_when_last_updated_older_than_ttl():
    stale_time = datetime.now(UTC) - timedelta(days=45)
    entry = make_entry(last_updated=stale_time, ttl_days=30)

    assert entry.is_stale is True


def test_is_stale_false_when_last_updated_within_ttl():
    recent_time = datetime.now(UTC) - timedelta(days=5)
    entry = make_entry(last_updated=recent_time, ttl_days=30)

    assert entry.is_stale is False


def test_optional_fields_default_correctly_when_omitted():
    entry = make_entry()

    assert entry.root_cause is None
    assert entry.severity_pattern is None
    assert entry.avg_resolution_time is None
    assert entry.resolution_steps == []
    assert entry.affected_services == []
    assert entry.related_error_codes == []
    assert entry.tags == []
    assert entry.source_refs == []
    assert entry.hit_count == 0
    assert entry.ttl_days == 30
    assert entry.source_content_hash is None
