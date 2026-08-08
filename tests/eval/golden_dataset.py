from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

QueryCategory = Literal["semantic", "exact_match", "relationship"]


@dataclass(frozen=True)
class GoldenQA:
    id: str
    category: QueryCategory
    question: str
    ground_truth: str


# Golden dataset for tests/eval/ragas_eval.py. Answers are grounded in the
# fictional incident/runbook corpus under tests/eval/fixtures/runbooks/ -
# ingest that directory before running the eval.
#
# - semantic: paraphrased, no exact identifiers - best suited to vector search.
# - exact_match: precise identifiers (error codes, incident/service names) -
#   best suited to keyword/BM25 search.
# - relationship: multi-hop ownership/dependency/causal questions - best
#   suited to graph traversal.
GOLDEN_DATASET: list[GoldenQA] = [
    # -- semantic --------------------------------------------------------
    GoldenQA(
        id="sem-1",
        category="semantic",
        question="Why did customers experience payment failures recently?",
        ground_truth=(
            "payment-service went down because postgres-primary's connection pool was "
            "exhausted after a deployment reduced the pool size while traffic was above "
            "baseline (INCIDENT-4521); it was resolved by raising the connection pool size."
        ),
    ),
    GoldenQA(
        id="sem-2",
        category="semantic",
        question="What's the usual fix when the auth service starts acting up under heavy load?",
        ground_truth=(
            "Flush redis-cache to relieve memory pressure and reduce aggressive key eviction, "
            "which is what was causing auth-service's slow logins and errors."
        ),
    ),
    GoldenQA(
        id="sem-3",
        category="semantic",
        question="Our checkout flow keeps timing out for logged-in users - what's usually to blame?",
        ground_truth=(
            "redis-cache being unreachable or unhealthy, since checkout-service depends on it "
            "for in-progress cart and session state; recovering redis-cache (including a "
            "cluster failover if a node failed) resolves it."
        ),
    ),
    GoldenQA(
        id="sem-4",
        category="semantic",
        question="Notifications seem delayed for a lot of users lately - what could cause that?",
        ground_truth=(
            "rabbitmq-broker's disk filling up, which makes it reject new publishes and backs "
            "up notification-service's outbound queue; freeing disk space and restarting the "
            "broker drains the backlog."
        ),
    ),
    GoldenQA(
        id="sem-5",
        category="semantic",
        question="If a user complains they keep getting logged out unexpectedly, what should we check?",
        ground_truth=(
            "Check for ERR-401s caused by an auth-service signing key rotation without an "
            "overlap window; accept both old and new keys temporarily and re-issue tokens for "
            "affected sessions."
        ),
    ),
    GoldenQA(
        id="sem-6",
        category="semantic",
        question="Which team should I loop in if the payment system is having trouble?",
        ground_truth="payments-team owns payment-service.",
    ),
    GoldenQA(
        id="sem-7",
        category="semantic",
        question="What should an on-call engineer try first when payment-service is unresponsive?",
        ground_truth=(
            "Restart payment-service pods as a first mitigation; if errors continue, raise "
            "postgres-primary's connection pool size for the payment-service role."
        ),
    ),
    # -- exact_match -------------------------------------------------------
    GoldenQA(
        id="exact-1",
        category="exact_match",
        question="What does error code ERR-503 mean in this system?",
        ground_truth=(
            "ERR-503 indicates a service couldn't obtain a needed resource - e.g. "
            "payment-service returns it when postgres-primary's connection pool is exhausted, "
            "and checkout-service returns it when redis-cache is unreachable."
        ),
    ),
    GoldenQA(
        id="exact-2",
        category="exact_match",
        question="What causes ERR-429 in notification-service?",
        ground_truth="rabbitmq-broker's disk filling up, which makes it reject new publishes.",
    ),
    GoldenQA(
        id="exact-3",
        category="exact_match",
        question="What is INCIDENT-4521?",
        ground_truth=(
            "A payment-service outage caused by postgres-primary's connection pool being "
            "exhausted; resolved by the scale-postgres-connections runbook."
        ),
    ),
    GoldenQA(
        id="exact-4",
        category="exact_match",
        question="Which runbook resolves INCIDENT-3311?",
        ground_truth="The flush-redis-cache runbook.",
    ),
    GoldenQA(
        id="exact-5",
        category="exact_match",
        question="What does ERR-401 indicate in auth-service?",
        ground_truth="An expired or invalid auth token, often from a signing key rotation without overlap.",
    ),
    GoldenQA(
        id="exact-6",
        category="exact_match",
        question="Besides auth-service, which other service depends on redis-cache for session state?",
        ground_truth="checkout-service.",
    ),
    GoldenQA(
        id="exact-7",
        category="exact_match",
        question="What is the exact name of the runbook used to fix rabbitmq-broker disk-full issues?",
        ground_truth="restart-rabbitmq-broker.",
    ),
    # -- relationship --------------------------------------------------------
    GoldenQA(
        id="rel-1",
        category="relationship",
        question="What incidents were caused by redis-cache issues?",
        ground_truth="INCIDENT-3311 (auth-service) and INCIDENT-2207 (checkout-service).",
    ),
    GoldenQA(
        id="rel-2",
        category="relationship",
        question="Which services depend on redis-cache, and what team owns it?",
        ground_truth="auth-service and checkout-service depend on redis-cache; platform-team owns it.",
    ),
    GoldenQA(
        id="rel-3",
        category="relationship",
        question="If postgres-primary goes down, which service and team are impacted?",
        ground_truth="payment-service, owned by payments-team, depends on postgres-primary.",
    ),
    GoldenQA(
        id="rel-4",
        category="relationship",
        question=(
            "What is the chain of events that led from postgres-primary's connection pool "
            "exhaustion to a customer-facing incident?"
        ),
        ground_truth=(
            "postgres-primary's connection pool exhaustion caused payment-service to return "
            "ERR-503, which caused INCIDENT-4521, which was resolved by the "
            "scale-postgres-connections runbook."
        ),
    ),
    GoldenQA(
        id="rel-5",
        category="relationship",
        question="Which team should be paged if rabbitmq-broker is degraded, given what depends on it?",
        ground_truth=(
            "messaging-team, since it owns rabbitmq-broker and notification-service (which "
            "depends on it)."
        ),
    ),
    GoldenQA(
        id="rel-6",
        category="relationship",
        question="How is checkout-service related to auth-service's incident history?",
        ground_truth=(
            "Both depend on redis-cache, so a redis-cache problem can affect both - it caused "
            "INCIDENT-3311 for auth-service and INCIDENT-2207 for checkout-service."
        ),
    ),
]
