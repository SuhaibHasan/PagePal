# Runbook: flush-redis-cache

**Dependency:** redis-cache
**Owning team:** platform-team
**Depended on by:** auth-service, checkout-service

## When to use this

Use this runbook when redis-cache is under memory pressure, is evicting keys aggressively, or has
become unreachable, which shows up downstream as ERR-500 in auth-service (see INCIDENT-3311) or
ERR-503 in checkout-service (see INCIDENT-2207).

## Steps

1. Check redis-cache memory usage and eviction rate, and check cluster node health.
2. If memory pressure is the cause, flush non-essential cached keys (not active session keys) to
   free memory immediately.
3. If a cluster node has failed, trigger a manual failover to promote a healthy replica.
4. Confirm auth-service latency and checkout-service error rate return to baseline once
   redis-cache is healthy again.
