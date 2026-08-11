# INCIDENT-3311: auth-service degraded

**Severity:** P2
**Service:** auth-service
**Owning team:** platform-team

## Summary

auth-service login latency climbed to over 8 seconds and a fraction of requests failed with
ERR-500. redis-cache, which auth-service depends on for session storage, was under heavy memory
pressure and had started evicting keys aggressively, forcing auth-service to fall back to slow
database lookups for every request.

## Timeline

- redis-cache memory usage reached 98% and eviction rate spiked.
- auth-service latency and ERR-500 error rate both increased in lockstep with the eviction rate.
- The on-call engineer ran the flush-redis-cache runbook to clear stale session keys and reduce
  memory pressure.

## Resolution

Resolved by the flush-redis-cache runbook. platform-team added a memory usage alert on
redis-cache to catch this earlier next time.
