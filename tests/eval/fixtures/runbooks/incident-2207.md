# INCIDENT-2207: checkout-service failures

**Severity:** P1
**Service:** checkout-service
**Owning team:** payments-team

## Summary

checkout-service returned ERR-503 for every logged-in user attempting to complete a purchase.
checkout-service depends on redis-cache to store in-progress cart and session state, and
redis-cache had gone fully unreachable due to a failed node in its cluster.

## Timeline

- redis-cache became unreachable after a cluster node failed without automatic failover.
- checkout-service could not read or write session state and returned ERR-503 for all
  authenticated checkout requests.
- The on-call engineer ran the flush-redis-cache runbook as part of bringing the cluster back to
  a healthy state, then confirmed checkout-service recovered once redis-cache was reachable
  again.

## Resolution

Resolved by the flush-redis-cache runbook combined with a manual redis-cache cluster failover.
