# INCIDENT-4521: payment-service outage

**Severity:** P1
**Service:** payment-service
**Owning team:** payments-team

## Summary

payment-service became fully unresponsive for 22 minutes. All checkout attempts failed with
ERR-503. The root cause was postgres-primary's connection pool being exhausted after a
deployment reduced the configured pool size while traffic was above baseline.

## Timeline

- payment-service started returning ERR-503 on every request that touched the database.
- postgres-primary showed a connection pool of 0 available connections for the payment-service
  role.
- The on-call engineer ran the scale-postgres-connections runbook to raise the pool size, which
  resolved the incident.

## Resolution

Resolved by the scale-postgres-connections runbook. A follow-up ticket was filed to make the
pool size configuration part of the deployment review checklist.
