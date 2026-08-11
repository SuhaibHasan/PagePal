# Runbook: scale-postgres-connections

**Dependency:** postgres-primary
**Owning team:** platform-team
**Depended on by:** payment-service

## When to use this

Use this runbook when payment-service is returning ERR-503 because postgres-primary's connection
pool for the payment-service role is exhausted, as happened in INCIDENT-4521.

## Steps

1. Confirm postgres-primary shows zero or near-zero available connections for the
   payment-service role.
2. Raise the configured connection pool size for payment-service's role on postgres-primary.
3. Restart payment-service pods (see restart-payment-pods) so they pick up the new pool size.
4. Add the new pool size to the deployment review checklist so future deploys don't silently
   shrink it again.
