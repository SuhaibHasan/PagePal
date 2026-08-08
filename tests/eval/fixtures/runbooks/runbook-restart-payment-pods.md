# Runbook: restart-payment-pods

**Service:** payment-service
**Owning team:** payments-team

## When to use this

Use this runbook when payment-service is returning ERR-503 or ERR-500 and restarting the pods is
a safe first mitigation while root-causing continues, such as during INCIDENT-4521-style
postgres-primary connection pool exhaustion.

## Steps

1. Confirm payment-service error rate is elevated and postgres-primary connection usage is at or
   near its configured limit.
2. Scale payment-service down to zero replicas, then back up to the standard replica count. This
   clears any pods holding stale, leaked database connections.
3. If ERR-503 continues after the restart, follow the scale-postgres-connections runbook to raise
   postgres-primary's connection pool size instead.
4. Confirm payment-service latency and error rate return to baseline before closing out.
