# Service ownership and dependency overview

This document summarizes which team owns which service, and which shared infrastructure each
service depends on. Use it as the reference for "who owns this" and "what depends on what"
questions.

## payments-team

- Owns **payment-service**, which depends on **postgres-primary** for order and payment records.
- Owns **checkout-service**, which depends on **redis-cache** for in-progress cart and session
  state.

## platform-team

- Owns **auth-service**, which depends on **redis-cache** for session storage.
- Owns the shared **redis-cache** and **postgres-primary** infrastructure used by other teams'
  services.

## messaging-team

- Owns **notification-service**, which depends on **rabbitmq-broker** to queue outbound emails
  and push notifications.
- Owns the shared **rabbitmq-broker** infrastructure.

## Escalation

If a shared dependency like redis-cache, postgres-primary, or rabbitmq-broker is degraded, page
platform-team (for redis-cache or postgres-primary) or messaging-team (for rabbitmq-broker) in
addition to the owning team of any affected downstream service.
