# Runbook: restart-rabbitmq-broker

**Dependency:** rabbitmq-broker
**Owning team:** messaging-team
**Depended on by:** notification-service

## When to use this

Use this runbook when rabbitmq-broker is rejecting publishes with ERR-429 because its disk is
full, which backs up notification-service's outbound queue, as happened in INCIDENT-5502.

## Steps

1. Confirm rabbitmq-broker disk usage is at or near 100%.
2. Clear old log segments and any expired message data to free disk space.
3. Restart rabbitmq-broker so it resumes accepting publishes.
4. Confirm notification-service's queue backlog is draining and disk usage stays below the
   alerting threshold.
