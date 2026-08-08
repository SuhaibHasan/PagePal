# INCIDENT-5502: notification-service backlog

**Severity:** P2
**Service:** notification-service
**Owning team:** messaging-team

## Summary

notification-service fell over three hours behind on outbound emails and push notifications.
notification-service depends on rabbitmq-broker to queue outbound messages, and rabbitmq-broker's
disk filled up, causing it to reject new publishes with ERR-429 until space was freed.

## Timeline

- rabbitmq-broker disk usage hit 100%, and publishers started receiving ERR-429 from
  notification-service as it backed off.
- The queue backlog grew for roughly three hours before the disk-full alert was acknowledged.
- The on-call engineer ran the restart-rabbitmq-broker runbook after clearing old log segments
  to free disk space, which let the broker accept publishes again and the backlog drained.

## Resolution

Resolved by the restart-rabbitmq-broker runbook. messaging-team added log rotation to prevent
rabbitmq-broker's disk from filling up again.
