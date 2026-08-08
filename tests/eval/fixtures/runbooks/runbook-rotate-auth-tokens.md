# Runbook: rotate-auth-tokens

**Service:** auth-service
**Owning team:** platform-team

## When to use this

Use this runbook when users report being unexpectedly logged out or unable to authenticate, and
auth-service logs show ERR-401 for tokens that should still be valid. This usually means the
signing key used to issue auth-service tokens was rotated without a sufficient overlap window.

## Steps

1. Confirm the error is ERR-401 and correlate the timestamps against the last signing key
   rotation for auth-service.
2. If a rotation happened without overlap, temporarily accept both the old and new signing keys
   on auth-service until active sessions age out.
3. Re-issue tokens for any sessions that were force-logged-out during the gap.
4. Update the key rotation process to always overlap old and new keys for at least one full
   token lifetime.
