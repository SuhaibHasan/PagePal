# Team deployment rollout

Today, everyone who runs `docker compose up` gets their own empty wiki. The self-maintaining
wiki layer (see the [README](../README.md#the-wiki-layer)) only compounds if the whole team is
reading and writing the *same* one — so adopting this as a team tool means running one shared,
persistent instance instead of N laptops each running their own isolated copy.

This is a phased rollout, not a single migration. Each phase ships independently and stays
fully working on its own; later phases only add a layer in front of what already shipped, they
never require reworking it.

| Phase | Adds | Depends on |
|---|---|---|
| 1 | Centralized, reachable, durable | — |
| 2 | TLS + reverse proxy | Phase 1's DNS record |
| 3 | Per-user identity (SSO) | Phase 2's HTTPS domain |
| 4 | Real observability | — |
| 5 | Managed backing services | pain, not a date |

## Phase 1 — Centralized, reachable, durable

Ship this first. One cloud VM, the existing `docker-compose.yml` almost unchanged, the team
reaches it directly by IP or a plain DNS name — no new infrastructure concepts beyond what the
repo already has.

- **One VM.** A single mid-sized instance (AWS EC2, GCP Compute Engine, a DigitalOcean droplet —
  8GB+ RAM) runs Chroma + Elasticsearch + Neo4j + Redis + the API + the UI. All of it fits on one
  box at team scale. Attach a persistent disk so a reboot doesn't wipe the data.
- **Add the API and UI as two more compose services**, running production commands instead of dev
  ones:
  - `uvicorn api.main:app --host 0.0.0.0 --port 8080 --workers 4` (no `--reload`)
  - a UI container running `next build && next start` instead of `next dev`
- **Two config values that must change** (easy to forget, breaks silently if missed):
  - `api_cors_origins` in `api/config.py` → the UI's real URL, not `http://localhost:3000`
  - `NEXT_PUBLIC_API_URL` for the UI build → the VM's real API address, not `http://localhost:8080`
- **Open just two ports** (3000, 8080) in the VM's firewall/security group, restricted to the
  team's office or VPN IP range — not the open internet. Plain HTTP for now; the API key travels
  in cleartext, which is a real tradeoff, but a reasonable one for an internal tool behind a
  firewall you control. TLS is additive later (Phase 2), not a rewrite.
- **DNS.** Point an A record at the VM's static IP so people get a name, not an IP.
- **Auth stays as-is.** Set `API_KEY` in the VM's `.env` (the mechanism already exists in
  `api/auth.py`). Combined with the firewall, that's a reasonable bar without SSO for now.
- **Secrets.** `chmod 600 .env` on the VM, not committed. A real secrets manager can wait for
  Phase 3+.
- **Backups.** Nightly snapshot of the ES / Neo4j / Chroma / Redis volumes to S3/GCS. This is
  cheap now and expensive after the first data loss — the wiki is the accumulated point of the
  whole tool.

**Ships when:** the team can hit one URL and it survives a VM reboot without losing data.

## Phase 2 — TLS + reverse proxy

- Caddy (or nginx) in front, terminating a Let's Encrypt certificate on the real domain from
  Phase 1's DNS record.
- Traffic moves from plain HTTP behind a firewall to actual HTTPS.
- Zero app code changes — purely an infrastructure layer in front of what Phase 1 already runs.

**Ships when:** the team's browser shows a valid certificate and the plain-HTTP port is closed.

## Phase 3 — Per-user identity

- `oauth2-proxy` (or the team's IdP equivalent) sits behind Phase 2's reverse proxy, in front of
  the UI. Needs Phase 2's stable HTTPS origin for the SSO callback URL — this is the one load-bearing
  ordering dependency in the whole plan.
- First real code change: extend `api/auth.py` to read the identity header the proxy forwards,
  replacing the single shared `API_KEY`.
- `/ingest` and wiki writes (`POST /wiki/{id}/validate`, `DELETE /wiki/{id}`) become attributable
  to a person instead of "whoever has the key."

**Ships when:** two different teammates' actions show up as two different identities, not one
shared secret.

## Phase 4 — Real observability

- Swap `ConsoleSpanExporter` in `api/telemetry.py` for an OTLP exporter pointed at a collector
  (self-hosted Jaeger/Tempo, or a SaaS).
- Alert on `GET /health` reporting `"status": "degraded"`.

**Ships when:** a dependency going down pages someone instead of silently degrading.

## Phase 5 — Managed backing services (pain-triggered, not scheduled)

Not on a calendar — do each swap only once self-hosting that specific piece becomes the actual
bottleneck (ops burden, backup pain, scaling). Each is a config-value swap, not a rewrite:

| Swap | Config value | Worth it when |
|---|---|---|
| Elasticsearch → Elastic Cloud | `elasticsearch_url` | cluster/version ops becomes the time sink |
| Neo4j → Neo4j Aura | `neo4j_uri` | same, plus you want managed backups for the graph specifically |
| Redis → ElastiCache / Upstash | `redis_url` | cheapest insurance on this list — sessions and rate limiting already degrade gracefully if Redis hiccups |
| Chroma → stays self-hosted | — | revisit only once the corpus or query volume outgrows a single node |

## What Phase 1 alone gets you (and doesn't)

**Gets:** one shared, persistent knowledge base the whole team hits over one URL — the actual
point of adopting this as a team tool instead of a personal one.

**Doesn't yet have:** TLS, per-user audit trail (still one shared API key), or distributed
tracing. All three are additive layers on top of Phase 1, not things that require re-architecting
it later.
