# PagePal on AWS (CDK)

A CDK (TypeScript) app that deploys PagePal onto AWS using managed services
where a real managed equivalent exists, and self-hosted Fargate services
where it doesn't.

## Architecture

```
                           Internet
                              |
                     Application Load Balancer (:80)
                     /                          \
          path: /health,/chat,          default route
          /ingest,/graph*,/wiki*                |
                |                                v
                v                       UI service (Fargate)
        API service (Fargate)           Next.js, standalone build
        FastAPI, 2 tasks                2 tasks
                |
   -------------+-----------------------------------
   |            |              |                    |
   v            v              v                    v
Chroma      Elasticsearch   Neo4j              ElastiCache Redis
(Fargate)   (Fargate)       (Fargate)          (managed)
   |            |              |
   +---- EFS access points ----+
     (chroma / es / neo4j-data / neo4j-logs)
```

- **ElastiCache (Redis)** is the one dependency with a true, protocol-compatible
  managed AWS service, so it uses that.
- **Chroma, Elasticsearch, and Neo4j** have no managed AWS equivalent that's
  actually compatible with this codebase's clients (AWS OpenSearch Service
  diverges from the Elasticsearch 8.x API this project's client expects; there
  is no managed Neo4j on AWS). They run as Fargate tasks instead, each with
  its own EFS access point so data survives a task restart or redeploy.
- All data-store services are **internal only** (no public IP, not attached to
  the ALB), reachable from the API service via Cloud Map private DNS
  (`chromadb.pagepal.local`, `elasticsearch.pagepal.local`, `neo4j.pagepal.local`).
- The **UI and API share one ALB**, split by path (`/health`, `/chat`,
  `/ingest`, `/graph*`, `/wiki*` → API; everything else → UI). The UI's
  `NEXT_PUBLIC_API_URL` is baked in as `""` at image build time, so the
  browser's own `fetch()` calls are same-origin and the ALB's path rule routes
  them to the right service - no CORS, and no chicken-and-egg problem of
  needing to know the ALB's DNS name before building the UI image.

## Before you deploy

- Docker running locally (`cdk deploy` builds the API and UI images on your
  machine and pushes them to an ECR repo CDK creates for you).
- AWS CLI configured with credentials for the target account.
- Node.js 20+, and this account bootstrapped for CDK: `npx cdk bootstrap`.

```bash
cd infra/cdk
npm install
npx cdk bootstrap   # once per account/region
npx cdk deploy
```

`cdk deploy` prints an `AppUrl` output when it finishes - that's the whole app.

## Setting the Anthropic API key

The stack creates an empty Secrets Manager secret for the key (CDK can't bake
a real secret value into source-controlled infrastructure code). After the
first deploy:

```bash
aws secretsmanager put-secret-value \
  --secret-id $(aws cloudformation describe-stacks --stack-name PagePalStack \
      --query "Stacks[0].Outputs[?OutputKey=='AnthropicApiKeySecretArn'].OutputValue" --output text) \
  --secret-string '<your-anthropic-api-key>'

aws ecs update-service --cluster <cluster-name> --service <api-service-name> --force-new-deployment
```

(Cluster and service names are in the ECS console, or `aws ecs list-services --cluster <cluster-name>`.)

## Cost (rough, us-east-1, light/demo traffic)

| Component | Estimate |
|---|---|
| NAT Gateway (1) | ~$32/mo + data |
| ALB | ~$18/mo + LCUs |
| Fargate: API (2x 0.5vCPU/1GB) | ~$30/mo |
| Fargate: UI (2x 0.25vCPU/0.5GB) | ~$15/mo |
| Fargate: Chroma (0.5vCPU/1GB) | ~$15/mo |
| Fargate: Elasticsearch (1vCPU/3GB) | ~$45/mo |
| Fargate: Neo4j (0.5vCPU/2GB) | ~$25/mo |
| ElastiCache cache.t3.micro | ~$12/mo |
| EFS (small volumes) | a few $/mo |
| **Total** | **roughly $190-220/mo** |

The biggest cost lever is `desiredCount` on the API/UI services (drop both to
1 for a personal demo) and the NAT Gateway (removable if you're comfortable
giving the data-store tasks no outbound internet access post-boot, which
needs VPC endpoints for ECR/CloudWatch Logs instead).

## Known simplifications (read before using this beyond a demo)

- **Elasticsearch and Neo4j have no auth/TLS between the API and themselves**,
  matching the existing `docker-compose.yml` for this repo. This is safe-ish
  only because they're in private subnets with security groups that permit
  nothing but the API service - it is not hardened for a real multi-tenant
  production environment.
- **The Neo4j initial password is generated once at CDK synth time** (not via
  Secrets Manager's own generator) because the Neo4j container needs a single
  composed `neo4j/<password>` value for `NEO4J_AUTH`, and ECS secret injection
  can only map a secret straight to an env var, not concatenate a prefix onto
  it. That means the generated password appears as plaintext in the
  synthesized CloudFormation template (`cdk.out/`) - treat that directory and
  the deployed stack's template as sensitive, and rotate the password via
  `cypher-shell` after first deploy if the stack will be shared or kept long-term.
- **EFS access points use root (uid/gid 0) with 0777 permissions** for
  simplicity across three different container images' default users. Fine
  for a locked-down demo; a real production setup should pin this per-image.
- No HTTPS on the ALB (no domain name assumed). Add an ACM certificate and a
  Route 53 record if you want one - the stack takes no changes to support it,
  just attach an HTTPS listener alongside or instead of the HTTP one.

## Tearing down

```bash
npx cdk destroy
```

EFS, the ElastiCache cluster, and both secrets are set to delete with the
stack (`RemovalPolicy.DESTROY` / default secret deletion) - there is no data
retained after this beyond what's in S3/CloudWatch Logs retention (14 days).
