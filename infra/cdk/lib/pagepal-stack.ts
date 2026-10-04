import * as crypto from "crypto";
import * as path from "path";
import * as cdk from "aws-cdk-lib";
import { Construct } from "constructs";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as ecs from "aws-cdk-lib/aws-ecs";
import * as efs from "aws-cdk-lib/aws-efs";
import * as elasticache from "aws-cdk-lib/aws-elasticache";
import * as elbv2 from "aws-cdk-lib/aws-elasticloadbalancingv2";
import * as logs from "aws-cdk-lib/aws-logs";
import * as secretsmanager from "aws-cdk-lib/aws-secretsmanager";

// Repo root: infra/cdk/lib -> infra/cdk -> infra -> repo root.
const REPO_ROOT = path.join(__dirname, "..", "..", "..");
const CLOUD_MAP_NAMESPACE = "pagepal.local";

export class PagePalStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // ---------------------------------------------------------------------
    // Networking
    // ---------------------------------------------------------------------
    const vpc = new ec2.Vpc(this, "Vpc", {
      maxAzs: 2,
      natGateways: 1,
      subnetConfiguration: [
        { name: "public", subnetType: ec2.SubnetType.PUBLIC, cidrMask: 24 },
        { name: "private", subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS, cidrMask: 24 },
      ],
    });

    const cluster = new ecs.Cluster(this, "Cluster", {
      vpc,
      containerInsightsV2: ecs.ContainerInsights.ENABLED,
    });
    cluster.addDefaultCloudMapNamespace({ name: CLOUD_MAP_NAMESPACE });

    const logGroup = new logs.LogGroup(this, "LogGroup", {
      logGroupName: "/pagepal",
      retention: logs.RetentionDays.TWO_WEEKS,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    // ---------------------------------------------------------------------
    // Security groups
    // ---------------------------------------------------------------------
    const albSg = new ec2.SecurityGroup(this, "AlbSg", { vpc, description: "Public ALB" });
    albSg.addIngressRule(ec2.Peer.anyIpv4(), ec2.Port.tcp(80), "Public HTTP");

    const uiSg = new ec2.SecurityGroup(this, "UiServiceSg", { vpc, description: "Next.js UI tasks" });
    uiSg.addIngressRule(albSg, ec2.Port.tcp(3000), "ALB to UI");

    const apiSg = new ec2.SecurityGroup(this, "ApiServiceSg", { vpc, description: "FastAPI tasks" });
    apiSg.addIngressRule(albSg, ec2.Port.tcp(8080), "ALB to API");

    const dataSg = new ec2.SecurityGroup(this, "DataServiceSg", {
      vpc,
      description: "Chroma, Elasticsearch, Neo4j tasks",
    });
    dataSg.addIngressRule(apiSg, ec2.Port.tcp(8000), "API to Chroma");
    dataSg.addIngressRule(apiSg, ec2.Port.tcp(9200), "API to Elasticsearch");
    dataSg.addIngressRule(apiSg, ec2.Port.tcp(7687), "API to Neo4j bolt");

    const redisSg = new ec2.SecurityGroup(this, "RedisSg", { vpc, description: "ElastiCache Redis" });
    redisSg.addIngressRule(apiSg, ec2.Port.tcp(6379), "API to Redis");

    const efsSg = new ec2.SecurityGroup(this, "EfsSg", { vpc, description: "EFS mount targets" });
    efsSg.addIngressRule(dataSg, ec2.Port.tcp(2049), "Data-store tasks to EFS");

    // ---------------------------------------------------------------------
    // Secrets
    // ---------------------------------------------------------------------
    // Generated once at synth time rather than via Secrets Manager's own
    // generator, because the Neo4j container needs NEO4J_AUTH as a single
    // composed "neo4j/<password>" value and ECS secret injection can only
    // map a secret (or one JSON field of it) straight to an env var, with
    // no way to concatenate a prefix onto it at deploy time.
    const neo4jRawPassword = crypto.randomBytes(18).toString("base64url");

    const neo4jPassword = new secretsmanager.Secret(this, "Neo4jPassword", {
      description: "Password for the self-hosted Neo4j service (consumed as NEO4J_PASSWORD)",
      secretStringValue: cdk.SecretValue.unsafePlainText(neo4jRawPassword),
    });

    const neo4jAuth = new secretsmanager.Secret(this, "Neo4jAuth", {
      description: "Composed 'neo4j/<password>' value consumed by the Neo4j container's NEO4J_AUTH",
      secretStringValue: cdk.SecretValue.unsafePlainText(`neo4j/${neo4jRawPassword}`),
    });

    // Placeholder on first deploy - overwrite with a real key after `cdk deploy`:
    //   aws secretsmanager put-secret-value --secret-id <arn> --secret-string '<key>'
    // then force a fresh API deployment to pick it up (see infra/cdk/README.md).
    const anthropicApiKey = new secretsmanager.Secret(this, "AnthropicApiKey", {
      description: "Anthropic API key consumed by the PagePal API service",
    });

    // ---------------------------------------------------------------------
    // Shared storage for the self-hosted data stores. Chroma, Elasticsearch
    // and Neo4j have no drop-in compatible managed AWS service, so they run
    // as Fargate tasks instead, each backed by its own EFS access point for
    // state that must survive a task restart. Redis is the one dependency
    // with a true managed equivalent (ElastiCache), so it uses that instead.
    // ---------------------------------------------------------------------
    const fileSystem = new efs.FileSystem(this, "DataFileSystem", {
      vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      securityGroup: efsSg,
      encrypted: true,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    const makeAccessPoint = (id: string, dirPath: string) =>
      fileSystem.addAccessPoint(id, {
        path: dirPath,
        createAcl: { ownerUid: "0", ownerGid: "0", permissions: "0777" },
        posixUser: { uid: "0", gid: "0" },
      });

    const chromaAp = makeAccessPoint("ChromaAp", "/chroma");
    const esAp = makeAccessPoint("ElasticsearchAp", "/elasticsearch");
    const neo4jDataAp = makeAccessPoint("Neo4jDataAp", "/neo4j-data");
    const neo4jLogsAp = makeAccessPoint("Neo4jLogsAp", "/neo4j-logs");

    // ---------------------------------------------------------------------
    // ElastiCache Redis (single node - raise numCacheClusters/add
    // automaticFailoverEnabled for production multi-AZ failover).
    // ---------------------------------------------------------------------
    const redisSubnetGroup = new elasticache.CfnSubnetGroup(this, "RedisSubnetGroup", {
      description: "Private subnets for PagePal Redis",
      subnetIds: vpc.privateSubnets.map((subnet) => subnet.subnetId),
    });

    const redisCluster = new elasticache.CfnCacheCluster(this, "RedisCluster", {
      engine: "redis",
      cacheNodeType: "cache.t3.micro",
      numCacheNodes: 1,
      vpcSecurityGroupIds: [redisSg.securityGroupId],
      cacheSubnetGroupName: redisSubnetGroup.ref,
    });
    const redisUrl = `redis://${redisCluster.attrRedisEndpointAddress}:${redisCluster.attrRedisEndpointPort}/0`;

    // ---------------------------------------------------------------------
    // Helper for the three internal, Cloud-Map-discoverable data services.
    // ---------------------------------------------------------------------
    const addDataService = (opts: {
      id: string;
      cloudMapName: string;
      image: string;
      containerPort: number;
      cpu: number;
      memoryLimitMiB: number;
      environment?: Record<string, string>;
      secrets?: Record<string, ecs.Secret>;
      mounts: { accessPoint: efs.AccessPoint; containerPath: string; volumeName: string }[];
      healthCheckCommand: string[];
    }) => {
      const taskDefinition = new ecs.FargateTaskDefinition(this, `${opts.id}Task`, {
        cpu: opts.cpu,
        memoryLimitMiB: opts.memoryLimitMiB,
      });

      for (const mount of opts.mounts) {
        taskDefinition.addVolume({
          name: mount.volumeName,
          efsVolumeConfiguration: {
            fileSystemId: fileSystem.fileSystemId,
            transitEncryption: "ENABLED",
            authorizationConfig: { accessPointId: mount.accessPoint.accessPointId, iam: "ENABLED" },
          },
        });
      }
      if (opts.mounts.length > 0) {
        // IAM-authorized EFS access points (used above) require the task
        // role itself to be allowed to mount/write - CDK does not infer
        // this from addVolume/addMountPoints alone.
        fileSystem.grant(
          taskDefinition.taskRole,
          "elasticfilesystem:ClientMount",
          "elasticfilesystem:ClientWrite",
          "elasticfilesystem:ClientRootAccess",
        );
      }

      const container = taskDefinition.addContainer(`${opts.id}Container`, {
        image: ecs.ContainerImage.fromRegistry(opts.image),
        environment: opts.environment,
        secrets: opts.secrets,
        logging: ecs.LogDrivers.awsLogs({ streamPrefix: opts.id, logGroup }),
        healthCheck: {
          command: opts.healthCheckCommand,
          interval: cdk.Duration.seconds(15),
          timeout: cdk.Duration.seconds(10),
          retries: 10,
          startPeriod: cdk.Duration.seconds(60),
        },
      });
      container.addPortMappings({ containerPort: opts.containerPort });
      for (const mount of opts.mounts) {
        container.addMountPoints({
          sourceVolume: mount.volumeName,
          containerPath: mount.containerPath,
          readOnly: false,
        });
      }

      return new ecs.FargateService(this, `${opts.id}Service`, {
        cluster,
        taskDefinition,
        desiredCount: 1,
        securityGroups: [dataSg],
        vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
        cloudMapOptions: { name: opts.cloudMapName, dnsTtl: cdk.Duration.seconds(10) },
        // Singleton, EFS-backed services: stop the old task before starting
        // the new one rather than briefly running both, since two instances
        // against the same data directory at once isn't safe for any of
        // Chroma/Elasticsearch/Neo4j.
        minHealthyPercent: 0,
        maxHealthyPercent: 100,
        circuitBreaker: { rollback: true },
      });
    };

    addDataService({
      id: "Chroma",
      cloudMapName: "chromadb",
      image: "chromadb/chroma:0.5.20",
      containerPort: 8000,
      cpu: 512,
      memoryLimitMiB: 1024,
      environment: { IS_PERSISTENT: "TRUE", ANONYMIZED_TELEMETRY: "FALSE" },
      mounts: [{ accessPoint: chromaAp, containerPath: "/chroma/chroma", volumeName: "chroma-data" }],
      healthCheckCommand: ["CMD-SHELL", "curl -f http://localhost:8000/api/v1/heartbeat || exit 1"],
    });

    addDataService({
      id: "Elasticsearch",
      cloudMapName: "elasticsearch",
      image: "docker.elastic.co/elasticsearch/elasticsearch:8.15.3",
      containerPort: 9200,
      cpu: 1024,
      memoryLimitMiB: 3072,
      environment: {
        "discovery.type": "single-node",
        "xpack.security.enabled": "false",
        "xpack.security.enrollment.enabled": "false",
        ES_JAVA_OPTS: "-Xms1g -Xmx1g",
      },
      mounts: [
        { accessPoint: esAp, containerPath: "/usr/share/elasticsearch/data", volumeName: "es-data" },
      ],
      healthCheckCommand: ["CMD-SHELL", "curl -sf http://localhost:9200/_cluster/health || exit 1"],
    });

    addDataService({
      id: "Neo4j",
      cloudMapName: "neo4j",
      image: "neo4j:5.24-community",
      containerPort: 7687,
      cpu: 512,
      memoryLimitMiB: 2048,
      environment: { NEO4J_PLUGINS: '["apoc"]' },
      secrets: { NEO4J_AUTH: ecs.Secret.fromSecretsManager(neo4jAuth) },
      mounts: [
        { accessPoint: neo4jDataAp, containerPath: "/data", volumeName: "neo4j-data" },
        { accessPoint: neo4jLogsAp, containerPath: "/logs", volumeName: "neo4j-logs" },
      ],
      healthCheckCommand: ["CMD-SHELL", "wget -O /dev/null -q http://localhost:7474 || exit 1"],
    });

    // ---------------------------------------------------------------------
    // Application load balancer
    // ---------------------------------------------------------------------
    const alb = new elbv2.ApplicationLoadBalancer(this, "Alb", {
      vpc,
      internetFacing: true,
      securityGroup: albSg,
    });
    const listener = alb.addListener("HttpListener", { port: 80, open: false });

    // ---------------------------------------------------------------------
    // API service - built from the repo's api/Dockerfile.
    // ---------------------------------------------------------------------
    const apiTaskDefinition = new ecs.FargateTaskDefinition(this, "ApiTask", {
      cpu: 512,
      memoryLimitMiB: 1024,
    });
    const apiContainer = apiTaskDefinition.addContainer("ApiContainer", {
      image: ecs.ContainerImage.fromAsset(REPO_ROOT, { file: "api/Dockerfile" }),
      environment: {
        ANTHROPIC_MODEL: "claude-sonnet-5",
        CHROMA_HOST: `chromadb.${CLOUD_MAP_NAMESPACE}`,
        CHROMA_PORT: "8000",
        ELASTICSEARCH_URL: `http://elasticsearch.${CLOUD_MAP_NAMESPACE}:9200`,
        NEO4J_URI: `bolt://neo4j.${CLOUD_MAP_NAMESPACE}:7687`,
        NEO4J_USER: "neo4j",
        REDIS_URL: redisUrl,
        // Same-origin via the ALB's path-based routing below, so this is
        // mostly a fallback for any direct, non-browser caller.
        API_CORS_ORIGINS: "*",
      },
      secrets: {
        ANTHROPIC_API_KEY: ecs.Secret.fromSecretsManager(anthropicApiKey),
        NEO4J_PASSWORD: ecs.Secret.fromSecretsManager(neo4jPassword),
      },
      logging: ecs.LogDrivers.awsLogs({ streamPrefix: "api", logGroup }),
      healthCheck: {
        command: ["CMD-SHELL", "curl -f http://localhost:8080/health || exit 1"],
        interval: cdk.Duration.seconds(15),
        timeout: cdk.Duration.seconds(10),
        retries: 5,
        startPeriod: cdk.Duration.seconds(60),
      },
    });
    apiContainer.addPortMappings({ containerPort: 8080 });

    const apiService = new ecs.FargateService(this, "ApiService", {
      cluster,
      taskDefinition: apiTaskDefinition,
      desiredCount: 2,
      securityGroups: [apiSg],
      vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      minHealthyPercent: 100,
      maxHealthyPercent: 200,
      circuitBreaker: { rollback: true },
    });
    const apiTargetGroup = listener.addTargets("ApiTargets", {
      port: 8080,
      protocol: elbv2.ApplicationProtocol.HTTP,
      priority: 10,
      conditions: [elbv2.ListenerCondition.pathPatterns(["/health", "/chat", "/ingest", "/graph*", "/wiki*"])],
      targets: [apiService],
      healthCheck: { path: "/health", interval: cdk.Duration.seconds(15) },
      deregistrationDelay: cdk.Duration.seconds(30),
    });

    // ---------------------------------------------------------------------
    // UI service - built from the repo's ui/Dockerfile, Next.js standalone
    // output. The browser's own fetch calls are relative (NEXT_PUBLIC_API_URL
    // is baked in as "" at image build time), so they land on this same ALB
    // and get routed to the API service by the path-pattern rule above.
    // ---------------------------------------------------------------------
    const uiTaskDefinition = new ecs.FargateTaskDefinition(this, "UiTask", {
      cpu: 256,
      memoryLimitMiB: 512,
    });
    const uiContainer = uiTaskDefinition.addContainer("UiContainer", {
      image: ecs.ContainerImage.fromAsset(path.join(REPO_ROOT, "ui")),
      logging: ecs.LogDrivers.awsLogs({ streamPrefix: "ui", logGroup }),
      healthCheck: {
        command: ["CMD-SHELL", "node -e \"require('http').get('http://localhost:3000',r=>process.exit(r.statusCode<500?0:1)).on('error',()=>process.exit(1))\""],
        interval: cdk.Duration.seconds(15),
        timeout: cdk.Duration.seconds(10),
        retries: 5,
        startPeriod: cdk.Duration.seconds(45),
      },
    });
    uiContainer.addPortMappings({ containerPort: 3000 });

    const uiService = new ecs.FargateService(this, "UiService", {
      cluster,
      taskDefinition: uiTaskDefinition,
      desiredCount: 2,
      securityGroups: [uiSg],
      vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      minHealthyPercent: 100,
      maxHealthyPercent: 200,
      circuitBreaker: { rollback: true },
    });
    listener.addTargets("UiTargets", {
      port: 3000,
      protocol: elbv2.ApplicationProtocol.HTTP,
      targets: [uiService],
      healthCheck: { path: "/", interval: cdk.Duration.seconds(15) },
    });

    // ---------------------------------------------------------------------
    // Outputs
    // ---------------------------------------------------------------------
    new cdk.CfnOutput(this, "AppUrl", { value: `http://${alb.loadBalancerDnsName}` });
    new cdk.CfnOutput(this, "AnthropicApiKeySecretArn", { value: anthropicApiKey.secretArn });
    new cdk.CfnOutput(this, "Neo4jPasswordSecretArn", { value: neo4jPassword.secretArn });
  }
}
