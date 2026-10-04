#!/usr/bin/env node
import "source-map-support/register";
import * as cdk from "aws-cdk-lib";
import { PagePalStack } from "../lib/pagepal-stack";

const app = new cdk.App();

new PagePalStack(app, "PagePalStack", {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION,
  },
  description: "PagePal: RAG assistant for production incidents (API, UI, Chroma, Elasticsearch, Neo4j, Redis).",
});
