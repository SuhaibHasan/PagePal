/** @type {import('next').NextConfig} */
const nextConfig = {
  // Don't let Next.js scaffold AGENTS.md/CLAUDE.md into this directory.
  agentRules: false,
  // Self-contained server bundle, needed for a lean production Docker image.
  output: "standalone",
};

module.exports = nextConfig;
