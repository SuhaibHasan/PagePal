FROM node:20-slim AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM node:20-slim AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
# Relative (same-origin) API calls: the ALB routes known API paths
# (/chat, /graph/*, /health, /wiki/*, /ingest) to the API service and
# everything else to this UI service, so the browser never needs an
# absolute, deploy-time-unknown API hostname baked into the bundle.
ENV NEXT_PUBLIC_API_URL=""
RUN npm run build

FROM node:20-slim AS runner
WORKDIR /app
ENV NODE_ENV=production
RUN useradd --create-home --uid 1000 app

COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

USER app
EXPOSE 3000
ENV PORT=3000 HOSTNAME=0.0.0.0

CMD ["node", "server.js"]
