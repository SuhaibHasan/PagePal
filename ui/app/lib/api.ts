import { parseSSEStream } from "./sse";
import type { Citation, Filters, Subgraph } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8080";

export interface ChatStreamHandlers {
  onSession?: (sessionId: string) => void;
  onToken: (text: string) => void;
  onCitations: (citations: Citation[]) => void;
  onDone?: () => void;
  onError?: (message: string) => void;
}

function buildChatFilters(filters: Filters | undefined) {
  if (!filters) return undefined;
  const payload = {
    service: filters.service || undefined,
    severity: filters.severity || undefined,
    date_from: filters.dateFrom || undefined,
    date_to: filters.dateTo || undefined,
  };
  const hasAny = Object.values(payload).some((value) => value !== undefined);
  return hasAny ? payload : undefined;
}

export async function streamChat(
  message: string,
  sessionId: string | null,
  filters: Filters | undefined,
  handlers: ChatStreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      session_id: sessionId ?? undefined,
      filters: buildChatFilters(filters),
    }),
    signal,
  });

  if (!response.ok) {
    throw new Error(`Chat request failed with status ${response.status}`);
  }

  for await (const { event, data } of parseSSEStream(response)) {
    const payload = data ? JSON.parse(data) : {};
    switch (event) {
      case "session":
        handlers.onSession?.(payload.session_id);
        break;
      case "token":
        handlers.onToken(payload.text ?? "");
        break;
      case "citations":
        handlers.onCitations(payload.citations ?? []);
        break;
      case "done":
        handlers.onDone?.();
        break;
      case "error":
        handlers.onError?.(payload.error ?? "Unknown error");
        break;
    }
  }
}

export async function fetchSubgraph(entityName: string): Promise<Subgraph> {
  const response = await fetch(
    `${API_URL}/graph/explore?entity_name=${encodeURIComponent(entityName)}`,
  );
  if (!response.ok) {
    throw new Error(`Graph explore request failed with status ${response.status}`);
  }
  return response.json();
}
