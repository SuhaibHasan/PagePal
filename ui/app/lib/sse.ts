export interface SSEEvent {
  event: string;
  data: string;
}

/**
 * Parses a `text/event-stream` response body into (event, data) pairs.
 * Native EventSource only supports GET, and /chat is a POST, so this reads
 * the fetch Response's ReadableStream directly instead.
 */
export async function* parseSSEStream(response: Response): AsyncGenerator<SSEEvent> {
  if (!response.body) {
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let separatorIndex: number;
      while ((separatorIndex = buffer.indexOf("\n\n")) !== -1) {
        const rawEvent = buffer.slice(0, separatorIndex);
        buffer = buffer.slice(separatorIndex + 2);
        const event = parseEventBlock(rawEvent);
        if (event) yield event;
      }
    }
  } finally {
    reader.releaseLock();
  }
}

function parseEventBlock(block: string): SSEEvent | null {
  let event = "message";
  const dataLines: string[] = [];

  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) {
      event = line.slice("event:".length).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).trim());
    }
  }

  if (dataLines.length === 0) return null;
  return { event, data: dataLines.join("\n") };
}
