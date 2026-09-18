"use client";

import { useMemo, useRef, useState } from "react";
import { ChatPanel } from "@/app/components/ChatPanel";
import { GraphExplorer } from "@/app/components/GraphExplorer";
import { Sidebar } from "@/app/components/Sidebar";
import { extractEntityNames } from "@/app/lib/graphPath";
import { streamChat } from "@/app/lib/api";
import type { ChatMessage, Filters } from "@/app/lib/types";

export default function HomePage() {
  const [filters, setFilters] = useState<Filters>({});
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [exploredEntity, setExploredEntity] = useState<string | null>(null);
  const sessionIdRef = useRef<string | null>(null);

  const suggestedEntities = useMemo(() => {
    const names = new Set<string>();
    for (const message of messages) {
      for (const citation of message.citations ?? []) {
        if (citation.graph_path) {
          for (const name of extractEntityNames(citation.graph_path)) {
            names.add(name);
          }
        }
      }
    }
    return Array.from(names);
  }, [messages]);

  function updateAssistantMessage(id: string, update: Partial<ChatMessage>) {
    setMessages((prev) =>
      prev.map((message) => (message.id === id ? { ...message, ...update } : message)),
    );
  }

  async function handleSend(text: string) {
    const assistantId = crypto.randomUUID();
    setMessages((prev) => [
      ...prev,
      { id: crypto.randomUUID(), role: "user", content: text },
      { id: assistantId, role: "assistant", content: "", isStreaming: true },
    ]);
    setIsStreaming(true);

    try {
      await streamChat(text, sessionIdRef.current, filters, {
        onSession: (sessionId) => {
          sessionIdRef.current = sessionId;
        },
        onToken: (chunk) => {
          setMessages((prev) =>
            prev.map((message) =>
              message.id === assistantId
                ? { ...message, content: message.content + chunk }
                : message,
            ),
          );
        },
        onCitations: (citations) => updateAssistantMessage(assistantId, { citations }),
        onDone: () => updateAssistantMessage(assistantId, { isStreaming: false }),
        onError: (error) => updateAssistantMessage(assistantId, { isStreaming: false, error }),
      });
    } catch (error) {
      updateAssistantMessage(assistantId, { isStreaming: false, error: (error as Error).message });
    } finally {
      setIsStreaming(false);
    }
  }

  return (
    <main className="flex h-screen overflow-hidden bg-slate-950 text-slate-100">
      <Sidebar filters={filters} onChange={setFilters} />

      <section className="flex min-w-0 flex-1 flex-col border-r border-slate-800">
        <header className="border-b border-slate-800 px-4 py-3">
          <h1 className="text-lg font-semibold">PagePal</h1>
          <p className="text-xs text-slate-500">Hybrid RAG assistant for production incidents</p>
        </header>
        <div className="min-h-0 flex-1">
          <ChatPanel
            messages={messages}
            isStreaming={isStreaming}
            onSend={handleSend}
            onExploreEntity={setExploredEntity}
          />
        </div>
      </section>

      <section className="w-[420px] shrink-0 p-3">
        <GraphExplorer
          entityName={exploredEntity}
          onSearch={setExploredEntity}
          suggestedEntities={suggestedEntities}
        />
      </section>
    </main>
  );
}
