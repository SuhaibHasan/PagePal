"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import type { ChatMessage } from "@/app/lib/types";
import { MessageBubble } from "./MessageBubble";

export function ChatPanel({
  messages,
  isStreaming,
  onSend,
  onExploreEntity,
}: {
  messages: ChatMessage[];
  isStreaming: boolean;
  onSend: (text: string) => void;
  onExploreEntity: (entityName: string) => void;
}) {
  const [input, setInput] = useState("");
  const scrollAnchorRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    scrollAnchorRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    const text = input.trim();
    if (!text || isStreaming) return;
    onSend(text);
    setInput("");
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex-1 space-y-4 overflow-y-auto p-4">
        {messages.length === 0 && (
          <p className="text-sm text-slate-500">
            Describe the production issue you&apos;re investigating…
          </p>
        )}
        {messages.map((message) => (
          <MessageBubble key={message.id} message={message} onExploreEntity={onExploreEntity} />
        ))}
        <div ref={scrollAnchorRef} />
      </div>

      <form onSubmit={handleSubmit} className="flex gap-2 border-t border-slate-800 p-4">
        <input
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder="Describe the production issue…"
          className="flex-1 rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-500 focus:border-blue-500 focus:outline-none"
        />
        <button
          type="submit"
          disabled={isStreaming || !input.trim()}
          className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
        >
          Send
        </button>
      </form>
    </div>
  );
}
