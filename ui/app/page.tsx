"use client";

import { FormEvent, useState } from "react";

type Source = {
  document_id: string;
  chunk_id: string;
  content_snippet: string;
  score: number;
  source_type: string;
};

type Message = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
};

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8080";

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    const message = input.trim();
    if (!message || isLoading) return;

    setMessages((prev) => [...prev, { role: "user", content: message }]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });
      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`);
      }
      const data = await response.json();
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.answer, sources: data.sources },
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${(error as Error).message}` },
      ]);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="chat">
      <h1>ProdSupportBuddy</h1>
      <div className="messages">
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.role}`}>
            <p>{message.content}</p>
            {message.sources && message.sources.length > 0 && (
              <ul className="sources">
                {message.sources.map((source) => (
                  <li key={source.chunk_id}>
                    [{source.source_type}] {source.content_snippet}
                  </li>
                ))}
              </ul>
            )}
          </div>
        ))}
        {isLoading && <div className="message assistant">Thinking…</div>}
      </div>
      <form onSubmit={handleSubmit} className="composer">
        <input
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder="Describe the production issue…"
        />
        <button type="submit" disabled={isLoading}>
          Send
        </button>
      </form>
    </main>
  );
}
