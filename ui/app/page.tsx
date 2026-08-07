"use client";

import { FormEvent, useState } from "react";

type Citation = {
  title: string | null;
  url: string | null;
  retrieval_path: string;
  graph_path: string | null;
};

type Message = {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
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
        { role: "assistant", content: data.answer, citations: data.citations },
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
            {message.citations && message.citations.length > 0 && (
              <ul className="sources">
                {message.citations.map((citation, citationIndex) => (
                  <li key={citationIndex}>
                    [{citation.retrieval_path}]{" "}
                    {citation.url ? (
                      <a href={citation.url} target="_blank" rel="noreferrer">
                        {citation.title ?? citation.url}
                      </a>
                    ) : (
                      (citation.title ?? "Untitled source")
                    )}
                    {citation.graph_path && (
                      <div className="graph-path">{citation.graph_path}</div>
                    )}
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
