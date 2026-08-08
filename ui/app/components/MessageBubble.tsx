import type { ChatMessage } from "@/app/lib/types";
import { CitationCard } from "./CitationCard";

export function MessageBubble({
  message,
  onExploreEntity,
}: {
  message: ChatMessage;
  onExploreEntity: (entityName: string) => void;
}) {
  const isUser = message.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[85%] rounded-2xl px-4 py-3 ${
          isUser ? "bg-blue-600 text-white" : "bg-slate-800 text-slate-100"
        }`}
      >
        <p className="whitespace-pre-wrap text-sm">
          {message.content}
          {message.isStreaming && (
            <span className="ml-1 inline-block h-3 w-1.5 animate-pulse bg-current align-middle" />
          )}
        </p>

        {message.error && <p className="mt-1 text-xs text-red-400">{message.error}</p>}

        {message.citations && message.citations.length > 0 && (
          <div className="mt-3 grid gap-2">
            {message.citations.map((citation, index) => (
              <CitationCard key={index} citation={citation} onExploreEntity={onExploreEntity} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
