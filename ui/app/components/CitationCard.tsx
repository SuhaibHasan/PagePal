import { extractEntityNames } from "@/app/lib/graphPath";
import type { Citation } from "@/app/lib/types";
import { RetrievalBadge } from "./RetrievalBadge";

export function CitationCard({
  citation,
  onExploreEntity,
}: {
  citation: Citation;
  onExploreEntity: (entityName: string) => void;
}) {
  return (
    <div className="rounded-lg border border-slate-700 bg-slate-800/60 p-3 text-sm">
      <div className="flex items-start justify-between gap-2">
        <span className="truncate font-medium text-slate-100">
          {citation.title ?? "Untitled source"}
        </span>
        <RetrievalBadge path={citation.retrieval_path} />
      </div>

      {citation.url && (
        <a
          href={citation.url}
          target="_blank"
          rel="noreferrer"
          className="mt-1 block truncate text-xs text-blue-400 hover:underline"
        >
          {citation.url}
        </a>
      )}

      {citation.graph_path && (
        <div className="mt-2 space-y-1.5">
          <p className="break-words font-mono text-[11px] leading-relaxed text-slate-400">
            {citation.graph_path}
          </p>
          <div className="flex flex-wrap gap-1">
            {extractEntityNames(citation.graph_path).map((entityName) => (
              <button
                key={entityName}
                type="button"
                onClick={() => onExploreEntity(entityName)}
                className="rounded border border-slate-600 px-1.5 py-0.5 text-[11px] text-slate-300 transition hover:border-purple-400 hover:text-purple-300"
              >
                {entityName}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
