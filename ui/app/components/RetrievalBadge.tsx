import { classifyRetrievalPath, RETRIEVAL_BADGE_CLASSES, RETRIEVAL_LABELS } from "@/app/lib/retrievalPath";

export function RetrievalBadge({ path }: { path: string }) {
  const kind = classifyRetrievalPath(path);
  return (
    <span
      className={`inline-flex shrink-0 items-center rounded-full border px-2 py-0.5 text-[11px] font-medium ${RETRIEVAL_BADGE_CLASSES[kind]}`}
      title={path}
    >
      {RETRIEVAL_LABELS[kind]}
    </span>
  );
}
