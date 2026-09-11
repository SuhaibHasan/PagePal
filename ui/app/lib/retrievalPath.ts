export type RetrievalKind = "vector" | "keyword" | "graph" | "wiki" | "hybrid";

/**
 * The API tags each citation's retrieval_path as one of "vector"/"keyword"/"graph"
 * (or a "+"-joined combination, e.g. "graph+keyword+vector", when the reranker found
 * the same chunk via more than one retriever), or "wiki" when the answer came from
 * the wiki fast path instead of a full RAG pass.
 */
export function classifyRetrievalPath(path: string): RetrievalKind {
  if (path === "wiki") return "wiki";
  const parts = path.split("+").filter(Boolean);
  if (parts.length > 1) return "hybrid";
  if (parts[0] === "vector" || parts[0] === "keyword" || parts[0] === "graph") {
    return parts[0];
  }
  return "hybrid";
}

export const RETRIEVAL_LABELS: Record<RetrievalKind, string> = {
  vector: "Vector",
  keyword: "Keyword",
  graph: "Graph",
  wiki: "Wiki",
  hybrid: "Hybrid",
};

export const RETRIEVAL_BADGE_CLASSES: Record<RetrievalKind, string> = {
  vector: "border-blue-500/40 bg-blue-500/15 text-blue-300",
  keyword: "border-emerald-500/40 bg-emerald-500/15 text-emerald-300",
  graph: "border-purple-500/40 bg-purple-500/15 text-purple-300",
  wiki: "border-pink-500/40 bg-pink-500/15 text-pink-300",
  hybrid: "border-amber-500/40 bg-amber-500/15 text-amber-300",
};
