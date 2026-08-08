// Relationship types are always ALL_CAPS_WITH_UNDERSCORES (CAUSED_BY, DEPENDS_ON, ...);
// entity names never match that pattern exactly (they're lowercase service names,
// hyphenated error codes with digits, incident IDs with digits, etc.), so this is
// a reliable way to split a serialized path back into just its entity names.
const RELATIONSHIP_TYPE_PATTERN = /^[A-Z_]+$/;

/**
 * citation.graph_path is a "; "-joined list of arrow chains, e.g.
 * "auth-service → DEPENDS_ON → redis-cache → CAUSED_BY → INCIDENT-4521".
 * Extracts the unique entity names mentioned across all chains.
 */
export function extractEntityNames(graphPath: string): string[] {
  const names = new Set<string>();
  for (const chain of graphPath.split(";")) {
    for (const part of chain.split("→")) {
      const trimmed = part.trim();
      if (trimmed && !RELATIONSHIP_TYPE_PATTERN.test(trimmed)) {
        names.add(trimmed);
      }
    }
  }
  return Array.from(names);
}
