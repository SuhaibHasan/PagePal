export interface Citation {
  title: string | null;
  url: string | null;
  retrieval_path: string;
  graph_path: string | null;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  isStreaming?: boolean;
  error?: string;
}

export interface Filters {
  service?: string;
  severity?: string;
  dateFrom?: string;
  dateTo?: string;
}

export interface GraphNode {
  id: string;
  name: string;
  labels: string[];
}

export interface GraphEdge {
  source: string;
  target: string;
  type: string;
}

export interface Subgraph {
  nodes: GraphNode[];
  edges: GraphEdge[];
}
