"use client";

import * as d3 from "d3";
import { FormEvent, useEffect, useRef, useState } from "react";
import { fetchSubgraph } from "@/app/lib/api";
import type { Subgraph } from "@/app/lib/types";

const LABEL_COLORS: Record<string, string> = {
  SERVICE: "#3b82f6", // blue
  INCIDENT: "#ef4444", // red
  RUNBOOK: "#22c55e", // green
  TEAM: "#eab308", // yellow
  ERROR_CODE: "#f97316", // orange
  DEPENDENCY: "#14b8a6", // teal
  CONFIG: "#a855f7", // violet
};
const DEFAULT_NODE_COLOR = "#94a3b8"; // slate, for any unrecognized label

function colorForLabels(labels: string[]): string {
  for (const label of labels) {
    if (LABEL_COLORS[label]) return LABEL_COLORS[label];
  }
  return DEFAULT_NODE_COLOR;
}

interface SimNode extends d3.SimulationNodeDatum {
  id: string;
  name: string;
  labels: string[];
}

interface SimLink extends d3.SimulationLinkDatum<SimNode> {
  type: string;
}

export function GraphExplorer({
  entityName,
  onSearch,
  suggestedEntities,
}: {
  entityName: string | null;
  onSearch: (entityName: string) => void;
  suggestedEntities: string[];
}) {
  const [inputValue, setInputValue] = useState(entityName ?? "");
  const [prevEntityName, setPrevEntityName] = useState(entityName);
  const [subgraph, setSubgraph] = useState<Subgraph | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  // Keep the input in sync with the parent-controlled entity (e.g. a citation chip
  // click) without an effect - adjusting state during render avoids the extra commit.
  if (entityName !== prevEntityName) {
    setPrevEntityName(entityName);
    setInputValue(entityName ?? "");
    if (!entityName) {
      setSubgraph(null);
      setError(null);
    }
  }

  // Fetch whenever the parent-controlled entity changes (e.g. a citation chip click).
  useEffect(() => {
    if (!entityName) return;

    let cancelled = false;

    async function run() {
      setIsLoading(true);
      setError(null);
      try {
        const data = await fetchSubgraph(entityName as string);
        if (!cancelled) setSubgraph(data);
      } catch (err) {
        if (!cancelled) {
          setError((err as Error).message);
          setSubgraph(null);
        }
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    }
    run();

    return () => {
      cancelled = true;
    };
  }, [entityName]);

  // Render/re-render the force-directed graph whenever the data changes.
  useEffect(() => {
    const svgEl = svgRef.current;
    const containerEl = containerRef.current;
    if (!subgraph || !svgEl || !containerEl || subgraph.nodes.length === 0) {
      if (svgEl) d3.select(svgEl).selectAll("*").remove();
      return;
    }

    const width = containerEl.clientWidth || 400;
    const height = containerEl.clientHeight || 360;

    const nodes: SimNode[] = subgraph.nodes.map((node) => ({ ...node }));
    const nodeIds = new Set(nodes.map((node) => node.id));
    const links: SimLink[] = subgraph.edges
      .filter((edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target))
      .map((edge) => ({ source: edge.source, target: edge.target, type: edge.type }));

    const svg = d3.select(svgEl);
    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${width} ${height}`);

    const zoomLayer = svg.append("g");
    svg.call(
      d3
        .zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.3, 3])
        .on("zoom", (event) => zoomLayer.attr("transform", event.transform)),
    );

    const simulation = d3
      .forceSimulation(nodes)
      .force(
        "link",
        d3
          .forceLink<SimNode, SimLink>(links)
          .id((d) => d.id)
          .distance(90),
      )
      .force("charge", d3.forceManyBody().strength(-220))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collide", d3.forceCollide(30));

    const link = zoomLayer
      .append("g")
      .attr("stroke", "#475569")
      .attr("stroke-opacity", 0.7)
      .selectAll("line")
      .data(links)
      .join("line")
      .attr("stroke-width", 1.5);

    const linkLabel = zoomLayer
      .append("g")
      .selectAll("text")
      .data(links)
      .join("text")
      .attr("font-size", 9)
      .attr("fill", "#94a3b8")
      .attr("text-anchor", "middle")
      .text((d) => d.type);

    const drag = d3
      .drag<SVGCircleElement, SimNode>()
      .on("start", (event, d) => {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on("drag", (event, d) => {
        d.fx = event.x;
        d.fy = event.y;
      })
      .on("end", (event, d) => {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      });

    const node = zoomLayer
      .append("g")
      .selectAll<SVGCircleElement, SimNode>("circle")
      .data(nodes)
      .join("circle")
      .attr("r", 13)
      .attr("fill", (d) => colorForLabels(d.labels))
      .attr("stroke", "#0f172a")
      .attr("stroke-width", 1.5)
      .style("cursor", "grab")
      .call(drag);

    node.append("title").text((d) => `${d.name} (${d.labels.join(", ") || "unknown"})`);

    const nodeLabel = zoomLayer
      .append("g")
      .selectAll("text")
      .data(nodes)
      .join("text")
      .attr("font-size", 11)
      .attr("fill", "#e2e8f0")
      .attr("text-anchor", "middle")
      .attr("dy", -18)
      .text((d) => d.name);

    simulation.on("tick", () => {
      link
        .attr("x1", (d) => (d.source as SimNode).x ?? 0)
        .attr("y1", (d) => (d.source as SimNode).y ?? 0)
        .attr("x2", (d) => (d.target as SimNode).x ?? 0)
        .attr("y2", (d) => (d.target as SimNode).y ?? 0);

      linkLabel
        .attr("x", (d) => (((d.source as SimNode).x ?? 0) + ((d.target as SimNode).x ?? 0)) / 2)
        .attr("y", (d) => (((d.source as SimNode).y ?? 0) + ((d.target as SimNode).y ?? 0)) / 2);

      node.attr("cx", (d) => d.x ?? 0).attr("cy", (d) => d.y ?? 0);
      nodeLabel.attr("x", (d) => d.x ?? 0).attr("y", (d) => d.y ?? 0);
    });

    return () => {
      simulation.stop();
    };
  }, [subgraph]);

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (inputValue.trim()) onSearch(inputValue.trim());
  }

  const usedLabels =
    subgraph && subgraph.nodes.length > 0
      ? Array.from(new Set(subgraph.nodes.flatMap((node) => node.labels))).filter(
          (label) => LABEL_COLORS[label],
        )
      : [];

  return (
    <div className="flex h-full flex-col rounded-lg border border-slate-700 bg-slate-900/60 p-3">
      <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
        Graph Explorer
      </h2>

      <form onSubmit={handleSubmit} className="mb-2 flex gap-2">
        <input
          value={inputValue}
          onChange={(event) => setInputValue(event.target.value)}
          placeholder="Entity name, e.g. payment-service"
          className="flex-1 rounded border border-slate-700 bg-slate-800 px-2 py-1.5 text-sm text-slate-100 placeholder:text-slate-500 focus:border-purple-500 focus:outline-none"
        />
        <button
          type="submit"
          className="rounded bg-purple-600 px-3 py-1.5 text-sm font-medium text-white transition hover:bg-purple-500"
        >
          Explore
        </button>
      </form>

      {suggestedEntities.length > 0 && (
        <div className="mb-2 flex flex-wrap gap-1">
          {suggestedEntities.map((entity) => (
            <button
              key={entity}
              type="button"
              onClick={() => onSearch(entity)}
              className={`rounded border px-1.5 py-0.5 text-[11px] transition ${
                entity === entityName
                  ? "border-purple-400 text-purple-300"
                  : "border-slate-600 text-slate-300 hover:border-purple-400 hover:text-purple-300"
              }`}
            >
              {entity}
            </button>
          ))}
        </div>
      )}

      <div
        ref={containerRef}
        className="relative min-h-[280px] flex-1 overflow-hidden rounded border border-slate-800 bg-slate-950"
      >
        {isLoading && <p className="absolute left-0 top-0 p-3 text-xs text-slate-400">Loading subgraph…</p>}
        {error && <p className="absolute left-0 top-0 p-3 text-xs text-red-400">{error}</p>}
        {!isLoading && !error && subgraph && subgraph.nodes.length === 0 && (
          <p className="absolute left-0 top-0 p-3 text-xs text-slate-400">
            No graph data found for &quot;{entityName}&quot;.
          </p>
        )}
        {!isLoading && !error && !subgraph && (
          <p className="absolute left-0 top-0 p-3 text-xs text-slate-500">
            Search an entity, or click a suggestion from a citation, to explore its subgraph.
          </p>
        )}
        <svg ref={svgRef} className="h-full w-full" />
      </div>

      {usedLabels.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-slate-400">
          {usedLabels.map((label) => (
            <div key={label} className="flex items-center gap-1.5">
              <span
                className="inline-block h-2.5 w-2.5 rounded-full"
                style={{ backgroundColor: LABEL_COLORS[label] }}
              />
              {label}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
