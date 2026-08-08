"use client";

import type { Filters } from "@/app/lib/types";

const SEVERITY_OPTIONS = ["P1", "P2", "P3", "P4"];

export function Sidebar({
  filters,
  onChange,
}: {
  filters: Filters;
  onChange: (filters: Filters) => void;
}) {
  const hasActiveFilters = Boolean(
    filters.service || filters.severity || filters.dateFrom || filters.dateTo,
  );

  return (
    <aside className="flex w-64 shrink-0 flex-col gap-5 overflow-y-auto border-r border-slate-800 bg-slate-900/60 p-4">
      <div>
        <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-400">Filters</h2>
        <p className="mt-1 text-[11px] text-slate-500">
          Narrows keyword search; overrides what the assistant infers from your question.
        </p>
      </div>

      <label className="flex flex-col gap-1.5 text-sm text-slate-300">
        Service
        <input
          value={filters.service ?? ""}
          onChange={(event) => onChange({ ...filters, service: event.target.value || undefined })}
          placeholder="e.g. payment-service"
          className="rounded border border-slate-700 bg-slate-800 px-2 py-1.5 text-sm text-slate-100 placeholder:text-slate-500 focus:border-blue-500 focus:outline-none"
        />
      </label>

      <label className="flex flex-col gap-1.5 text-sm text-slate-300">
        Severity
        <select
          value={filters.severity ?? ""}
          onChange={(event) =>
            onChange({ ...filters, severity: event.target.value || undefined })
          }
          className="rounded border border-slate-700 bg-slate-800 px-2 py-1.5 text-sm text-slate-100 focus:border-blue-500 focus:outline-none"
        >
          <option value="">Any</option>
          {SEVERITY_OPTIONS.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      </label>

      <fieldset className="flex flex-col gap-1.5 text-sm text-slate-300">
        <legend className="mb-0.5">Date range</legend>
        <input
          type="date"
          aria-label="From date"
          value={filters.dateFrom ?? ""}
          onChange={(event) =>
            onChange({ ...filters, dateFrom: event.target.value || undefined })
          }
          className="rounded border border-slate-700 bg-slate-800 px-2 py-1.5 text-sm text-slate-100 focus:border-blue-500 focus:outline-none [color-scheme:dark]"
        />
        <input
          type="date"
          aria-label="To date"
          value={filters.dateTo ?? ""}
          onChange={(event) => onChange({ ...filters, dateTo: event.target.value || undefined })}
          className="rounded border border-slate-700 bg-slate-800 px-2 py-1.5 text-sm text-slate-100 focus:border-blue-500 focus:outline-none [color-scheme:dark]"
        />
      </fieldset>

      <button
        type="button"
        onClick={() => onChange({})}
        disabled={!hasActiveFilters}
        className="mt-auto rounded border border-slate-700 px-2 py-1.5 text-xs text-slate-400 transition hover:border-slate-500 hover:text-slate-200 disabled:cursor-not-allowed disabled:opacity-40"
      >
        Clear filters
      </button>
    </aside>
  );
}
