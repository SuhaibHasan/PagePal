from __future__ import annotations

import re

import anthropic
from pydantic import BaseModel, Field

from retrieval.models import RetrievalResult

# claude-sonnet-4-6 doesn't exist; claude-sonnet-5 is the current Sonnet model.
SONNET_MODEL = "claude-sonnet-5"

MAX_CONTEXT_CHUNKS = 8

SYSTEM_PROMPT = (
    "You are a production support assistant. Answer only from the provided "
    "context. Be specific about services, error codes, and remediation steps. "
    "If the answer isn't in the context, say so. Cite sources as [Source N]. "
    "If graph paths are provided, use them to explain service impact chains."
)

CITATION_PATTERN = re.compile(r"\[Source (\d+)\]")


class Citation(BaseModel):
    title: str | None = None
    url: str | None = None
    retrieval_path: str
    graph_path: str | None = None


class AnswerResult(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)


class AnswerGenerator:
    def __init__(
        self,
        anthropic_client: anthropic.Anthropic | None = None,
        model: str = SONNET_MODEL,
    ) -> None:
        self._llm = anthropic_client or anthropic.Anthropic()
        self._model = model

    def generate(self, query: str, context: list[RetrievalResult]) -> AnswerResult:
        top_context = context[:MAX_CONTEXT_CHUNKS]
        user_message = self._build_user_message(query, top_context)

        answer_text = self._stream_answer(user_message)
        citations = self._build_citations(answer_text, top_context)
        return AnswerResult(answer=answer_text, citations=citations)

    def _stream_answer(self, user_message: str) -> str:
        with self._llm.messages.stream(
            model=self._model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            chunks = list(stream.text_stream)
        return "".join(chunks)

    @staticmethod
    def _build_user_message(query: str, context: list[RetrievalResult]) -> str:
        if not context:
            chunks_section = "No relevant context was found in the knowledge base."
        else:
            chunks_section = "\n\n".join(
                f"[Source {i + 1}]: {result.content}" for i, result in enumerate(context)
            )

        parts = [chunks_section]

        graph_paths = AnswerGenerator._collect_graph_paths(context)
        if graph_paths:
            parts.append("[Graph Context]:\n" + "\n".join(graph_paths))

        parts.append(f"Question: {query}")
        return "\n\n".join(parts)

    @staticmethod
    def _collect_graph_paths(context: list[RetrievalResult]) -> list[str]:
        seen: set[str] = set()
        paths: list[str] = []
        for result in context:
            for path in result.metadata.get("graph_paths") or []:
                if path not in seen:
                    seen.add(path)
                    paths.append(path)
        return paths

    @staticmethod
    def _build_citations(answer_text: str, context: list[RetrievalResult]) -> list[Citation]:
        cited_numbers = sorted({int(match) for match in CITATION_PATTERN.findall(answer_text)})

        citations = []
        for number in cited_numbers:
            if not 1 <= number <= len(context):
                continue
            result = context[number - 1]
            graph_paths = result.metadata.get("graph_paths")
            citations.append(
                Citation(
                    title=result.metadata.get("title"),
                    url=result.metadata.get("url"),
                    retrieval_path=result.source_type,
                    graph_path="; ".join(graph_paths) if graph_paths else None,
                )
            )
        return citations
