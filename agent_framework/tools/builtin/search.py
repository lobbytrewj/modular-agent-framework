from __future__ import annotations

import re
from typing import Any, Callable, Optional

from agent_framework.tools.base import Tool, ToolPermission

# Search, with the network left out by default.
#
# The tool has two backends behind one interface:
#
#   MOCK (default)  ranks a small built-in corpus. Deterministic, offline and
#                   free, which is what a test suite needs - an assertion
#                   about search results cannot depend on what the live web
#                   said this morning.
#   WEB             any callable the caller supplies. This module deliberately
#                   ships no HTTP client: which provider, which key and which
#                   rate limit are deployment decisions, and hard-coding one
#                   would make the offline path the exception instead of the
#                   default.
#
# Both return the same shape, so an agent prompt written against the mock
# backend works unchanged against a real one.

# A backend takes (query, max_results) and returns a list of result dicts.
SearchBackend = Callable[[str, int], list[dict[str, Any]]]

DEFAULT_MAX_RESULTS = 3
RESULT_LIMIT = 10

# The offline corpus. Small on purpose: it exists so tests and demos have
# something stable to retrieve, not to be a knowledge base.
MOCK_CORPUS: list[dict[str, str]] = [
    {
        "title": "Multi-agent LLM systems: an overview",
        "url": "https://example.test/multi-agent-overview",
        "snippet": (
            "Multi-agent systems split a task across specialised LLM agents - "
            "a router picks the handler, workers run in parallel, and an "
            "orchestrator merges the results."
        ),
    },
    {
        "title": "Tool use and function calling in language models",
        "url": "https://example.test/tool-use",
        "snippet": (
            "Models call external tools by emitting a structured call against "
            "a declared schema; the runtime validates the arguments, executes "
            "the tool and feeds the result back into the conversation."
        ),
    },
    {
        "title": "Sandboxing and permissions for agent tools",
        "url": "https://example.test/agent-permissions",
        "snippet": (
            "Least privilege applies to agents as much as to processes: grant "
            "read-only access by default and require an explicit write "
            "permission before a tool may modify files."
        ),
    },
    {
        "title": "Retrieval-augmented generation in practice",
        "url": "https://example.test/rag-in-practice",
        "snippet": (
            "RAG grounds a model's answer in retrieved documents, trading some "
            "latency for citations and a much lower hallucination rate."
        ),
    },
    {
        "title": "Evaluating agent workflows",
        "url": "https://example.test/evaluating-agents",
        "snippet": (
            "Evaluator-optimizer loops score a draft against a rubric and feed "
            "the critique back to the generator until it passes a threshold."
        ),
    },
]

# Words too common to say anything about relevance. Kept tiny - a real stop
# list belongs to a real search engine, and this one only has to stop "the"
# from matching every document.
_STOP_WORDS = frozenset(
    {"a", "an", "and", "are", "for", "how", "in", "is", "of", "on", "or",
     "the", "to", "what", "with"}
)

_WORD = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [word for word in _WORD.findall(text.lower()) if word not in _STOP_WORDS]


def _score(query_terms: list[str], document: dict[str, str]) -> int:
    """How well one document answers the query.

    A title hit counts double: the title is the document's own summary of
    itself, so a term appearing there is a stronger signal than the same term
    buried in the body.
    """
    title_terms = set(_tokenize(document["title"]))
    body_terms = set(_tokenize(document["snippet"]))
    return sum(
        2 * (term in title_terms) + (term in body_terms) for term in query_terms
    )


def mock_search(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, Any]]:
    """Rank the built-in corpus against `query`. Deterministic by construction.

    Ties break on corpus order rather than arbitrarily, so the same query
    returns the same list in the same order on every run - the property that
    makes this usable in an assertion.
    """
    terms = _tokenize(query)
    scored = [
        (_score(terms, document), position, document)
        for position, document in enumerate(MOCK_CORPUS)
    ]
    hits = sorted(
        (entry for entry in scored if entry[0] > 0),
        key=lambda entry: (-entry[0], entry[1]),
    )
    return [
        {**document, "score": score, "source": "mock"}
        for score, _, document in hits[:max_results]
    ]


def search(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    backend: Optional[SearchBackend] = None,
) -> dict[str, Any]:
    """Run one search and return results plus the backend that produced them.

    Naming the backend in the output matters more than it looks: an agent (or
    a reader of the audit log) can otherwise not tell a real web result from a
    canned fixture, and "the mock backend was still wired up in production" is
    exactly the failure this makes visible.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    if not isinstance(max_results, int) or max_results < 1:
        raise ValueError("max_results must be a positive integer")
    max_results = min(max_results, RESULT_LIMIT)

    if backend is None:
        results = mock_search(query, max_results)
        backend_name = "mock"
    else:
        results = list(backend(query, max_results))[:max_results]
        backend_name = getattr(backend, "__name__", type(backend).__name__)

    return {
        "query": query,
        "backend": backend_name,
        "result_count": len(results),
        "results": results,
    }


SEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "What to search for."},
        "max_results": {
            "type": "integer",
            "description": (
                f"How many results to return (1-{RESULT_LIMIT}). "
                f"Defaults to {DEFAULT_MAX_RESULTS}."
            ),
        },
    },
    "required": ["query"],
}


def build_search_tool(backend: Optional[SearchBackend] = None) -> Tool:
    """The search tool, offline unless a `backend` is supplied.

    A web backend is still READ_ONLY here: it changes nothing. Note that it
    does send the query off the machine, so a deployment that treats
    exfiltration as the greater risk should gate it behind its own permission
    rather than reusing READ_ONLY.
    """
    description = (
        "Search for information on a topic and get back ranked results with "
        "titles, URLs and snippets."
    )
    if backend is None:
        description += (
            " Currently backed by a small offline corpus, so results are "
            "illustrative rather than current."
        )

    return Tool(
        name="search",
        description=description,
        func=lambda query, max_results=DEFAULT_MAX_RESULTS: search(
            query, max_results, backend=backend
        ),
        parameters_schema=SEARCH_SCHEMA,
        required_permission=ToolPermission.READ_ONLY,
    )
