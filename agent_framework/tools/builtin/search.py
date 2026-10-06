from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from typing import Any, Callable, Optional

from agent_framework.tools.base import Tool, ToolPermission

# Search, with the network left out by default.
#
# Three backends sit behind one interface, chosen per call in this order:
#
#   INJECTED   any callable the caller supplies (a test's mock responder, or a
#              deployment's own provider with its own key and rate limit).
#   LIVE       DuckDuckGo - the `ddgs` package when installed, else its
#              keyless Instant Answer REST endpoint over urllib. Opt-in only,
#              via AGENT_FRAMEWORK_SEARCH_BACKEND=live, so a machine that
#              happens to have network access does not quietly turn the test
#              suite into a test of the internet.
#   MOCK       ranks a small built-in corpus. The default, and the fallback
#              when the live path fails: deterministic, offline and free,
#              which is what a test suite needs - an assertion about search
#              results cannot depend on what the live web said this morning.
#
# All three return the same shape, so an agent prompt written against the mock
# backend works unchanged against a real one.

# A backend takes (query, max_results) and returns a list of result dicts.
SearchBackend = Callable[[str, int], list[dict[str, Any]]]

DEFAULT_MAX_RESULTS = 3
WEB_DEFAULT_MAX_RESULTS = 5
RESULT_LIMIT = 10

# "live" enables the network path; anything else (or unset) keeps search
# offline. Read on every call rather than at import, so the process-wide
# default registry follows the environment instead of freezing it.
SEARCH_BACKEND_ENV = "AGENT_FRAMEWORK_SEARCH_BACKEND"

# Upper bound on one live request. A search that cannot answer in this long
# falls back to the offline corpus rather than stalling the agent's turn.
LIVE_TIMEOUT_SECONDS = 8.0

INSTANT_ANSWER_URL = "https://api.duckduckgo.com/"

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


# --- Live providers ---


class LiveSearchUnavailable(RuntimeError):
    """Every live provider failed; the message says why each one did."""


def live_search_enabled() -> bool:
    return os.environ.get(SEARCH_BACKEND_ENV, "").strip().lower() == "live"


def ddgs_search(query: str, max_results: int) -> list[dict[str, Any]]:
    """Web results from the `ddgs` package (or its older name, duckduckgo_search).

    Raises ImportError when neither is installed, which the provider chain
    treats as "try the next one" rather than as a failure worth reporting loudly.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS  # the package's pre-rename name

    hits = DDGS(timeout=LIVE_TIMEOUT_SECONDS).text(query, max_results=max_results) or []
    return [
        {
            "title": hit.get("title", ""),
            "url": hit.get("href") or hit.get("url", ""),
            "snippet": hit.get("body") or hit.get("snippet", ""),
            "source": "ddgs",
        }
        for hit in hits[:max_results]
    ]


def duckduckgo_instant_answer(query: str, max_results: int) -> list[dict[str, Any]]:
    """Results from DuckDuckGo's keyless Instant Answer API, using only urllib.

    This is an answers endpoint, not full web search: it returns an abstract
    and related topics for entity-like queries and nothing for most others.
    It is here so live mode works without an extra dependency; install `ddgs`
    for real result pages.
    """
    url = INSTANT_ANSWER_URL + "?" + urllib.parse.urlencode(
        {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
    )
    request = urllib.request.Request(url, headers={"User-Agent": "agent-framework/1.0"})
    with urllib.request.urlopen(request, timeout=LIVE_TIMEOUT_SECONDS) as response:
        payload = json.loads(response.read().decode("utf-8"))

    results: list[dict[str, Any]] = []
    if payload.get("AbstractText") and payload.get("AbstractURL"):
        results.append({
            "title": payload.get("Heading") or query,
            "url": payload["AbstractURL"],
            "snippet": payload["AbstractText"],
            "source": "duckduckgo_instant_answer",
        })

    # Related topics arrive either flat or grouped under a "Topics" key.
    topics: list[dict[str, Any]] = []
    for topic in payload.get("RelatedTopics", []):
        topics.extend(topic.get("Topics", [topic]))
    for topic in topics:
        text, link = topic.get("Text"), topic.get("FirstURL")
        if text and link:
            results.append({
                "title": text.split(" - ")[0],
                "url": link,
                "snippet": text,
                "source": "duckduckgo_instant_answer",
            })

    return results[:max_results]


# Tried in order; the first that answers wins. A module attribute so a test
# can swap in a provider that simulates being offline.
LIVE_PROVIDERS: list[SearchBackend] = [ddgs_search, duckduckgo_instant_answer]


def live_search(query: str, max_results: int) -> tuple[list[dict[str, Any]], str]:
    """Results plus the name of the provider that produced them.

    Raises LiveSearchUnavailable when no provider answered - no package, no
    network, a timeout, a malformed response. The caller decides what to do
    about that; `search` falls back to the offline corpus.
    """
    failures = []
    for provider in LIVE_PROVIDERS:
        try:
            return list(provider(query, max_results)), provider.__name__
        except Exception as exc:  # ImportError, URLError, timeouts, bad JSON alike
            failures.append(f"{provider.__name__}: {type(exc).__name__}: {exc}")
    raise LiveSearchUnavailable("; ".join(failures) or "no live providers configured")


# --- The tools ---


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

    fallback_reason = None
    if backend is not None:
        results = list(backend(query, max_results))[:max_results]
        backend_name = getattr(backend, "__name__", type(backend).__name__)
    elif live_search_enabled():
        try:
            results, backend_name = live_search(query, max_results)
            results = results[:max_results]
        except LiveSearchUnavailable as exc:
            # Degrade rather than fail the agent's turn - but say so, so the
            # agent and the audit log both know these results are canned.
            results = mock_search(query, max_results)
            backend_name = "mock"
            fallback_reason = f"live search unavailable ({exc})"
    else:
        results = mock_search(query, max_results)
        backend_name = "mock"

    return {
        "query": query,
        "backend": backend_name,
        "result_count": len(results),
        "results": results,
        "fallback_reason": fallback_reason,
    }


def format_search_results(found: dict[str, Any]) -> str:
    """Render a `search` result as the numbered text block a model reads best."""
    header = f"Search results for {found['query']!r} (backend: {found['backend']})"
    lines = [header]
    if found.get("fallback_reason"):
        lines.append(f"Note: {found['fallback_reason']}; showing offline results.")
    if not found["results"]:
        lines.append("No results found.")
    for rank, hit in enumerate(found["results"], start=1):
        lines.append(f"{rank}. {hit.get('title', '')}")
        lines.append(f"   URL: {hit.get('url', '')}")
        if hit.get("snippet"):
            lines.append(f"   {hit['snippet']}")
    return "\n".join(lines)


def web_search(
    query: str,
    max_results: int = WEB_DEFAULT_MAX_RESULTS,
    backend: Optional[SearchBackend] = None,
) -> str:
    """Search and return the results as text: title, URL and snippet per hit.

    The same engine as `search` - same backend choice, same fallback - with
    the output already rendered for a prompt. Callers that want to pick the
    results apart should use `search` and keep the dict.
    """
    return format_search_results(search(query, max_results, backend=backend))


def _search_schema(default_max_results: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "max_results": {
                "type": "integer",
                "description": (
                    f"How many results to return (1-{RESULT_LIMIT}). "
                    f"Defaults to {default_max_results}."
                ),
            },
        },
        "required": ["query"],
    }


SEARCH_SCHEMA = _search_schema(DEFAULT_MAX_RESULTS)
WEB_SEARCH_SCHEMA = _search_schema(WEB_DEFAULT_MAX_RESULTS)

_OFFLINE_NOTE = (
    " Unless live search is enabled, results come from a small offline "
    "corpus and are illustrative rather than current."
)


def build_search_tool(backend: Optional[SearchBackend] = None) -> Tool:
    """The search tool: structured results, offline unless told otherwise.

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
        description += _OFFLINE_NOTE

    return Tool(
        name="search",
        description=description,
        func=lambda query, max_results=DEFAULT_MAX_RESULTS: search(
            query, max_results, backend=backend
        ),
        parameters_schema=SEARCH_SCHEMA,
        required_permission=ToolPermission.READ_ONLY,
    )


def build_web_search_tool(backend: Optional[SearchBackend] = None) -> Tool:
    """`web_search`: the same engine as `search`, answering in text.

    READ_ONLY for the same reason as `search`, and with the same caveat - in
    live mode the query leaves the machine.
    """
    description = (
        "Search the web and get back a numbered list of results, each with "
        "a title, URL and snippet."
    )
    if backend is None:
        description += _OFFLINE_NOTE

    return Tool(
        name="web_search",
        description=description,
        func=lambda query, max_results=WEB_DEFAULT_MAX_RESULTS: web_search(
            query, max_results, backend=backend
        ),
        parameters_schema=WEB_SEARCH_SCHEMA,
        required_permission=ToolPermission.READ_ONLY,
    )
