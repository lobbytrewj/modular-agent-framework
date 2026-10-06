"""Demo 3: Routing Assistant - one entry point, several sub-workflows.

    user query
        |
    RouterAgent (LLM)       {"route": ..., "reasoning": ...}; keyword rules if the reply is unusable
        |
        +-- code  -> SequentialOrchestrator: CodePlannerAgent -> TeamCoderAgent
        +-- math  -> MathAnalysisPipeline:   MathAgent -> calculator tool -> AnalystAgent
        +-- qa    -> DirectAnswerAgent
        +-- other -> FallbackAgent

Every query in a session shares one blackboard and one tracer, so the trace
(reports/demo3_trace.json) reads as the whole conversation: each route
decision with the router's reasoning, the hand-off into the sub-workflow, and
the steps and tool calls that sub-workflow ran.

    python examples/demo_routing_assistant.py                  # three sample queries
    python examples/demo_routing_assistant.py --query "..."    # your own (repeatable)
    python examples/demo_routing_assistant.py --interactive    # type queries until blank
    python examples/demo_routing_assistant.py --mock           # offline, scripted replies
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from demo_common import (  # noqa: E402
    REPORTS_DIR,
    ClientFactory,
    DemoResult,
    build_agent,
    demo_arg_parser,
    extract_calls,
    finish_run,
    load_config,
    one_line,
    say,
    scripted_client_factory,
    section,
    start_run,
)

from agent_framework.agents import BaseAgent, execute_agent  # noqa: E402
from agent_framework.core.memory import SharedWorkflowMemory  # noqa: E402
from agent_framework.core.tasks import RunResult, Task  # noqa: E402
from agent_framework.orchestration.hierarchical import OBJECT_FIRST, extract_json  # noqa: E402
from agent_framework.orchestration.router import Destination, RouterOrchestrator  # noqa: E402
from agent_framework.orchestration.sequential import SequentialOrchestrator  # noqa: E402

DEFAULT_TRACE = REPORTS_DIR / "demo3_trace.json"
DEFAULT_QUERIES = (
    "Write a Python function that checks whether a string is a palindrome, ignoring case and spaces.",
    "Calculate the total cost of 12 notebooks at $3.75 each plus 8% sales tax.",
    "Explain what a race condition is in one short paragraph.",
)

ROUTER_KEY = "request_router"
PLANNER_KEY = "code_planner"
CODER_KEY = "team_coder"
MATH_KEY = "math"
ANALYST_KEY = "analyst"
QA_KEY = "direct_answer"
FALLBACK_KEY = "fallback"

MAX_CALCULATIONS = 5

# Route key -> (what the router is told it handles, keyword rules for when the
# router's own reply cannot be used). Order is the keyword matching order.
ROUTES: dict[str, tuple[str, list[str]]] = {
    "code": (
        "writing, fixing or explaining source code",
        ["python", "code", "function", "bug", "implement", "script"],
    ),
    "math": (
        "arithmetic, calculations, prices, percentages and numeric analysis",
        ["calculate", "sum", "total", "average", "percent", "%", "how much", "math"],
    ),
    "qa": (
        "any question answered in words: facts, definitions and explanations of a "
        "concept, including technical ones, when nothing needs coding or calculating",
        ["what is", "explain", "define", "who", "why", "how does"],
    ),
}

# Replies meaning "none of the above", sent to the fallback with the router's reasoning.
_NO_ROUTE = {"other", "none", "fallback", "unknown"}


# --- The router ---


class LLMRouter(RouterOrchestrator):
    """A RouterOrchestrator whose decision is made by a router agent.

    Only the decision changes: RouterOrchestrator.run still records the route
    and its reasoning on the trace, the audit log and routing_log, and still
    hands the task and the blackboard to the destination. When the agent's
    reply names no usable route, the keyword rules decide instead, and the
    recorded reasoning says so - a trace should never present a guess as the
    model's choice.
    """

    def __init__(self, router_agent: BaseAgent, fallback_destination: Destination):
        super().__init__(fallback_destination=fallback_destination)
        self.router_agent = router_agent
        self.route_descriptions: dict[str, str] = {}
        self._shared_memory: Optional[SharedWorkflowMemory] = None

    def register_route(
        self,
        key: str,
        destination: Destination,
        keywords: Optional[list[str]] = None,
        description: str = "",
    ) -> None:
        super().register_route(key, destination, keywords=keywords)
        self.route_descriptions[key] = description or key

    def run(self, task: Task, shared_memory: Optional[SharedWorkflowMemory] = None) -> RunResult:
        # _decide_route_with_reason is not handed the blackboard, and the
        # router agent's own call should be traced like any other step.
        self._shared_memory = shared_memory
        try:
            return super().run(task, shared_memory)
        finally:
            self._shared_memory = None

    def _decide_route_with_reason(self, task: Task) -> tuple[str, str]:
        catalog = "\n".join(f"- {key}: {text}" for key, text in self.route_descriptions.items())
        decision = execute_agent(
            self.router_agent,
            Task(
                description=(
                    f"Routes:\n{catalog}\n- other: only when none of the routes above fits at all\n\n"
                    f"User request:\n{task.description}\n\n"
                    'Reply with ONLY a JSON object: {"route": "<route key>", '
                    '"reasoning": "<one short sentence>"}'
                ),
                assigned_agent=self.router_agent.name,
            ),
            self._shared_memory,
        )

        problem = f"router agent failed: {decision.error}"
        if decision.success:
            payload = extract_json(decision.output or "", OBJECT_FIRST)
            if isinstance(payload, dict):
                route = str(payload.get("route", "")).strip().lower()
                reasoning = one_line(payload.get("reasoning") or "no reasoning given", 200)
                if route in self.routes:
                    return route, f"{self.router_agent.name}: {reasoning}"
                if route in _NO_ROUTE:
                    return self._FALLBACK_KEY, f"{self.router_agent.name}: {reasoning}"
                problem = f"router named unknown route {route!r}"
            else:
                problem = "router reply was not JSON"

        key, why = super()._decide_route_with_reason(task)
        return key, f"{problem}; keyword rules: {why}"


# --- The math/analysis sub-workflow ---


class MathAnalysisPipeline:
    """Solver writes the arithmetic, the calculator does it, the analyst explains it.

    The model never does the arithmetic itself: it is asked only to write the
    expressions, which run through the solver's calculator tool - permission-
    checked against its config grants and traced - and the analyst is handed
    the verified numbers to explain.
    """

    name = "MathAnalysisPipeline"

    def __init__(self, solver: BaseAgent, analyst: BaseAgent):
        self.solver = solver
        self.analyst = analyst

    def run(self, task: Task, shared_memory: Optional[SharedWorkflowMemory] = None) -> RunResult:
        solved = execute_agent(
            self.solver,
            Task(
                description=(
                    f"{task.description}\n\n"
                    "Do not compute anything yourself. Write each calculation "
                    "needed on its own line in the form calculator(<expression>), "
                    "using only numbers, + - * / ** and parentheses. Then stop."
                ),
                assigned_agent=self.solver.name,
            ),
            shared_memory,
        )
        if not solved.success:
            return RunResult(task_id=task.id, success=False, error=solved.error)

        calculations = []
        for expression in extract_calls(solved.output or "", "calculator")[:MAX_CALCULATIONS]:
            record = self.solver.use_tool("calculator", shared_memory=shared_memory, expression=expression)
            calculations.append(
                {"expression": expression, "result": record["output"], "error": record["error"]}
            )
        if shared_memory is not None:
            shared_memory.set("calculations", calculations)

        verified = "\n".join(
            f"calculator({c['expression']}) = {c['result']}"
            if c["error"] is None
            else f"calculator({c['expression']}) failed: {c['error']}"
            for c in calculations
        ) or "(no calculations were produced)"

        explained = execute_agent(
            self.analyst,
            Task(
                description=(
                    f"Question:\n{task.description}\n\n"
                    f"Verified calculator results:\n{verified}\n\n"
                    "Answer the question in two or three sentences using these "
                    "exact results. Do not recompute them."
                ),
                assigned_agent=self.analyst.name,
            ),
            shared_memory,
        )
        return RunResult(
            task_id=task.id, success=explained.success, output=explained.output, error=explained.error
        )


# --- The demo ---


def build_assistant(config: dict[str, dict], client_factory: Optional[ClientFactory]) -> LLMRouter:
    """The router and its sub-workflows, every agent built from agents.json."""

    def agent(key: str, **overrides: Any):
        # One blackboard spans the whole session; an agent answering query 3
        # should not have query 1's artifacts pasted into its prompt.
        overrides.setdefault("context_keys", [])
        return build_agent(config, key, client_factory=client_factory, **overrides)

    router = LLMRouter(agent(ROUTER_KEY), fallback_destination=agent(FALLBACK_KEY))
    destinations: dict[str, Destination] = {
        "code": SequentialOrchestrator([agent(PLANNER_KEY), agent(CODER_KEY)]),
        "math": MathAnalysisPipeline(agent(MATH_KEY), agent(ANALYST_KEY)),
        "qa": agent(QA_KEY),
    }
    for key, (description, keywords) in ROUTES.items():
        router.register_route(key, destinations[key], keywords=keywords, description=description)
    return router


def run_demo(
    queries: Iterable[str] = DEFAULT_QUERIES,
    *,
    client_factory: Optional[ClientFactory] = None,
    config_path: Optional[str] = None,
    trace_path: Path = DEFAULT_TRACE,
    quiet: bool = False,
) -> DemoResult:
    config = load_config(config_path)
    shared_memory, tracer = start_run("routing_assistant", mock=client_factory is not None)
    assistant = build_assistant(config, client_factory)

    answers: list[dict[str, Any]] = []
    for number, query in enumerate(queries, start=1):
        section(f"Query {number}: {query}", quiet)
        result = assistant.run(Task(description=query, assigned_agent="router"), shared_memory)
        decision = assistant.routing_log[-1]
        answers.append(
            {
                "query": query,
                "route": decision["route"],
                "reasoning": decision["reasoning"],
                "success": result.success,
                "output": result.output,
                "error": result.error,
            }
        )
        say(f"  route:     {decision['route']}", quiet)
        say(f"  reasoning: {decision['reasoning']}", quiet)
        say(f"  answer:\n{result.output if result.success else 'FAILED: ' + str(result.error)}", quiet)
    shared_memory.set("session", [{k: a[k] for k in ("query", "route", "success")} for a in answers])

    section("Routes taken", quiet)
    for answer in answers:
        say(f"  {answer['route']:<9} {one_line(answer['query'], 90)}", quiet)

    path = finish_run(tracer, shared_memory, trace_path, quiet)
    success = bool(answers) and all(answer["success"] for answer in answers)
    output = "\n\n".join(answer["output"] or "" for answer in answers)
    return DemoResult(success, output, shared_memory, tracer, path, {"answers": answers})


# --- Offline script for --mock ---


def mock_route(query: str) -> tuple[str, str]:
    """What the scripted router picks, and why - the keyword table, phrased as a model would."""
    lowered = query.lower()
    for key, (description, keywords) in ROUTES.items():
        hit = next((keyword for keyword in keywords if keyword in lowered), None)
        if hit:
            return key, f"The request mentions '{hit}', which is {description}."
    return "other", "The request does not fit code, math or factual Q&A."


def mock_responder(key: str, system_prompt: str, user_message: str) -> str:
    if key == ROUTER_KEY:
        query = user_message.split("User request:\n", 1)[-1].split("\n\n", 1)[0]
        route, reasoning = mock_route(query)
        return f'{{"route": "{route}", "reasoning": "{reasoning}"}}'
    if key == PLANNER_KEY:
        return "Signature: is_palindrome(text: str) -> bool\nNormalise: lowercase, drop spaces.\nEdge cases: empty string is a palindrome; non-str raises TypeError."
    if key == CODER_KEY:
        return (
            "```python\ndef is_palindrome(text):\n    if not isinstance(text, str):\n"
            "        raise TypeError('text must be a string')\n"
            "    cleaned = text.replace(' ', '').lower()\n    return cleaned == cleaned[::-1]\n```"
        )
    if key == MATH_KEY:
        if "12 notebooks" in user_message:
            return "calculator(12 * 3.75 * 1.08)"
        numbers = re.findall(r"\d+(?:\.\d+)?", user_message.split("\n\n", 1)[0])
        return f"calculator({' + '.join(numbers) or '0'})"
    if key == ANALYST_KEY:
        results = re.findall(r"\) = (\S+)", user_message)
        return f"The result is {results[0] if results else 'unavailable'}."
    if key == QA_KEY:
        return (
            "A race condition is a bug where the outcome depends on the timing of "
            "concurrent operations on shared state, so the program can behave "
            "differently from run to run."
        )
    return "I can help with code, calculations or factual questions - could you rephrase?"


def _interactive_queries() -> Iterable[str]:
    while True:
        try:
            query = input("\nquery> ").strip()
        except EOFError:
            return
        if not query or query.lower() in {"quit", "exit"}:
            return
        yield query


def main(argv: Optional[list[str]] = None) -> int:
    parser = demo_arg_parser(__doc__.splitlines()[0], DEFAULT_TRACE)
    parser.add_argument("--query", action="append", help="a query to route (repeatable)")
    parser.add_argument(
        "--interactive", action="store_true", help="read queries from the terminal until a blank line"
    )
    args = parser.parse_args(argv)

    if args.interactive:
        queries: Iterable[str] = _interactive_queries()
    else:
        queries = args.query or DEFAULT_QUERIES

    result = run_demo(
        queries,
        client_factory=scripted_client_factory(mock_responder) if args.mock else None,
        config_path=args.config,
        trace_path=Path(args.trace),
        quiet=args.quiet,
    )
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
