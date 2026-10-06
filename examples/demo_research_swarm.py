"""Demo 1: Research Swarm - orchestrator-worker research with a quality gate.

    research question
           |
    ResearchPlannerAgent           decomposes the question into subtopics
           |
     +-----+-----+                 ParallelOrchestrator fan-out
     v     v     v
    ResearchAgent-1..N             each: search tool -> grounded findings
     +-----+-----+                 fan-in
           v
    ResearchSynthesizerAgent       one structured report -> shared_memory["research_report"]
           |
    ResearchEvaluatorAgent         verdict; incomplete -> one revision of the report

Every step runs on one SharedWorkflowMemory with a Tracer attached, so the
worker threads, the search calls and the evaluator's verdicts all land in one
trace, exported to reports/demo1_trace.json.

    python examples/demo_research_swarm.py            # local Qwen model
    python examples/demo_research_swarm.py --mock     # offline, scripted replies
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from demo_common import (  # noqa: E402
    REPORTS_DIR,
    ClientFactory,
    DemoResult,
    build_agent,
    demo_arg_parser,
    finish_run,
    load_config,
    one_line,
    say,
    scripted_client_factory,
    section,
    start_run,
)

from agent_framework.agents import LLMAgent, execute_agent  # noqa: E402
from agent_framework.core.memory import SharedWorkflowMemory  # noqa: E402
from agent_framework.core.tasks import RunResult, Task  # noqa: E402
from agent_framework.orchestration import (  # noqa: E402
    ParallelOrchestrator,
    VerificationVerdict,
    parse_verification_verdict,
)
from agent_framework.orchestration.hierarchical import ARRAY_FIRST, extract_json  # noqa: E402

DEFAULT_TRACE = REPORTS_DIR / "demo1_trace.json"
DEFAULT_QUESTION = (
    "How do multi-agent LLM systems stay reliable? Cover how they use tools, "
    "how tool permissions are sandboxed, and how agent workflows are evaluated."
)

PLANNER_KEY = "research_planner"
WORKER_KEY = "researcher"
SYNTHESIZER_KEY = "research_synthesizer"
EVALUATOR_KEY = "research_evaluator"

REPORT_KEY = "research_report"  # the synthesizer's output_key in agents.json
FINDINGS_PREFIX = "findings/"

_SUBTOPIC_FIELDS = ("subtopic", "topic", "title", "task", "name")


# --- Step 1: decomposition ---


def parse_subtopics(text: str, limit: int) -> list[str]:
    """Read the planner's subtopics: a JSON array first, then a bulleted list.

    Small models ignore "JSON only" often enough that the list fallback earns
    its place. Duplicates are dropped and the count is capped, since every
    subtopic costs a worker.
    """
    payload = extract_json(text or "", ARRAY_FIRST)
    candidates: list[str] = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                item = next((item[f] for f in _SUBTOPIC_FIELDS if isinstance(item.get(f), str)), None)
            if isinstance(item, str):
                candidates.append(item)
    if not candidates:
        for line in (text or "").splitlines():
            match = re.match(r"^\s*(?:[-*•]|\d+[.)])\s+(.+)$", line)
            if match:
                candidates.append(match.group(1))

    subtopics: list[str] = []
    for candidate in candidates:
        cleaned = candidate.strip().strip("\"'*").strip()
        if cleaned and cleaned.lower() not in {s.lower() for s in subtopics}:
            subtopics.append(cleaned)
    return subtopics[:limit]


# --- Step 2: the workers ---


class SubtopicResearchAgent(LLMAgent):
    """A researcher assigned one subtopic: search first, then write findings.

    ParallelOrchestrator hands every worker the same task - the whole question
    - so the subtopic travels on the agent instead. The search goes through
    `use_tool`, so it is permission-checked against the researcher's config
    grants and lands in the trace as a TOOL_CALL.
    """

    def __init__(self, subtopic: str, finding_key: str, **kwargs: Any):
        super().__init__(**kwargs)
        self.subtopic = subtopic
        self.finding_key = finding_key

    def execute(
        self, task: Task, shared_memory: Optional[SharedWorkflowMemory] = None
    ) -> RunResult:
        searched = self.use_tool(
            "search", shared_memory=shared_memory, query=self.subtopic, max_results=3
        )
        hits = searched["output"]["results"] if searched["success"] else []
        if hits:
            sources = "\n".join(
                f"- {hit['title']} ({hit['url']}): {hit['snippet']}" for hit in hits
            )
        elif searched["success"]:
            sources = "- (the search returned no results)"
        else:
            sources = f"- (search unavailable: {searched['error']})"

        focused = Task(
            description=(
                f"Overall research question:\n{task.description}\n\n"
                f"Your subtopic:\n{self.subtopic}\n\n"
                f"Search results:\n{sources}\n\n"
                "Write 3-5 concise findings about your subtopic only, grounded "
                "in the search results above. Cite the source title in brackets "
                "after each finding."
            ),
            assigned_agent=self.name,
            input_data=task.input_data,
        )
        result = super().execute(focused, shared_memory)
        if result.success and shared_memory is not None:
            shared_memory.set(
                self.finding_key,
                {
                    "subtopic": self.subtopic,
                    "sources": [hit["title"] for hit in hits],
                    "findings": result.output,
                },
            )
        return RunResult(
            task_id=task.id, success=result.success, output=result.output, error=result.error
        )


# --- Step 4: evaluation ---


def build_evaluation_task(question: str, report: str, evaluator_name: str) -> Task:
    return Task(
        description=(
            f"Original research question:\n{question}\n\n"
            f"Report to review:\n{report}\n\n"
            "Does the report answer every part of the question, use the sections "
            "Overview, Key Findings, Open Questions and Sources, and stay grounded "
            "in its cited sources?\n\n"
            'Reply with ONLY a JSON object: {"is_complete": <true or false>, '
            '"feedback": "<the specific gaps to fix>"}'
        ),
        assigned_agent=evaluator_name,
    )


def build_revision_task(
    question: str,
    report: str,
    feedback: str,
    shared_memory: SharedWorkflowMemory,
    synthesizer_name: str,
) -> Task:
    """The synthesizer's second pass: the findings again, its report, the critique.

    The findings are read back off the blackboard, where each worker published
    them, rather than kept around from the fan-out - the blackboard is the
    run's record of what the workers produced.
    """
    blocks = []
    for key in shared_memory.keys():
        if key.startswith(FINDINGS_PREFIX):
            finding = shared_memory.get(key)
            blocks.append(f"--- {finding['subtopic']} ---\n{finding['findings']}")
    return Task(
        description=(
            f"Original Request:\n{question}\n\n"
            "Worker Findings:\n" + "\n\n".join(blocks) + "\n\n"
            f"Your previous report:\n{report}\n\n"
            f"A reviewer found these problems with it:\n{feedback}\n\n"
            "Rewrite the report so every problem is fixed. Keep the sections "
            "Overview, Key Findings, Open Questions and Sources, and reply with "
            "the complete report only."
        ),
        assigned_agent=synthesizer_name,
    )


# --- The demo ---


def run_demo(
    question: str = DEFAULT_QUESTION,
    *,
    client_factory: Optional[ClientFactory] = None,
    config_path: Optional[str] = None,
    trace_path: Path = DEFAULT_TRACE,
    max_subtopics: int = 3,
    max_revisions: int = 1,
    quiet: bool = False,
) -> DemoResult:
    config = load_config(config_path)
    shared_memory, tracer = start_run(
        "research_swarm", question=question, mock=client_factory is not None
    )

    def agent(key: str, **overrides: Any):
        return build_agent(config, key, client_factory=client_factory, **overrides)

    planner = agent(PLANNER_KEY)
    synthesizer = agent(SYNTHESIZER_KEY)
    evaluator = agent(EVALUATOR_KEY)

    # --- 1. Decompose ---
    section("1. Planner decomposes the question", quiet)
    plan = execute_agent(
        planner,
        Task(
            description=(
                f"Research question:\n{question}\n\n"
                f"Split it into at most {max_subtopics} distinct subtopics. Reply "
                "with ONLY a JSON array of short subtopic strings."
            ),
            assigned_agent=planner.name,
        ),
        shared_memory,
    )
    subtopics = parse_subtopics(plan.output or "", max_subtopics) if plan.success else []
    if not subtopics:
        # Degrade to a single worker on the whole question rather than stopping:
        # the swarm still runs, just without the fan-out.
        say("  planner gave no usable subtopics; researching the question as one topic", quiet)
        subtopics = [question]
    shared_memory.set("subtopics", subtopics)
    for position, subtopic in enumerate(subtopics, start=1):
        say(f"  {position}. {subtopic}", quiet)

    # --- 2. Fan out ---
    section(f"2. {len(subtopics)} researchers work in parallel", quiet)
    workers = []
    for position, subtopic in enumerate(subtopics, start=1):
        worker = agent(
            WORKER_KEY,
            agent_cls=SubtopicResearchAgent,
            name=f"{config[WORKER_KEY]['name']}-{position}",
            subtopic=subtopic,
            finding_key=f"{FINDINGS_PREFIX}{position}",
            # Workers run concurrently on one blackboard; showing each worker
            # its siblings' half-finished findings would make every prompt
            # depend on thread timing.
            context_keys=[],
        )
        tracer.record_delegation(
            planner, worker, subtopic, reason="subtopic assignment", workflow="research_swarm"
        )
        workers.append(worker)

    swarm = ParallelOrchestrator(workers, synthesizer, max_workers=len(workers))
    synthesis = swarm.run(Task(description=question, assigned_agent="research_swarm"), shared_memory)
    for worker, result in zip(workers, swarm.step_results):
        status = "ok" if result.success else f"FAILED: {result.error}"
        say(f"  {worker.name}: {status}", quiet)

    if not synthesis.success:
        say(f"\nSynthesis failed: {synthesis.error}", quiet)
        path = finish_run(tracer, shared_memory, trace_path, quiet)
        return DemoResult(False, "", shared_memory, tracer, path, {"subtopics": subtopics})

    # --- 3 + 4. Synthesize, evaluate, revise ---
    section("3. Synthesizer report -> shared_memory['research_report']", quiet)
    report = shared_memory.get(REPORT_KEY) or synthesis.output or ""
    verdicts: list[VerificationVerdict] = []

    for attempt in range(1, max_revisions + 2):
        section(f"4. Evaluator review (attempt {attempt})", quiet)
        review = execute_agent(
            evaluator, build_evaluation_task(question, report, evaluator.name), shared_memory
        )
        verdict = (
            parse_verification_verdict(review.output or "")
            if review.success
            else VerificationVerdict(False, f"evaluator failed: {review.error}")
        )
        verdicts.append(verdict)
        tracer.record_verification(
            evaluator,
            is_complete=verdict.is_complete,
            feedback=verdict.feedback,
            attempt=attempt,
            raw_response=verdict.raw_response,
            workflow="research_swarm",
        )
        say(f"  is_complete={verdict.is_complete}", quiet)
        if not verdict.is_complete:
            say(f"  feedback: {one_line(verdict.feedback)}", quiet)

        if verdict.is_complete or attempt > max_revisions or not review.success:
            break

        revision = build_revision_task(
            question, report, verdict.feedback, shared_memory, synthesizer.name
        )
        tracer.record_delegation(
            evaluator, synthesizer, revision,
            reason=f"attempt {attempt} judged incomplete", workflow="research_swarm",
        )
        revised = execute_agent(synthesizer, revision, shared_memory)
        if not revised.success:
            break
        report = shared_memory.get(REPORT_KEY) or revised.output or report

    final = verdicts[-1]
    shared_memory.set(
        "research_evaluation",
        {"is_complete": final.is_complete, "feedback": final.feedback, "attempts": len(verdicts)},
    )

    section("Final report", quiet)
    say(report, quiet)
    path = finish_run(tracer, shared_memory, trace_path, quiet)
    return DemoResult(
        success=final.is_complete,
        output=report,
        shared_memory=shared_memory,
        tracer=tracer,
        trace_path=path,
        details={"subtopics": subtopics, "verdicts": verdicts, "worker_count": len(workers)},
    )


# --- Offline script for --mock ---


def mock_responder(key: str, system_prompt: str, user_message: str) -> str:
    """Deterministic stand-ins for each role, shaped like real model replies."""
    if key == PLANNER_KEY:
        return (
            '["Tool use and function calling", "Sandboxing and permissions for '
            'agent tools", "Evaluating agent workflows"]'
        )
    if key == WORKER_KEY:
        subtopic = _field(user_message, "Your subtopic:")
        titles = re.findall(r"^- (.+?) \(https?://", user_message, re.MULTILINE)
        source = titles[0] if titles else "no source"
        return (
            f"- {subtopic} is a core reliability lever for multi-agent systems [{source}].\n"
            f"- Practitioners pair it with explicit checks rather than trusting model output [{source}]."
        )
    if key == SYNTHESIZER_KEY:
        subtopics = re.findall(r"^- (.+?) is a core reliability lever", user_message, re.MULTILINE)
        bullets = "\n".join(f"- {name}: a core reliability lever." for name in subtopics)
        return (
            "## Overview\nMulti-agent systems stay reliable by constraining tools, "
            "sandboxing permissions and evaluating outputs.\n\n"
            f"## Key Findings\n{bullets}\n\n"
            "## Open Questions\n- How well do these controls hold up with smaller models?\n\n"
            "## Sources\n- [Tool use and function calling in language models]\n"
            "- [Sandboxing and permissions for agent tools]\n- [Evaluating agent workflows]"
        )
    if key == EVALUATOR_KEY:
        return '{"is_complete": true, "feedback": ""}'
    return f"[mock {key}] {user_message[:80]}"


def _field(text: str, label: str) -> str:
    """The line after `label` in a prompt."""
    after = text.split(label, 1)[1] if label in text else ""
    return after.strip().splitlines()[0] if after.strip() else ""


def main(argv: Optional[list[str]] = None) -> int:
    parser = demo_arg_parser(__doc__.splitlines()[0], DEFAULT_TRACE)
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="the research question")
    parser.add_argument("--subtopics", type=int, default=3, help="maximum parallel researchers")
    args = parser.parse_args(argv)

    result = run_demo(
        args.question,
        client_factory=scripted_client_factory(mock_responder) if args.mock else None,
        config_path=args.config,
        trace_path=Path(args.trace),
        max_subtopics=args.subtopics,
        quiet=args.quiet,
    )
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
