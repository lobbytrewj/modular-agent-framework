"""Demo 2: Coding Agent Team - plan, code, test, review, loop, release.

    request
       |
    CodePlannerAgent      signature, algorithm, edge cases -> shared_memory["design_plan"]
       |
       |   EvaluatorOptimizerPipeline
       |   +---------------------------------------------------------------+
       +-> | TeamCoderAgent    implementation -> shared_memory["code_draft"] |
           |      |                                                        |
           |   ReviewGate                                                  |
           |      TestAgent       writes unit tests once, then runs them   |
           |                      on every draft (python_test_runner tool) |
           |      CodeEvaluator   scores the draft, seeing the test results|
           |      failing tests veto approval                              |
           |      |                                                        |
           |   not approved -> feedback goes back to the coder, next round |
           +---------------------------------------------------------------+
       |  approved
    ReleaseAgent          file_write (needs 'write') -> <output-dir>/<function>.py + tests

The trace, with every round's score and every loop-back, is exported to
reports/demo2_trace.json.

    python examples/demo_coding_team.py            # local Qwen model
    python examples/demo_coding_team.py --mock     # offline, scripted replies

Note that the test runner executes model-written code. It does so in a
separate Python process, in a throwaway directory, with a timeout - but that is
isolation, not a security sandbox. Only run this demo on a machine where that
is acceptable.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from demo_common import (  # noqa: E402
    REPO_ROOT,
    REPORTS_DIR,
    ClientFactory,
    DemoResult,
    build_agent,
    demo_arg_parser,
    extract_code,
    finish_run,
    load_config,
    one_line,
    say,
    scripted_client_factory,
    section,
    start_run,
)

from agent_framework.agents import BaseAgent, LLMAgent, execute_agent  # noqa: E402
from agent_framework.core.memory import SharedWorkflowMemory  # noqa: E402
from agent_framework.core.tasks import RunResult, Task  # noqa: E402
from agent_framework.observability import resolve_tracer  # noqa: E402
from agent_framework.orchestration import EvaluatorOptimizerPipeline  # noqa: E402
from agent_framework.orchestration.hierarchical import OBJECT_FIRST, extract_json  # noqa: E402
from agent_framework.tools import Tool, ToolPermission, ToolRegistry  # noqa: E402
from agent_framework.tools.registry import build_default_registry  # noqa: E402

DEFAULT_TRACE = REPORTS_DIR / "demo2_trace.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "workspace" / "demo2"
DEFAULT_REQUEST = (
    "Write a Python function `median(values)` that returns the median of a list "
    "of numbers as a float. For an even count, return the mean of the two middle "
    "values. Do not modify the input list. Raise ValueError for an empty list, "
    "and raise TypeError if any element is not an int or float."
)

PLANNER_KEY = "code_planner"
CODER_KEY = "team_coder"
TEST_KEY = "test_writer"
REVIEWER_KEY = "code_evaluator"
RELEASER_KEY = "code_releaser"

# Blackboard keys. design_plan and code_draft are the planner's and coder's
# output_key in agents.json; the rest are written by the classes below.
PLAN_KEY = "design_plan"
CODE_KEY = "code_draft"
TESTS_KEY = "unit_tests"
REPORT_KEY = "test_report"
FINAL_CODE_KEY = "final_code"
RELEASE_KEY = "release"

TEST_RUNNER_TIMEOUT = 10.0
_REPORT_MARKER = "@@TEST_REPORT@@"


# --- The test runner tool ---

# Prepended to every test file, here and in the released copy. Models reach
# for math.isclose and pytest.raises no matter what the prompt says; a test
# file that dies on a NameError would veto every draft for the tests' fault.
TEST_PRELUDE = """import math

try:
    import pytest
except ImportError:  # the tests can still use try/except instead
    pytest = None
"""

# Runs in the child process. Imports the implementation first, so an error
# can be pinned on the file that raised it. The tests then run under pytest
# when it is installed - models write parametrized tests, fixtures and test
# classes, which only a real runner handles - and under a plain loop over
# test_* functions otherwise. Either way the result is one JSON line.
_HARNESS = """
import json, sys
sys.path.insert(0, {here!r})
result = {{"passed": 0, "failed": 0, "total": 0, "failures": [],
           "collection_error": None, "error_source": None}}

def describe(exc):
    return type(exc).__name__ + ": " + str(exc)

def last_line(longrepr):
    line = str(longrepr).strip().splitlines()[-1]
    return line[1:].strip() if line.startswith("E ") else line  # pytest's "E   " marker

def fail(name, error):
    result["failed"] += 1
    result["failures"].append({{"test": name, "error": error}})

try:
    import solution
except BaseException as exc:
    result["collection_error"] = describe(exc)
    result["error_source"] = "implementation"
else:
    try:
        import pytest
    except ImportError:
        pytest = None

    if pytest is not None:
        class Collector:
            def pytest_collectreport(self, report):
                if report.failed and result["collection_error"] is None:
                    result["collection_error"] = last_line(report.longrepr)
                    result["error_source"] = "tests"

            def pytest_runtest_logreport(self, report):
                name = report.nodeid.split("::", 1)[-1]
                if report.when == "call":
                    result["total"] += 1
                    if report.passed:
                        result["passed"] += 1
                    else:
                        crash = getattr(report.longrepr, "reprcrash", None)
                        fail(name, crash.message if crash else str(report.longrepr)[-300:])
                elif report.failed:  # setup/teardown error, e.g. a missing fixture
                    result["total"] += 1
                    fail(name, last_line(report.longrepr))

        pytest.main(["-q", "-p", "no:cacheprovider", "test_solution.py"], plugins=[Collector()])
        if result["total"] == 0 and result["collection_error"] is None:
            result["collection_error"] = "no tests were collected (name them test_*)"
            result["error_source"] = "tests"
    else:
        try:
            import test_solution as module
        except BaseException as exc:
            result["collection_error"] = describe(exc)
            result["error_source"] = "tests"
        else:
            tests = [(n, f) for n, f in vars(module).items() if n.startswith("test_") and callable(f)]
            if not tests:
                # Bare module-level asserts already ran on import, and passed.
                tests = [("module_level_assertions", lambda: None)]
            for name, test in tests:
                result["total"] += 1
                try:
                    test()
                    result["passed"] += 1
                except BaseException as exc:
                    fail(name, describe(exc))
print({marker!r} + json.dumps(result))
"""


def _no_report(error: str, source: Optional[str] = None) -> dict[str, Any]:
    return {
        "passed": 0, "failed": 0, "total": 0, "failures": [],
        "collection_error": error, "error_source": source,
    }


def run_python_tests(code: str, tests: str, timeout: float = TEST_RUNNER_TIMEOUT) -> dict[str, Any]:
    """Run `tests` against `code` in a child interpreter and report the outcome.

    Failing tests are a successful tool call with failures in the report.
    `error_source` says whether an import-time error came from the
    implementation or from the tests, because the two need different fixes
    from different agents.
    """
    with tempfile.TemporaryDirectory(prefix="demo2_tests_") as workdir:
        here = Path(workdir)
        (here / "solution.py").write_text(code, encoding="utf-8")
        (here / "test_solution.py").write_text(
            f"{TEST_PRELUDE}from solution import *\n\n{tests}\n", encoding="utf-8"
        )
        (here / "harness.py").write_text(
            _HARNESS.format(here=str(here), marker=_REPORT_MARKER), encoding="utf-8"
        )
        try:
            # -I: ignore PYTHONPATH, the working directory and user
            # site-packages, so the child imports only the two files it was
            # given plus the interpreter's own libraries.
            completed = subprocess.run(
                [sys.executable, "-I", "harness.py"],
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            report = _no_report(f"timed out after {timeout:g}s")
        else:
            lines = [l for l in completed.stdout.splitlines() if l.startswith(_REPORT_MARKER)]
            if lines:
                report = json.loads(lines[-1][len(_REPORT_MARKER):])
            else:
                report = _no_report(one_line(completed.stderr[-400:] or "no report produced", 400))

    report["all_passed"] = (
        report["collection_error"] is None and report["failed"] == 0 and report["total"] > 0
    )
    return report


def build_test_runner_tool() -> Tool:
    """`python_test_runner`, gated at WRITE.

    It reads nothing sensitive, but it writes files and executes code - not
    something a read-only agent should be able to do on the strength of a
    model reply.
    """
    return Tool(
        name="python_test_runner",
        description=(
            "Run plain-Python unit tests (test_* functions with asserts) against "
            "an implementation in an isolated child process and report which passed."
        ),
        func=run_python_tests,
        parameters_schema={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "The implementation module."},
                "tests": {"type": "string", "description": "Test functions; the implementation is star-imported."},
                "timeout": {"type": "number", "description": "Seconds before the run is killed."},
            },
            "required": ["code", "tests"],
        },
        required_permission=ToolPermission.WRITE,
    )


def build_team_registry(output_dir: Path) -> ToolRegistry:
    """The built-in tools with the file sandbox at `output_dir`, plus the test runner."""
    registry = build_default_registry(sandbox_root=output_dir)
    registry.register(build_test_runner_tool())
    return registry


def format_test_report(report: dict[str, Any]) -> str:
    error = report.get("collection_error")
    if error and report.get("error_source") == "implementation":
        return (
            f"Importing the implementation raised {error}. Remove every top-level "
            "statement other than imports and definitions (example calls, self-checks, "
            "prints) and fix the error."
        )
    if error and report.get("error_source") == "tests":
        return f"The test file itself failed to load ({error}); this is not a defect in the code."
    if error:
        return f"The tests could not run: {error}"
    lines = [f"{report['passed']}/{report['total']} tests passed."]
    lines += [f"- FAILED {f['test']}: {f['error']}" for f in report.get("failures", [])]
    return "\n".join(lines)


def function_name(code: str) -> Optional[str]:
    match = re.search(r"^def\s+([A-Za-z_]\w*)\s*\(", code or "", re.MULTILINE)
    return match.group(1) if match else None


# --- The team's non-trivial members ---


class TestAgent(LLMAgent):
    """Writes the unit tests once, then runs them against every draft.

    The tests are written from the request, the design plan and the first
    draft (its context_keys in agents.json), then frozen: re-writing them each
    round would let a buggy draft be "fixed" by weakening its tests. Two
    bounded exceptions, each allowed once per run, cover tests that are wrong
    rather than code that is:

      * a test file that cannot even load is rewritten with the error in hand;
      * a DISPUTE: when the same tests fail with the same errors after the
        coder has revised, the failing tests are re-checked against the
        request, and only those that contradict it are corrected. Without
        this, one wrong expectation (say `median([]) == None` when the request
        says raise ValueError) deadlocks the loop against correct code.
    """

    MAX_TEST_REWRITES = 1
    MAX_DISPUTES = 1

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.rewrites = 0
        self.disputes = 0
        self._last_failures: Optional[list] = None

    def _write_tests(
        self, request: str, shared_memory: SharedWorkflowMemory, problem: str = ""
    ) -> Optional[str]:
        """Ask the model for tests and publish them; returns an error or None."""
        fix = (
            f"Your previous test file failed to load with {problem}. Write a corrected one.\n\n"
            if problem
            else ""
        )
        written = super().execute(
            Task(
                description=(
                    f"Request:\n{request}\n\n{fix}"
                    "Write at most 6 unit tests for the function in the draft above. "
                    "Reply with ONLY one ```python code block of test_* functions."
                ),
                assigned_agent=self.name,
            ),
            shared_memory,
        )
        if not written.success:
            return written.error
        shared_memory.set(TESTS_KEY, extract_code(written.output or ""))
        return None

    def _run(self, code: str, shared_memory: SharedWorkflowMemory) -> dict[str, Any]:
        return self.use_tool(
            "python_test_runner",
            shared_memory=shared_memory,
            code=code,
            tests=shared_memory.get(TESTS_KEY),
        )

    def execute(
        self, task: Task, shared_memory: Optional[SharedWorkflowMemory] = None
    ) -> RunResult:
        if shared_memory is None:
            return RunResult(task_id=task.id, success=False, error="TestAgent needs the shared blackboard")

        code = extract_code(shared_memory.get(CODE_KEY) or "")
        if not code:
            return RunResult(task_id=task.id, success=False, error=f"no '{CODE_KEY}' to test")

        if not shared_memory.has(TESTS_KEY):
            error = self._write_tests(task.description, shared_memory)
            if error:
                return RunResult(task_id=task.id, success=False, error=error)

        ran = self._run(code, shared_memory)
        while (
            ran["success"]
            and ran["output"].get("error_source") == "tests"
            and self.rewrites < self.MAX_TEST_REWRITES
        ):
            self.rewrites += 1
            error = self._write_tests(task.description, shared_memory, ran["output"]["collection_error"])
            if error:
                return RunResult(task_id=task.id, success=False, error=error)
            ran = self._run(code, shared_memory)

        if not ran["success"]:
            return RunResult(task_id=task.id, success=False, error=ran["error"])

        failures = ran["output"].get("failures") or None
        if failures and failures == self._last_failures and self.disputes < self.MAX_DISPUTES:
            self.disputes += 1
            error = self._dispute(task.description, code, failures, shared_memory)
            if error:
                return RunResult(task_id=task.id, success=False, error=error)
            ran = self._run(code, shared_memory)
            if not ran["success"]:
                return RunResult(task_id=task.id, success=False, error=ran["error"])
            failures = ran["output"].get("failures") or None
        self._last_failures = failures

        report = ran["output"]
        shared_memory.set(REPORT_KEY, report)
        return RunResult(task_id=task.id, success=True, output=format_test_report(report))

    def _dispute(
        self, request: str, code: str, failures: list, shared_memory: SharedWorkflowMemory
    ) -> Optional[str]:
        """Re-check tests that keep failing unchanged; returns an error or None."""
        failing = "\n".join(f"- {f['test']}: {f['error']}" for f in failures)
        tracer = resolve_tracer(shared_memory)
        if tracer is not None:
            tracer.record_delegation(
                self, self, failing,
                reason="the same tests failed unchanged after a revision; re-checking them against the request",
                workflow="coding_team",
            )
        reviewed = super().execute(
            Task(
                description=(
                    f"Request:\n{request}\n\n"
                    f"Current test file:\n```python\n{shared_memory.get(TESTS_KEY)}\n```\n\n"
                    f"Implementation under test:\n```python\n{code}\n```\n\n"
                    f"These tests still fail with the same errors after the developer revised the code:\n{failing}\n\n"
                    "Check each failing test against the REQUEST, not against the code. If a test "
                    "expects behaviour the request contradicts, correct that test. Keep every test "
                    "that matches the request unchanged, even if it fails. Reply with ONLY one "
                    "```python code block containing the complete test file."
                ),
                assigned_agent=self.name,
            ),
            shared_memory,
        )
        if not reviewed.success:
            return reviewed.error
        revised = extract_code(reviewed.output or "")
        if revised:
            shared_memory.set(TESTS_KEY, revised)
        return None


class ReviewGate(BaseAgent):
    """The evaluator the optimizer loop talks to: tests first, then review.

    Its reply is the reviewer's JSON verdict, unchanged, when every test
    passes. When any test fails it replies with its own failing verdict whose
    feedback leads with the failures - a reviewer that likes the code cannot
    wave through a draft its own tests reject.
    """

    def __init__(self, test_agent: TestAgent, reviewer: BaseAgent, request: str):
        super().__init__(
            name="ReviewGate",
            role="review_gate",
            system_prompt="Runs the unit tests, then the code review; failing tests veto approval.",
        )
        self.test_agent = test_agent
        self.reviewer = reviewer
        self.request = request

    def execute(
        self, task: Task, shared_memory: Optional[SharedWorkflowMemory] = None
    ) -> RunResult:
        tested = execute_agent(
            self.test_agent,
            Task(description=self.request, assigned_agent=self.test_agent.name),
            shared_memory,
        )
        summary = tested.output if tested.success else f"The tests could not be run: {tested.error}"
        report = shared_memory.get(REPORT_KEY) if (tested.success and shared_memory) else None
        tests_ok = bool(report and report.get("all_passed"))

        review = execute_agent(
            self.reviewer,
            Task(
                description=f"Automated unit test results for this draft:\n{summary}\n\n{task.description}",
                assigned_agent=self.reviewer.name,
            ),
            shared_memory,
        )
        if not review.success:
            return RunResult(task_id=task.id, success=False, error=review.error)
        if tests_ok:
            return RunResult(task_id=task.id, success=True, output=review.output)

        passed_ratio = report["passed"] / report["total"] if report and report["total"] else 0.0
        notes = _feedback_of(review.output or "")
        verdict = {
            # Kept well under any sensible pass threshold, but still ordered by
            # how many tests pass, so the loop's "best draft" is meaningful.
            "score": round(4 * passed_ratio, 1),
            "passed": False,
            "feedback": f"Failing unit tests - fix these first:\n{summary}"
            + (f"\n\nReviewer notes: {notes}" if notes else ""),
        }
        return RunResult(task_id=task.id, success=True, output=json.dumps(verdict))


def _feedback_of(reply: str) -> str:
    payload = extract_json(reply, OBJECT_FIRST)
    if isinstance(payload, dict) and isinstance(payload.get("feedback"), str):
        return payload["feedback"].strip()
    return one_line(reply, 400)


class ReleaseAgent(BaseAgent):
    """Writes the approved code and its tests into the sandbox.

    No model call: releasing is mechanical. What makes it this agent's job is
    the grant - of the whole team, only code_releaser holds 'write' plus the
    file_write tool in agents.json.
    """

    def execute(
        self, task: Task, shared_memory: Optional[SharedWorkflowMemory] = None
    ) -> RunResult:
        code = shared_memory.get(FINAL_CODE_KEY) if shared_memory else None
        if not code:
            return RunResult(task_id=task.id, success=False, error=f"no '{FINAL_CODE_KEY}' to release")

        module = function_name(code) or "solution"
        files = [(f"{module}.py", code.rstrip() + "\n")]
        tests = shared_memory.get(TESTS_KEY)
        if tests:
            files.append(
                (f"test_{module}.py", f"{TEST_PRELUDE}from {module} import *\n\n\n{tests.rstrip()}\n")
            )

        written = []
        for path, content in files:
            record = self.use_tool("file_write", shared_memory=shared_memory, path=path, content=content)
            if not record["success"]:
                return RunResult(task_id=task.id, success=False, error=record["error"])
            written.append(record["output"]["path"])

        shared_memory.set(RELEASE_KEY, {"files": written})
        return RunResult(task_id=task.id, success=True, output="Released " + ", ".join(written))


# --- The demo ---


def run_demo(
    request: str = DEFAULT_REQUEST,
    *,
    client_factory: Optional[ClientFactory] = None,
    config_path: Optional[str] = None,
    trace_path: Path = DEFAULT_TRACE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_rounds: int = 3,
    min_pass_score: float = 8.0,
    quiet: bool = False,
) -> DemoResult:
    config = load_config(config_path)
    registry = build_team_registry(Path(output_dir))
    shared_memory, tracer = start_run(
        "coding_team", request=request, mock=client_factory is not None
    )

    def agent(key: str, **overrides: Any):
        return build_agent(
            config, key, client_factory=client_factory, tool_registry=registry, **overrides
        )

    planner = agent(PLANNER_KEY)
    coder = agent(CODER_KEY)
    test_agent = agent(TEST_KEY, agent_cls=TestAgent)
    # The reviewer is handed the draft and the test results in its prompt; the
    # plan is the only artifact it additionally needs.
    reviewer = agent(REVIEWER_KEY, context_keys=[PLAN_KEY])
    releaser = agent(RELEASER_KEY, agent_cls=ReleaseAgent)
    gate = ReviewGate(test_agent, reviewer, request)

    details: dict[str, Any] = {"rounds": [], "released_files": []}

    # --- 1. Plan ---
    section("1. Planner outlines the design", quiet)
    planned = execute_agent(planner, Task(description=request, assigned_agent=planner.name), shared_memory)
    if not planned.success:
        say(f"  planning failed: {planned.error}", quiet)
        path = finish_run(tracer, shared_memory, trace_path, quiet)
        return DemoResult(False, "", shared_memory, tracer, path, details)
    say(planned.output or "", quiet)

    # --- 2-4. Code, test, review, loop ---
    section("2-4. Coder <-> tests + reviewer loop", quiet)
    loop = EvaluatorOptimizerPipeline(coder, gate, min_pass_score=min_pass_score, max_retries=max_rounds)
    looped = loop.run(Task(description=request, assigned_agent=coder.name), shared_memory)

    for step in loop.history:
        details["rounds"].append(
            {"round": step.iteration, "score": step.evaluation.score, "passed": step.evaluation.passed}
        )
        say(
            f"  round {step.iteration}: score {step.evaluation.score:g}/10 "
            f"{'APPROVED' if step.evaluation.passed else 'rejected'}",
            quiet,
        )
        if not step.evaluation.passed:
            say(f"    feedback: {one_line(step.evaluation.feedback, 110)}", quiet)
    say(f"  -> {loop.status_summary()}", quiet)

    final_code = extract_code(looped.output or "")
    success = False
    if loop.passed_threshold and final_code:
        # --- 5. Release ---
        section("5. Release to disk", quiet)
        shared_memory.set(FINAL_CODE_KEY, final_code)
        release_task = Task(description="Publish the approved implementation and its tests.", assigned_agent=releaser.name)
        tracer.record_delegation(gate, releaser, release_task, reason="code approved", workflow="coding_team")
        released = execute_agent(releaser, release_task, shared_memory)
        if released.success:
            details["released_files"] = [
                str(Path(output_dir) / name) for name in shared_memory.get(RELEASE_KEY)["files"]
            ]
            for name in details["released_files"]:
                say(f"  wrote {name}", quiet)
        else:
            say(f"  release failed: {released.error}", quiet)
        success = released.success
    else:
        say("\nNot released: the code never passed its tests and review.", quiet)

    section("Final code", quiet)
    say(final_code, quiet)
    path = finish_run(tracer, shared_memory, trace_path, quiet)
    return DemoResult(success, final_code, shared_memory, tracer, path, details)


# --- Offline script for --mock ---
#
# The coder's first draft forgets the empty-list and type checks, so the
# frozen tests fail, the gate rejects it, and the loop has to go round once
# before the fixed draft is approved - the whole feedback path, offline.

MOCK_PLAN = """Signature: median(values: list) -> float
Algorithm: validate, sort a copy, pick the middle element or average the two middle ones.
Edge cases:
- empty list -> ValueError
- non-numeric element (including bool) -> TypeError
- even count -> mean of the two middle values
- input list must not be mutated"""

MOCK_FIRST_DRAFT = """```python
def median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2
```"""

MOCK_FIXED_DRAFT = """```python
def median(values):
    \"\"\"Return the median of a list of numbers as a float.\"\"\"
    if not values:
        raise ValueError("median() of an empty list")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"median() needs numbers, got {type(value).__name__}")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2
```"""

MOCK_TESTS = """```python
def test_odd_count():
    assert median([3, 1, 2]) == 2.0


def test_even_count():
    assert median([4, 1, 3, 2]) == 2.5


def test_does_not_mutate_input():
    data = [3, 1, 2]
    median(data)
    assert data == [3, 1, 2]


def test_empty_list_raises_value_error():
    try:
        median([])
    except ValueError:
        return
    raise AssertionError("expected ValueError for an empty list")


def test_bool_raises_type_error():
    try:
        median([1, True, 3])
    except TypeError:
        return
    raise AssertionError("expected TypeError for a bool element")
```"""


def mock_responder(key: str, system_prompt: str, user_message: str) -> str:
    if key == PLANNER_KEY:
        return MOCK_PLAN
    if key == CODER_KEY:
        return MOCK_FIXED_DRAFT if "Your previous attempt" in user_message else MOCK_FIRST_DRAFT
    if key == TEST_KEY:
        return MOCK_TESTS
    if key == REVIEWER_KEY:
        counts = re.search(r"(\d+)/(\d+) tests passed", user_message)
        if counts and counts.group(1) == counts.group(2):
            return '{"score": 9, "passed": true, "feedback": ""}'
        return (
            '{"score": 4, "passed": false, "feedback": "Validate the input before '
            'sorting: raise ValueError on an empty list and TypeError on non-numbers."}'
        )
    return f"[mock {key}]"


def main(argv: Optional[list[str]] = None) -> int:
    parser = demo_arg_parser(__doc__.splitlines()[0], DEFAULT_TRACE)
    parser.add_argument("--request", default=DEFAULT_REQUEST, help="the function to build")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="sandbox the approved files are written into (default: workspace/demo2)",
    )
    parser.add_argument("--rounds", type=int, default=3, help="maximum code/review rounds")
    args = parser.parse_args(argv)

    result = run_demo(
        args.request,
        client_factory=scripted_client_factory(mock_responder) if args.mock else None,
        config_path=args.config,
        trace_path=Path(args.trace),
        output_dir=Path(args.output_dir),
        max_rounds=args.rounds,
        quiet=args.quiet,
    )
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
