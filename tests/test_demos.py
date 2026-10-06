"""Offline tests for Goal 13: the three end-to-end demo applications.

Each demo runs against scripted clients through its real agents, config
grants, tools, orchestrators and tracer; only the model is replaced. A guard
fixture fails any test that tries to load a real model, so the suite stays
fast and runs on a machine without the weights.

    python -m pytest tests/test_demos.py -q
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EXAMPLES))

import demo_coding_team as coding  # noqa: E402
import demo_research_swarm as research  # noqa: E402
import demo_routing_assistant as routing  # noqa: E402
from demo_common import build_agent, load_config, scripted_client_factory  # noqa: E402

from agent_framework.core import llm  # noqa: E402
from agent_framework.observability import EventType, WorkflowTrace  # noqa: E402
from agent_framework.orchestration import sequential  # noqa: E402
from agent_framework.tools import ToolStatus  # noqa: E402


@pytest.fixture(autouse=True)
def no_model_loads(monkeypatch):
    """Every agent here must run on an injected client; loading Qwen is a bug."""

    def refuse(*args, **kwargs):
        raise AssertionError("an offline demo test tried to load a real model")

    monkeypatch.setattr(sequential, "build_llm_client", refuse)
    monkeypatch.setattr(llm.LLMClient, "__init__", refuse)


def load_trace(path: Path) -> WorkflowTrace:
    assert Path(path).is_file(), f"no trace exported at {path}"
    return WorkflowTrace.load_json(path)


def agent_calls(trace: WorkflowTrace) -> list[str]:
    return [event.agent_name for event in trace.events_of(EventType.AGENT_CALL)]


# --- Config ---


def test_demo_agents_are_defined_in_config_with_explicit_grants() -> None:
    config = load_config()
    expected = {
        # key: (tools, permissions)
        "research_planner": ([], []),
        "researcher": (["search", "web_search"], ["read_only"]),
        "research_synthesizer": ([], []),
        "research_evaluator": ([], []),
        "code_planner": ([], []),
        "team_coder": ([], []),
        "test_writer": (["python_test_runner"], ["read_only", "write"]),
        "code_evaluator": ([], []),
        "code_releaser": (["file_read", "file_list", "file_write"], ["read_only", "write"]),
        "request_router": ([], []),
        "math": (["calculator"], ["read_only"]),
        "analyst": (["calculator", "file_read", "web_search"], ["read_only"]),
        "direct_answer": ([], []),
        "fallback": ([], []),
    }
    for key, (tools, permissions) in expected.items():
        assert config[key]["tools"] == tools, key
        assert config[key]["permissions"] == permissions, key

    # Blackboard wiring the demos rely on lives in config, not in code.
    assert config["research_synthesizer"]["output_key"] == research.REPORT_KEY
    assert config["code_planner"]["output_key"] == coding.PLAN_KEY
    assert config["team_coder"]["output_key"] == coding.CODE_KEY


# --- Demo 1: research swarm ---


def test_research_swarm_runs_offline_and_traces_every_step(tmp_path: Path) -> None:
    result = research.run_demo(
        client_factory=scripted_client_factory(research.mock_responder),
        trace_path=tmp_path / "demo1_trace.json",
        quiet=True,
    )
    assert result.success, result.details

    memory = result.shared_memory
    report = memory.get(research.REPORT_KEY)
    assert report == result.output
    for heading in ("## Overview", "## Key Findings", "## Open Questions", "## Sources"):
        assert heading in report

    subtopics = memory.get("subtopics")
    assert len(subtopics) == 3
    for position, subtopic in enumerate(subtopics, start=1):
        finding = memory.get(f"{research.FINDINGS_PREFIX}{position}")
        assert finding["subtopic"] == subtopic
        assert finding["sources"], "the offline search corpus should match every subtopic"

    trace = load_trace(result.trace_path)
    assert trace.status == "completed"
    calls = agent_calls(trace)
    assert calls[0] == "ResearchPlannerAgent"
    assert sorted(name for name in calls if name.startswith("ResearchAgent-")) == [
        "ResearchAgent-1", "ResearchAgent-2", "ResearchAgent-3",
    ]
    assert calls[-2:] == ["ResearchSynthesizerAgent", "ResearchEvaluatorAgent"]

    searches = trace.events_of(EventType.TOOL_CALL)
    assert len(searches) == 3
    assert all(event.data["tool"] == "search" and event.data["status"] == "ok" for event in searches)
    assert {event.data["arguments"]["query"] for event in searches} == set(subtopics)

    assert [event.data["to"] for event in trace.events_of(EventType.DELEGATION)] == [
        "ResearchAgent-1", "ResearchAgent-2", "ResearchAgent-3",
    ]
    assert [event.data["is_complete"] for event in trace.events_of(EventType.VERIFICATION)] == [True]
    assert trace.events_of(EventType.STATE_SNAPSHOT)[-1].data["label"] == "final"


def test_research_swarm_revises_the_report_when_the_evaluator_rejects_it(tmp_path: Path) -> None:
    verdicts = iter([
        '{"is_complete": false, "feedback": "Add a sentence on retrieval."}',
        '{"is_complete": true, "feedback": ""}',
    ])

    def responder(key: str, system_prompt: str, user_message: str) -> str:
        if key == research.EVALUATOR_KEY:
            return next(verdicts)
        reply = research.mock_responder(key, system_prompt, user_message)
        if key == research.SYNTHESIZER_KEY and "A reviewer found these problems" in user_message:
            reply += "\n\nRetrieval grounds answers in cited sources."
        return reply

    factory = scripted_client_factory(responder)
    result = research.run_demo(
        client_factory=factory, trace_path=tmp_path / "trace.json", quiet=True
    )
    assert result.success
    assert "Retrieval grounds" in result.shared_memory.get(research.REPORT_KEY)
    assert result.shared_memory.get("research_evaluation")["attempts"] == 2

    # The revision prompt carries the critique AND the findings, read back
    # off the blackboard.
    revision_prompt = factory.clients[research.SYNTHESIZER_KEY].calls[-1]["user_message"]
    assert "Add a sentence on retrieval." in revision_prompt
    assert revision_prompt.count("is a core reliability lever") >= 3

    trace = load_trace(result.trace_path)
    assert [event.data["is_complete"] for event in trace.events_of(EventType.VERIFICATION)] == [False, True]
    loop_back = trace.events_of(EventType.DELEGATION)[-1]
    assert (loop_back.data["from"], loop_back.data["to"]) == (
        "ResearchEvaluatorAgent", "ResearchSynthesizerAgent",
    )


def test_research_swarm_degrades_to_one_worker_when_the_plan_is_unusable(tmp_path: Path) -> None:
    def responder(key: str, system_prompt: str, user_message: str) -> str:
        if key == research.PLANNER_KEY:
            return "I would research this carefully."
        return research.mock_responder(key, system_prompt, user_message)

    result = research.run_demo(
        client_factory=scripted_client_factory(responder),
        trace_path=tmp_path / "trace.json",
        quiet=True,
    )
    assert result.success
    assert result.details["subtopics"] == [research.DEFAULT_QUESTION]
    assert result.details["worker_count"] == 1


def test_parse_subtopics_reads_json_lists_and_caps_the_count() -> None:
    assert research.parse_subtopics('Sure: ["a", "b", "a", "c", "d"]', 3) == ["a", "b", "c"]
    assert research.parse_subtopics('[{"subtopic": "x"}, {"topic": "y"}]', 3) == ["x", "y"]
    assert research.parse_subtopics("1. first\n2) second\n- third", 5) == ["first", "second", "third"]
    assert research.parse_subtopics("no list here", 3) == []


# --- Demo 2: coding team ---


def test_coding_team_loops_on_failing_tests_then_releases_verified_code(tmp_path: Path) -> None:
    out = tmp_path / "workspace"
    factory = scripted_client_factory(coding.mock_responder)
    result = coding.run_demo(
        client_factory=factory,
        trace_path=tmp_path / "demo2_trace.json",
        output_dir=out,
        quiet=True,
    )
    assert result.success, result.details

    # Round 1's draft misses the edge cases, the frozen tests catch it, and
    # the coder fixes it on round 2.
    assert [round_["passed"] for round_ in result.details["rounds"]] == [False, True]
    assert len(factory.clients[coding.TEST_KEY].calls) == 1, "tests are written once, then frozen"
    revision_prompt = factory.clients[coding.CODER_KEY].calls[1]["user_message"]
    assert "Failing unit tests" in revision_prompt
    assert "test_empty_list_raises_value_error" in revision_prompt
    assert result.shared_memory.get(coding.REPORT_KEY)["all_passed"] is True

    # The released code is the approved code, and its released tests pass
    # against it in a fresh interpreter.
    released = out / "median.py"
    assert released.read_text(encoding="utf-8").strip() == result.output
    assert "raise ValueError" in result.output
    assert (out / "test_median.py").is_file()
    completed = subprocess.run(
        [sys.executable, "-c",
         "import test_median as t; [getattr(t, n)() for n in dir(t) if n.startswith('test_')]"],
        cwd=out, capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stderr

    trace = load_trace(result.trace_path)
    assert trace.status == "completed"
    verifications = trace.events_of(EventType.VERIFICATION)
    assert [event.data["is_complete"] for event in verifications] == [False, True]
    assert verifications[0].data["score"] < 8 <= verifications[1].data["score"]

    delegations = trace.events_of(EventType.DELEGATION)
    assert [(event.data["from"], event.data["to"]) for event in delegations] == [
        ("ReviewGate", "TeamCoderAgent"),
        ("ReviewGate", "ReleaseAgent"),
    ]

    tools = [(event.agent_name, event.data["tool"], event.data["status"])
             for event in trace.events_of(EventType.TOOL_CALL)]
    assert tools == [
        ("TestAgent", "python_test_runner", "ok"),
        ("TestAgent", "python_test_runner", "ok"),
        ("ReleaseAgent", "file_write", "ok"),
        ("ReleaseAgent", "file_write", "ok"),
    ]
    writes = [event for event in trace.events_of(EventType.TOOL_CALL) if event.data["tool"] == "file_write"]
    assert all(event.data["required_permission"] == "write" for event in writes)


def test_coding_team_never_releases_code_that_fails_its_tests(tmp_path: Path) -> None:
    """Failing tests veto approval even when the reviewer is happy."""

    def responder(key: str, system_prompt: str, user_message: str) -> str:
        if key == coding.CODER_KEY:
            return coding.MOCK_FIRST_DRAFT  # never fixed
        if key == coding.REVIEWER_KEY:
            return '{"score": 10, "passed": true, "feedback": ""}'
        return coding.mock_responder(key, system_prompt, user_message)

    out = tmp_path / "workspace"
    result = coding.run_demo(
        client_factory=scripted_client_factory(responder),
        trace_path=tmp_path / "trace.json",
        output_dir=out,
        max_rounds=2,
        quiet=True,
    )
    assert not result.success
    assert [round_["passed"] for round_ in result.details["rounds"]] == [False, False]
    assert result.details["released_files"] == []
    assert not (out / "median.py").exists()

    trace = load_trace(result.trace_path)
    assert "file_write" not in {event.data["tool"] for event in trace.events_of(EventType.TOOL_CALL)}
    assert "ReleaseAgent" not in agent_calls(trace)


def test_only_the_releaser_may_write_files(tmp_path: Path) -> None:
    config = load_config()
    registry = coding.build_team_registry(tmp_path)
    factory = scripted_client_factory(coding.mock_responder)

    coder = build_agent(config, coding.CODER_KEY, client_factory=factory, tool_registry=registry)
    tester = build_agent(
        config, coding.TEST_KEY, agent_cls=coding.TestAgent, client_factory=factory, tool_registry=registry
    )
    releaser = build_agent(config, coding.RELEASER_KEY, agent_cls=coding.ReleaseAgent, tool_registry=registry)

    assert coder.tool_names() == []
    assert tester.tool_names() == ["python_test_runner"]
    assert "file_write" in releaser.tool_names()
    # The coder holds no tools at all, so it is not even given a registry; the
    # tester has one but was never assigned file_write.
    refusals = {
        coder.name: ToolStatus.NOT_FOUND.value,
        tester.name: ToolStatus.UNAUTHORIZED.value,
    }
    for agent in (coder, tester):
        refused = agent.use_tool("file_write", path="sneaky.py", content="x = 1")
        assert refused["status"] == refusals[agent.name]
    assert not (tmp_path / "sneaky.py").exists()

    # Running tests executes code, so the runner itself needs 'write'.
    assert not registry.get("python_test_runner").is_allowed_for(["read_only"])

    # Grants come from config only.
    with pytest.raises(ValueError):
        build_agent(config, coding.CODER_KEY, client_factory=factory, permissions=["write"])


def test_python_test_runner_reports_failures_crashes_and_timeouts() -> None:
    passing = coding.run_python_tests("def f(x):\n    return x * 2\n", "def test_f():\n    assert f(2) == 4\n")
    assert passing["all_passed"] and passing["total"] == 1

    failing = coding.run_python_tests("def f(x):\n    return x\n", "def test_f():\n    assert f(2) == 4\n")
    assert not failing["all_passed"] and failing["failures"][0]["test"] == "test_f"

    # Import-time errors are pinned on whichever file raised them.
    broken_code = coding.run_python_tests("def f(:\n", "def test_f():\n    pass\n")
    assert broken_code["collection_error"].startswith("SyntaxError")
    assert broken_code["error_source"] == "implementation"
    self_check = coding.run_python_tests("def f(x):\n    return x\n\nassert f(1) == 2\n", "")
    assert self_check["error_source"] == "implementation"
    assert "Importing the implementation raised" in coding.format_test_report(self_check)

    broken_tests = coding.run_python_tests("def f(x):\n    return x\n", "def test_f(:\n")
    assert broken_tests["error_source"] == "tests"
    assert "not a defect in the code" in coding.format_test_report(broken_tests)

    # math and pytest are pre-imported, because models use them regardless.
    idiomatic = coding.run_python_tests(
        "def f(x):\n    if x < 0:\n        raise ValueError\n    return x / 3\n",
        "def test_raises():\n    with pytest.raises(ValueError):\n        f(-1)\n\n"
        "def test_close():\n    assert math.isclose(f(1), 1 / 3)\n",
    )
    assert idiomatic["all_passed"], idiomatic

    # Parametrized tests run case by case under pytest (seen live with Qwen).
    parametrized = coding.run_python_tests(
        "def double(x):\n    return x * 2\n",
        "@pytest.mark.parametrize('x, expected', [(1, 2), (2, 4), (3, 7)])\n"
        "def test_double(x, expected):\n    assert double(x) == expected\n",
    )
    assert (parametrized["passed"], parametrized["total"]) == (2, 3)
    assert parametrized["failures"][0]["test"] == "test_double[3-7]"

    hung = coding.run_python_tests("while True:\n    pass\n", "", timeout=1)
    assert "timed out" in hung["collection_error"] and not hung["all_passed"]


def test_coding_team_rewrites_tests_once_when_the_test_file_is_broken(tmp_path: Path) -> None:
    replies = iter(["```python\ndef test_broken(:\n    pass\n```", coding.MOCK_TESTS])

    def responder(key: str, system_prompt: str, user_message: str) -> str:
        if key == coding.TEST_KEY:
            return next(replies)
        return coding.mock_responder(key, system_prompt, user_message)

    factory = scripted_client_factory(responder)
    result = coding.run_demo(
        client_factory=factory, trace_path=tmp_path / "trace.json", output_dir=tmp_path / "out", quiet=True
    )
    assert result.success
    rewrite_prompt = factory.clients[coding.TEST_KEY].calls[1]["user_message"]
    assert "failed to load with SyntaxError" in rewrite_prompt
    # Round 1 still fails on the buggy draft - against the rewritten tests.
    assert [round_["passed"] for round_ in result.details["rounds"]] == [False, True]


def test_coding_team_disputes_a_test_that_contradicts_the_request(tmp_path: Path) -> None:
    """The failure seen live with Qwen: correct code, one wrong test, a deadlock.

    The code raises ValueError on [] as the request says; the test expects
    None. The coder cannot fix that, so the same failure repeats - and the
    repeat triggers one re-check of the failing test against the request.
    """
    wrong_tests = coding.MOCK_TESTS.replace(
        "def test_empty_list_raises_value_error():\n    try:\n        median([])\n"
        "    except ValueError:\n        return\n    raise AssertionError(\"expected ValueError for an empty list\")",
        "def test_empty_list_raises_value_error():\n    assert median([]) is None",
    )
    assert wrong_tests != coding.MOCK_TESTS

    def responder(key: str, system_prompt: str, user_message: str) -> str:
        if key == coding.CODER_KEY:
            return coding.MOCK_FIXED_DRAFT  # correct from the start
        if key == coding.TEST_KEY:
            disputed = "still fail with the same errors" in user_message
            return coding.MOCK_TESTS if disputed else wrong_tests
        return coding.mock_responder(key, system_prompt, user_message)

    factory = scripted_client_factory(responder)
    result = coding.run_demo(
        client_factory=factory, trace_path=tmp_path / "trace.json", output_dir=tmp_path / "out", quiet=True
    )
    assert result.success
    assert [round_["passed"] for round_ in result.details["rounds"]] == [False, True]
    dispute_prompt = factory.clients[coding.TEST_KEY].calls[-1]["user_message"]
    assert "test_empty_list_raises_value_error: ValueError" in dispute_prompt
    assert "against the REQUEST" in dispute_prompt

    reasons = [event.data["reason"] for event in load_trace(result.trace_path).events_of(EventType.DELEGATION)]
    assert any("re-checking them against the request" in reason for reason in reasons)


def test_coding_team_tells_the_coder_when_its_module_raises_on_import(tmp_path: Path) -> None:
    self_checking = coding.MOCK_FIXED_DRAFT.replace(
        "\n```", "\n\nassert median([1, 2]) == 2  # a wrong self-check\n```"
    )

    def responder(key: str, system_prompt: str, user_message: str) -> str:
        if key == coding.CODER_KEY and "Your previous attempt" not in user_message:
            return self_checking
        return coding.mock_responder(key, system_prompt, user_message)

    factory = scripted_client_factory(responder)
    result = coding.run_demo(
        client_factory=factory, trace_path=tmp_path / "trace.json", output_dir=tmp_path / "out", quiet=True
    )
    assert result.success
    revision_prompt = factory.clients[coding.CODER_KEY].calls[1]["user_message"]
    assert "Importing the implementation raised AssertionError" in revision_prompt
    assert "Remove every top-level statement" in revision_prompt


# --- Demo 3: routing assistant ---


def test_routing_assistant_sends_each_query_to_its_sub_workflow(tmp_path: Path) -> None:
    result = routing.run_demo(
        client_factory=scripted_client_factory(routing.mock_responder),
        trace_path=tmp_path / "demo3_trace.json",
        quiet=True,
    )
    assert result.success
    answers = result.details["answers"]
    assert [answer["route"] for answer in answers] == ["code", "math", "qa"]
    assert all(answer["reasoning"].startswith("RouterAgent: ") for answer in answers)
    assert "def is_palindrome" in answers[0]["output"]
    assert "48.6" in answers[1]["output"]
    assert result.shared_memory.get("calculations")[0]["result"] == pytest.approx(48.6)

    trace = load_trace(result.trace_path)
    assert trace.status == "completed"
    decisions = trace.events_of(EventType.ROUTE_DECISION)
    assert [event.data["route"] for event in decisions] == ["code", "math", "qa"]
    assert all("RouterAgent" in event.data["reasoning"] for event in decisions)
    # Sub-workflows get a delegation event; a single-agent route does not need one.
    assert [event.data["to"] for event in trace.events_of(EventType.DELEGATION)] == [
        "SequentialOrchestrator", "MathAnalysisPipeline",
    ]
    assert agent_calls(trace) == [
        "RouterAgent", "CodePlannerAgent", "TeamCoderAgent",
        "RouterAgent", "MathAgent", "AnalystAgent",
        "RouterAgent", "DirectAnswerAgent",
    ]
    (calculation,) = trace.events_of(EventType.TOOL_CALL)
    assert (calculation.agent_name, calculation.data["tool"], calculation.data["status"]) == (
        "MathAgent", "calculator", "ok",
    )


def test_routing_assistant_falls_back_to_keywords_when_the_router_reply_is_unusable(tmp_path: Path) -> None:
    replies = iter(["Sure! That sounds like code to me.", '{"route": "poetry", "reasoning": "?"}'])

    def responder(key: str, system_prompt: str, user_message: str) -> str:
        if key == routing.ROUTER_KEY:
            return next(replies)
        return routing.mock_responder(key, system_prompt, user_message)

    result = routing.run_demo(
        routing.DEFAULT_QUERIES[:2],
        client_factory=scripted_client_factory(responder),
        trace_path=tmp_path / "trace.json",
        quiet=True,
    )
    assert result.success
    code, math = result.details["answers"]
    assert code["route"] == "code"
    assert code["reasoning"] == "router reply was not JSON; keyword rules: keyword 'python' matched"
    assert math["route"] == "math"
    assert math["reasoning"].startswith("router named unknown route 'poetry'; keyword rules:")


def test_routing_assistant_sends_unmatched_queries_to_the_fallback(tmp_path: Path) -> None:
    result = routing.run_demo(
        ["Tell me a haiku about autumn leaves"],
        client_factory=scripted_client_factory(routing.mock_responder),
        trace_path=tmp_path / "trace.json",
        quiet=True,
    )
    assert result.success
    assert result.details["answers"][0]["route"] == "fallback"
    assert agent_calls(load_trace(result.trace_path)) == ["RouterAgent", "FallbackAgent"]


# --- The scripts themselves ---


@pytest.mark.parametrize(
    "script, extra_args",
    [
        ("demo_research_swarm.py", []),
        ("demo_coding_team.py", ["--output-dir", "{tmp}/out"]),
        ("demo_routing_assistant.py", []),
    ],
)
def test_demo_scripts_run_from_the_command_line_in_mock_mode(
    tmp_path: Path, script: str, extra_args: list[str]
) -> None:
    trace = tmp_path / "trace.json"
    completed = subprocess.run(
        [sys.executable, str(EXAMPLES / script), "--mock", "--trace", str(trace),
         *[arg.format(tmp=tmp_path) for arg in extra_args]],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "RUN SUMMARY" in completed.stdout
    assert str(trace.resolve()) in completed.stdout
    assert load_trace(trace).status == "completed"
