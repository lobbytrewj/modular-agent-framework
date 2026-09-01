from __future__ import annotations

from agent_framework.agents import LLMAgent, MockAgent
from agent_framework.core.memory import (
    AgentMemory,
    Message,
    SharedWorkflowMemory,
)
from agent_framework.core.tasks import Task
from agent_framework.orchestration.parallel import ParallelOrchestrator
from agent_framework.orchestration.sequential import SequentialOrchestrator


class RecordingClient:
    def __init__(self, reply: str = "drafted output"):
        self.reply = reply
        self.calls: list[dict] = []

    def complete(self, system_prompt: str, user_message: str, history=None) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_message": user_message,
                "history": list(history or []),
            }
        )
        return self.reply


# --- Agent-local memory ---


def test_agent_memory_records_and_replays_its_own_turns() -> None:
    memory = AgentMemory()
    memory.add_user_message("first question")
    memory.add_assistant_message("first answer")
    memory.add_user_message("second question")
    memory.add_assistant_message("second answer")

    assert len(memory) == 4
    assert all(isinstance(message, Message) for message in memory.messages)
    assert [m.role for m in memory.messages] == ["user", "assistant", "user", "assistant"]

    # last_k counts turns, not exchanges.
    assert len(memory.recent(2)) == 2
    context = memory.get_context_string(last_k=2)
    assert "second question" in context and "first question" not in context
    assert memory.get_context_string().count("question") == 2

    # The chat-message form is what actually reaches the model.
    assert memory.as_chat_messages(1) == [{"role": "assistant", "content": "second answer"}]

    memory.clear()
    assert memory.messages == []
    assert memory.get_context_string() == ""

    print("[agent memory] records, windows and clears its own turns")


def test_agent_memory_is_private_to_one_agent() -> None:
    # Two agents, two memories: what one remembers never appears in the other.
    first = MockAgent("First", "role_a", "prompt", memory=AgentMemory())
    second = MockAgent("Second", "role_b", "prompt", memory=AgentMemory())

    first.execute(Task(description="only first hears this", assigned_agent="First"))

    assert len(first.memory) == 2
    assert len(second.memory) == 0

    print("[agent memory] stays private to the agent that owns it")


def test_max_messages_bounds_growth() -> None:
    memory = AgentMemory(max_messages=2)
    for index in range(4):
        memory.add_user_message(f"turn {index}")

    assert len(memory) == 2
    # The oldest turns are the ones dropped.
    assert [m.content for m in memory.messages] == ["turn 2", "turn 3"]

    print("[agent memory] max_messages drops the oldest turns first")


# --- Shared workflow memory ---


def test_scratchpad_stores_and_formats_artifacts() -> None:
    shared = SharedWorkflowMemory()

    shared.set("database_schema", "CREATE TABLE users (id INT PRIMARY KEY);")
    shared.set("api_code", "def get_user(id): ...")
    shared.set("config", {"port": 8080, "debug": False})

    assert shared.has("api_code")
    assert not shared.has("frontend")
    assert shared.get("frontend", "missing") == "missing"
    assert shared.keys() == ["database_schema", "api_code", "config"]
    assert "api_code" in shared

    # Everything, in the order it was produced.
    context = shared.to_prompt_context()
    assert "## Shared workflow context" in context
    assert "### database_schema" in context and "### api_code" in context
    # Non-string values render as JSON, not as a Python repr.
    assert '"port": 8080' in context

    # A narrowed view shows only what was asked for, and silently skips a name
    # nothing has been published under yet.
    focused = shared.to_prompt_context(["api_code", "never_written"])
    assert "### api_code" in focused
    assert "database_schema" not in focused
    assert "never_written" not in focused

    # Nothing to show means no header at all, so callers can guard on "".
    assert SharedWorkflowMemory().to_prompt_context() == ""

    # Long artifacts are cut short, and say so.
    shared.set("huge", "x" * 5000)
    assert "[truncated, 5000 chars total]" in shared.to_prompt_context(["huge"])

    shared.clear()
    assert shared.artifacts == {} and shared.execution_log == []

    print("[shared memory] stores artifacts and renders them as prompt context")


# --- Artifacts crossing agents, steps and workflows ---


def test_artifacts_flow_across_steps_and_workflows() -> None:
    shared = SharedWorkflowMemory()
    # A deliverable produced before this run even starts - the case where one
    # workflow hands work to the next.
    shared.set("database_schema", "CREATE TABLE users (id INT PRIMARY KEY);")

    pipeline = SequentialOrchestrator(
        [
            MockAgent("Backend", "backend", "prompt"),
            MockAgent("Reviewer", "review", "prompt"),
        ]
    )
    task = Task(description="Build the user service", assigned_agent="Backend")
    result = pipeline.run(task, shared_memory=shared)

    assert result.success is True
    # MockAgent reports what it could see, which proves the blackboard reached
    # it rather than merely being stored next to it.
    assert all("database_schema" in step.output for step in pipeline.step_results)

    # One audit entry per executed step, in order, naming the agent.
    assert len(shared.execution_log) == 2
    assert [record["agent"] for record in shared.execution_log] == ["Backend", "Reviewer"]
    assert [record["step"] for record in shared.execution_log] == [1, 2]
    assert all(record["success"] for record in shared.execution_log)

    # A second, structurally different workflow writes onto the SAME board, and
    # the log keeps counting where the first one left off.
    fan_out = ParallelOrchestrator(
        [MockAgent("WorkerA", "a", "prompt"), MockAgent("WorkerB", "b", "prompt")],
        MockAgent("Synth", "synth", "prompt"),
    )
    shared.set("api_code", "def get_user(id): ...")
    fan_out.run(Task(description="Review the service", assigned_agent="WorkerA"), shared)

    assert len(shared.execution_log) == 5
    assert [record["step"] for record in shared.execution_log] == [1, 2, 3, 4, 5]
    assert {"WorkerA", "WorkerB", "Synth"} <= {r["agent"] for r in shared.execution_log}
    # The workers ran concurrently, so the lock is what keeps the log intact.
    assert len(shared.log_for("Synth")) == 1
    # Both workflows' artifacts sit on one board.
    assert shared.keys() == ["database_schema", "api_code"]

    print("[shared memory] artifacts and the audit log survive across workflows")


def test_pipelines_still_run_without_shared_memory() -> None:
    # The whole feature is opt-in: omit it and nothing changes.
    pipeline = SequentialOrchestrator([MockAgent("Solo", "solo", "prompt")])
    result = pipeline.run(Task(description="Do the thing", assigned_agent="Solo"))

    assert result.success is True
    assert "shared artifacts visible" not in result.output

    print("[compatibility] run() without shared memory behaves exactly as before")


# --- The LLM agent's use of both kinds of memory ---


def test_llm_agent_reads_shared_context_and_publishes_its_output() -> None:
    client = RecordingClient("class UserService: ...")
    agent = LLMAgent(
        name="Backend",
        role="backend",
        system_prompt="You write services.",
        llm_client=client,
        context_keys=["database_schema"],
        output_key="api_code",
    )

    shared = SharedWorkflowMemory()
    shared.set("database_schema", "CREATE TABLE users (id INT PRIMARY KEY);")
    shared.set("unrelated_notes", "ignore me")

    result = agent.execute(
        Task(description="Write the user service", assigned_agent="Backend"),
        shared_memory=shared,
    )

    prompt = client.calls[-1]["user_message"]
    # Only the artifact this agent asked for, and the instruction last.
    assert "CREATE TABLE users" in prompt
    assert "ignore me" not in prompt
    assert prompt.rstrip().endswith("Write the user service")

    # Its answer is published under the agreed name for later steps to read.
    assert result.success is True
    assert shared.get("api_code") == "class UserService: ..."

    print("[llm agent] reads named artifacts and publishes its own")


def test_llm_agent_replays_its_own_history() -> None:
    client = RecordingClient("second reply")
    agent = LLMAgent(
        name="Chatty",
        role="assistant",
        system_prompt="You answer questions.",
        llm_client=client,
        memory=AgentMemory(),
    )

    agent.execute(Task(description="first question", assigned_agent="Chatty"))
    # Turn one had no history to replay, and is recorded only after it succeeds.
    assert client.calls[0]["history"] == []
    assert len(agent.memory) == 2

    agent.execute(Task(description="second question", assigned_agent="Chatty"))
    replayed = client.calls[1]["history"]
    assert [m["role"] for m in replayed] == ["user", "assistant"]
    assert replayed[0]["content"] == "first question"
    # The prompt itself is not duplicated into the history it was built from.
    assert "second question" not in [m["content"] for m in replayed]
    assert len(agent.memory) == 4

    # A stateless agent sends no history at all.
    stateless = LLMAgent("Blank", "blank", "prompt", llm_client=RecordingClient())
    stateless.execute(Task(description="q", assigned_agent="Blank"))
    assert stateless.llm_client.calls[0]["history"] == []

    print("[llm agent] replays its own turns and only its own")


def test_history_window_bounds_what_is_replayed() -> None:
    client = RecordingClient("reply")
    agent = LLMAgent(
        name="Windowed",
        role="assistant",
        system_prompt="prompt",
        llm_client=client,
        memory=AgentMemory(),
        history_window=2,
    )

    for index in range(3):
        agent.execute(Task(description=f"question {index}", assigned_agent="Windowed"))

    assert len(agent.memory) == 6  # everything is remembered ...
    assert len(client.calls[-1]["history"]) == 2  # ... but only the tail is replayed

    print("[llm agent] history_window caps how much of memory reaches the prompt")


TESTS = (
    test_agent_memory_records_and_replays_its_own_turns,
    test_agent_memory_is_private_to_one_agent,
    test_max_messages_bounds_growth,
    test_scratchpad_stores_and_formats_artifacts,
    test_artifacts_flow_across_steps_and_workflows,
    test_pipelines_still_run_without_shared_memory,
    test_llm_agent_reads_shared_context_and_publishes_its_output,
    test_llm_agent_replays_its_own_history,
    test_history_window_bounds_what_is_replayed,
)


if __name__ == "__main__":
    print("---- Memory and Shared State Trace ----")
    for test in TESTS:
        test()
    print("\nAll assertions passed.")
