# End-to-end demos

Three runnable applications built on `agent_framework`. Each one uses a different orchestration pattern:

| Demo | Pattern | Script | Trace |
|---|---|---|---|
| 1. Research Swarm | Orchestrator-worker: parallel fan-out, then synthesis, then an evaluator gate | `demo_research_swarm.py` | `reports/demo1_trace.json` |
| 2. Coding Agent Team | Team workflow inside an evaluator-optimizer loop, with executed unit tests | `demo_coding_team.py` | `reports/demo2_trace.json` |
| 3. Routing Assistant | An LLM router dispatching to sub-workflows | `demo_routing_assistant.py` | `reports/demo3_trace.json` |

Every demo builds its agents from `config/agents.json`. That includes each agent's `tools` and `permissions`, which the demo code cannot override. Each run uses one `SharedWorkflowMemory` with a `Tracer` attached. At the end it exports the trace to JSON and prints the run summary table.

## Setup

Run every command from the **repository root**.

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Each demo has two modes:

- **Real mode** (the default) runs every agent on the local `Qwen/Qwen2.5-1.5B-Instruct` model through `transformers`. The first run downloads the weights (about 3 GB). After that, runs are fully offline. The device is picked automatically (CUDA, then Apple MPS, then CPU). On an Apple-silicon laptop, Demo 3 takes about 2 minutes. Demo 1 can take close to 10 minutes, mostly in the 512-token synthesis step.
- **Mock mode** (`--mock`) swaps the model for scripted replies. Everything else still runs for real: the agents, config grants, tools, orchestrators, shared memory and tracer. It finishes in under a second and needs no model. Use it to see the flow, then drop `--mock` to watch the model do the work.

Flags shared by all three demos:

| Flag | Meaning |
|---|---|
| `--mock` | Scripted replies instead of the local model |
| `--trace PATH` | Where to export the JSON trace (default `reports/demoN_trace.json`) |
| `--config PATH` | Agent definitions (default `config/agents.json`) |
| `--quiet` | No progress output or summary table; still exports the trace |

Each script exits with `0` when the run succeeds and `1` when it does not. What counts as success is listed under each demo.

---

## Demo 1: Research Swarm

```
research question
       |
ResearchPlannerAgent          splits the question into subtopics (JSON array)
       |
 +-----+-----+                ParallelOrchestrator fan-out (threads)
 v     v     v
ResearchAgent-1..N            each one: search tool -> grounded findings -> findings/<n>
 +-----+-----+                fan-in
       v
ResearchSynthesizerAgent      structured report -> shared_memory["research_report"]
       |
ResearchEvaluatorAgent        {"is_complete", "feedback"}; if incomplete, the synthesizer revises once
```

```bash
python examples/demo_research_swarm.py --mock
python examples/demo_research_swarm.py
python examples/demo_research_swarm.py --question "How is retrieval-augmented generation evaluated?" --subtopics 2
```

**How it works**
- The planner's reply is parsed as a JSON array, or as a bulleted list if it isn't JSON. If neither works, the swarm runs a single worker on the whole question.
- `ParallelOrchestrator` gives every worker the same task, so each `SubtopicResearchAgent` carries its own subtopic. It calls `search` through `use_tool`, which checks the call against the `researcher` grants (`search`, `read_only`).
- Workers get `context_keys=[]` because they run concurrently on one blackboard. Otherwise their prompts would depend on thread timing.
- The synthesizer writes to `research_report` through the `output_key` set in its config.
- If the evaluator rejects the report, the synthesizer gets the worker findings again (read back from the blackboard), its own report and the critique. It then writes one revision.

**Config:** `research_planner`, `researcher`, `research_synthesizer`, `research_evaluator`.

**Success** means the evaluator's final verdict is `is_complete: true`.

**What the trace shows:** a delegation from the planner to each worker, one `search` tool call per worker, a `verification` event per review, and a final blackboard snapshot. That snapshot holds `subtopics`, `findings/1..N`, `research_report` and `research_evaluation`.

> The search tool uses the framework's small offline corpus. Its topics are multi-agent systems, tool use, permissions, RAG and evaluation. Questions near those topics give the workers real snippets to cite.

---

## Demo 2: Coding Agent Team

```
request
   |
CodePlannerAgent       signature, algorithm, edge cases -> shared_memory["design_plan"]
   |
   |   EvaluatorOptimizerPipeline
   |   +------------------------------------------------------------+
   +-> | TeamCoderAgent   implementation -> shared_memory["code_draft"] |
       |    |                                                         |
       |  ReviewGate                                                  |
       |    TestAgent     writes the unit tests once, then runs them  |
       |                  on every draft (python_test_runner tool)    |
       |    CodeEvaluator scores the draft, with the test results     |
       |    failing tests veto approval                               |
       |    |                                                         |
       |  rejected -> the feedback goes back to the coder             |
       +------------------------------------------------------------+
   |  approved
ReleaseAgent           file_write -> <output-dir>/<function>.py and test_<function>.py
```

```bash
python examples/demo_coding_team.py --mock
python examples/demo_coding_team.py
python examples/demo_coding_team.py --request "Write a Python function slugify(text) ..." --rounds 4 --output-dir workspace/slugify
```

**How it works**
- **Frozen tests.** `TestAgent` writes the tests from the request, the plan and the first draft. After that they don't change, so a buggy draft can't pass by weakening its own tests. Two exceptions, each allowed once per run, cover tests that are wrong rather than code that is:
  - If the test file fails to load, TestAgent rewrites it with the error in hand.
  - **Dispute.** If the same tests fail with the same errors after the coder has revised, TestAgent re-checks the failing tests against the *request*. It corrects only the tests that contradict it. This came from a real run: Qwen wrote `assert median([]) == None` although the request says to raise `ValueError`, and correct code was rejected every round. The dispute appears in the trace as a `delegation` from TestAgent to itself.
- **Test execution.** `python_test_runner` is a tool the demo registers. It runs the draft and the tests in a separate `python -I` process, in a temporary directory, with a 10 s timeout. It requires the `write` tier because running tests means writing files and executing code. The tests run under pytest when it's installed, so parametrized tests, fixtures and test classes work. Otherwise a plain loop calls each `test_*` function. `math` and `pytest` are pre-imported for the tests. The runner imports the implementation and the tests separately, so an import-time error is reported against the file that raised it. For example, if the coder leaves a failing self-check at the bottom of its module, it is told exactly that and asked to remove top-level code.
- **ReviewGate.** This is the evaluator the loop sees. If every test passes, it returns the reviewer's JSON score unchanged. If any test fails, it returns its own verdict: a score of at most 4/10, with feedback that starts with the failing tests. A reviewer that likes the code still cannot approve a draft that fails its tests.
- **The loop.** `EvaluatorOptimizerPipeline` sends the coder its previous draft together with that feedback, for up to `--rounds` rounds. The pass threshold is 8/10.
- **Release.** Files are written only after approval, by `ReleaseAgent`. Its config (`code_releaser`) is the only one in the team that grants both `file_write` and `write`. The coder has no tools and the tester was never assigned `file_write`, so neither can write a file.

**Config:** `code_planner`, `team_coder`, `test_writer`, `code_evaluator`, `code_releaser`.

**Success** means the code was approved and released. If no draft passes within the round limit, nothing is written to disk and the script exits `1`.

In mock mode, the first draft deliberately misses the empty-list and type checks, so the run passes through every step:
round 1 is rejected (3/5 tests pass) → the feedback goes back to the coder → round 2 is approved (5/5 tests pass, 9/10 score) → two files are written.

**What the trace shows:** each round's `verification` event (with `score` and `threshold`), a `delegation` from ReviewGate to the coder for each loop-back, `python_test_runner` calls, and the `file_write` calls with `required_permission: "write"`.

> ⚠️ This demo runs code written by the model. The child process, temp directory and timeout are isolation, not a security sandbox. Run it only on a machine where that is acceptable.

---

## Demo 3: Routing Assistant

```
user query
    |
RouterAgent (LLM)     {"route": ..., "reasoning": ...}
    |                 falls back to keyword rules if the reply is unusable
    +-- code  -> SequentialOrchestrator: CodePlannerAgent -> TeamCoderAgent
    +-- math  -> MathAnalysisPipeline:   MathAgent -> calculator tool -> AnalystAgent
    +-- qa    -> DirectAnswerAgent
    +-- other -> FallbackAgent
```

```bash
python examples/demo_routing_assistant.py --mock
python examples/demo_routing_assistant.py
python examples/demo_routing_assistant.py --query "What is 15% of 240?" --query "Explain what a mutex is."
python examples/demo_routing_assistant.py --interactive      # blank line or "quit" ends the session
```

**How it works**
- **The router.** `LLMRouter` subclasses `RouterOrchestrator` and replaces only the routing decision. The router agent sees the list of routes and replies with a route key and one sentence of reasoning. That reasoning is what gets recorded in the trace's `route_decision` event and in `routing_log`.
- **Fallback.** If the reply isn't JSON or names an unknown route, the keyword rules pick the route instead. The recorded reasoning then says so, for example `router reply was not JSON; keyword rules: keyword 'python' matched`. A reply of `"other"` goes to the fallback agent.
- **Math pipeline.** The model only writes `calculator(...)` expressions and never does the arithmetic itself. The expressions run through the math agent's `calculator` tool (`read_only`). The analyst then explains the checked numbers.
- **Session state.** All queries in a session share one blackboard and one trace. Every agent has `context_keys=[]`, so an answer to one query never has earlier queries' artifacts pasted into its prompt.

**Config:** `request_router`, `code_planner`, `team_coder`, `math`, `analyst`, `direct_answer`, `fallback`.

**Success** means every query's sub-workflow completed.

**What the trace shows:** for each query, the router's `agent_call` and a `route_decision` with its reasoning. Queries routed to a sub-workflow also get a `delegation` to `SequentialOrchestrator` or `MathAnalysisPipeline`. After that come the sub-workflow's own steps and tool calls.

## Tests

`tests/test_demos.py` runs all three demos offline with scripted clients. A guard fixture fails any test that tries to load a model. The tests check the flows, the blackboard contents, the exported traces, the permission boundaries and the loop-back paths (a rejected report, a rejected draft, an unusable router reply). They also run each script from the command line with `--mock`.

```bash
python -m pytest tests/test_demos.py -q
```
