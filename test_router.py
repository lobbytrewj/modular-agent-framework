from __future__ import annotations

from agent_framework.agents.mock import MockAgent
from agent_framework.core.tasks import Task
from agent_framework.orchestration.router import RouterOrchestrator
from agent_framework.orchestration.sequential import SequentialOrchestrator

coding_agent = MockAgent(
    name="CodingAgent", role="coding", system_prompt="Fix and write code."
)
math_agent = MockAgent(
    name="MathAgent", role="math", system_prompt="Solve math problems."
)
writing_agent = MockAgent(
    name="WritingAgent", role="writing", system_prompt="Draft written content."
)
research_agent = MockAgent(
    name="ResearchAgent", role="research", system_prompt="Gather raw findings."
)
direct_answer_agent = MockAgent(
    name="DirectAnswerAgent",
    role="direct_answer",
    system_prompt="Answer simple factual questions directly.",
)
fallback_agent = MockAgent(
    name="FallbackAgent",
    role="fallback",
    system_prompt="Handle anything that doesn't match a known category.",
)

research_pipeline = SequentialOrchestrator([research_agent, writing_agent])

router = RouterOrchestrator(fallback_destination=fallback_agent)
router.register_route("coding", coding_agent, keywords=["python", "code", "bug"])
router.register_route("math", math_agent, keywords=["calculate", "math", "sum"])
router.register_route("writing", writing_agent, keywords=["essay", "write", "draft"])
router.register_route(
    "research", research_pipeline, keywords=["research", "investigate"]
)
router.register_route(
    "direct_answer", direct_answer_agent, keywords=["define", "explain"]
)

tasks = [
    Task(description="Fix a bug in my python code", assigned_agent="router"),
    Task(description="Calculate the sum of primes", assigned_agent="router"),
    Task(
        description="Research and investigate AI market trends",
        assigned_agent="router",
    ),
    Task(description="What is the color of the sky?", assigned_agent="router"),
]

print("=== Router Execution Trace ===")
for task in tasks:
    result = router.run(task)
    decision = router.routing_log[-1]
    status = "OK" if result.success else "FAILED"

    print(f"\nTask: {task.description!r}")
    print(f"  -> route selected: {decision['route']} ({status})")
    print(f"  -> output: {result.output}")

print("\n=== Full Routing Log ===")
for entry in router.routing_log:
    print(entry)
