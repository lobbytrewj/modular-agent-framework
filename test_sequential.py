from __future__ import annotations

from agent_framework.agents.mock import MockAgent
from agent_framework.core.tasks import Task
from agent_framework.orchestration.registry import AgentRegistry
from agent_framework.orchestration.sequential import SequentialOrchestrator

# Register the Research Agent, Analysis Agent and Report Agent
registry = AgentRegistry()
registry.register("ResearchAgent", MockAgent)
registry.register("AnalysisAgent", MockAgent)
registry.register("ReportAgent", MockAgent)

# Give system prompts for each agent
research_agent = registry.get("ResearchAgent")(
    name="ResearchAgent",
    role="research",
    system_prompt="Gather raw findings on the given topic.",
)
analysis_agent = registry.get("AnalysisAgent")(
    name="AnalysisAgent",
    role="analysis",
    system_prompt="Analyze the research findings and surface key insights.",
)
report_agent = registry.get("ReportAgent")(
    name="ReportAgent",
    role="report",
    system_prompt="Turn the analysis into a polished final report.",
)

orchestrator = SequentialOrchestrator([research_agent, analysis_agent, report_agent])

# Give it a task with description
initial_task = Task(
    description="Investigate electric vehicle market trends",
    assigned_agent="ResearchAgent",
)

final_result = orchestrator.run(initial_task)

# Print out the results and how the input when from one agent to another
print("=== Sequential Pipeline Execution Trace ===")
for agent, result in zip(orchestrator.agents, orchestrator.step_results):
    status = "OK" if result.success else "FAILED"
    print(f"\n[{agent.name}] ({status})")
    print(f"  output: {result.output}")

print("\n=== Final Result ===")
print(final_result)
