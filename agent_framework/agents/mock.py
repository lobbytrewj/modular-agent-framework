from __future__ import annotations

from typing import Optional

from agent_framework.agents.base import BaseAgent
from agent_framework.core.memory import SharedWorkflowMemory
from agent_framework.core.tasks import RunResult, Task


# A fake agent that always succeeds instantly, used for testing orchestration logic
class MockAgent(BaseAgent):
    def execute(
        self,
        task: Task,
        shared_memory: Optional[SharedWorkflowMemory] = None,
    ) -> RunResult:
        # Automatically runs success everytime
        output = f"MockAgent {self.name} successfully completed task: {task.description}"

        # Naming what it could see on the blackboard makes shared state
        # testable without loading a model: an assertion can check the artifact
        # actually reached the agent, not just that it was stored. Appended
        # only when a blackboard was passed, so the output of an ordinary
        # MockAgent run is unchanged.
        if shared_memory is not None:
            available = ", ".join(shared_memory.keys()) or "none"
            output += f" [shared artifacts visible: {available}]"

        # Agent-local memory works the same here as on a real agent, so a
        # memory test doesn't need an LLM.
        if self.memory is not None:
            self.memory.add_user_message(task.description)
            self.memory.add_assistant_message(output)

        return RunResult(task_id=task.id, success=True, output=output)
