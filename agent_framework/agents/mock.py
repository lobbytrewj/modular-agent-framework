from __future__ import annotations

from agent_framework.agents.base import BaseAgent
from agent_framework.core.tasks import RunResult, Task


# A fake agent that always succeeds instantly, used for testing orchestration logic
class MockAgent(BaseAgent):
    def execute(self, task: Task) -> RunResult:
        # Automatically runs success everytime
        output = f"MockAgent {self.name} successfully completed task: {task.description}"
        return RunResult(task_id=task.id, success=True, output=output)
