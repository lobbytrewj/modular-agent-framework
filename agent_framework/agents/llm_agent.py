from __future__ import annotations

from typing import Optional

from agent_framework.agents.base import BaseAgent
from agent_framework.core.llm import LLMClient, LLMError
from agent_framework.core.tasks import RunResult, Task


# An agent that delegates its thinking to a real LLM using LLMClient
class LLMAgent(BaseAgent):
    def __init__(
        self,
        name: str,
        role: str,
        system_prompt: str,
        llm_client: Optional[LLMClient] = None,
    ):
        super().__init__(name, role, system_prompt)
        # Accept an injected client so tests can pass a fake; build a real one otherwise
        self.llm_client = llm_client or LLMClient()

    def execute(self, task: Task) -> RunResult:
        user_message = task.description
        if task.input_data:
            user_message += f"\n\nInput data:\n{task.input_data}"

        try:
            output = self.llm_client.complete(
                system_prompt=self.system_prompt,
                user_message=user_message,
            )
        except LLMError as exc:
            return RunResult(task_id=task.id, success=False, error=str(exc))

        return RunResult(task_id=task.id, success=True, output=output)
