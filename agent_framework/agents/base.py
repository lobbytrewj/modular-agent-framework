from __future__ import annotations

from abc import ABC, abstractmethod

from agent_framework.core.tasks import RunResult, Task


# The casting contract every agent actor must satisfy before they can go on stage
class BaseAgent(ABC):
    def __init__(self, name: str, role: str, system_prompt: str):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt

    # Every agent must know how to perform when handed a Task script
    @abstractmethod
    def execute(self, task: Task) -> RunResult:
        ...
