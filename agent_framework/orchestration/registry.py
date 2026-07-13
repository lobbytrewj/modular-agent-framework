from __future__ import annotations

from agent_framework.agents.base import BaseAgent


# Looks up an agent's blueprint by a string name
class AgentRegistry:
    def __init__(self):
        # Maps a name -> an agent CLASS, not an instance
        self._agents: dict[str, type[BaseAgent]] = {}

    def register(self, name: str, agent_class: type[BaseAgent]) -> None:
        # Add a new listing to the phone book
        self._agents[name] = agent_class

    def get(self, name: str) -> type[BaseAgent]:
        # Look up a listing and fails if the name was never registered
        if name not in self._agents:
            raise KeyError(f"No agent registered under name '{name}'")
        return self._agents[name]
