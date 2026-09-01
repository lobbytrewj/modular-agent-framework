from __future__ import annotations

from typing import Optional

from agent_framework.agents.base import BaseAgent
from agent_framework.core.llm import LLMClient, LLMError
from agent_framework.core.memory import AgentMemory, SharedWorkflowMemory
from agent_framework.core.tasks import RunResult, Task

# How many of an agent's own past turns are replayed by default when it has
# agent-local memory. Six is three exchanges - enough for a refinement loop to
# remember what it already tried, short enough not to crowd out the actual
# instruction on the small local models this framework targets.
DEFAULT_HISTORY_WINDOW = 6


# An agent that delegates its thinking to a real LLM using LLMClient
class LLMAgent(BaseAgent):
    def __init__(
        self,
        name: str,
        role: str,
        system_prompt: str,
        llm_client: Optional[LLMClient] = None,
        memory: Optional[AgentMemory] = None,
        history_window: Optional[int] = DEFAULT_HISTORY_WINDOW,
        context_keys: Optional[list[str]] = None,
        output_key: Optional[str] = None,
    ):
        super().__init__(name, role, system_prompt, memory=memory)
        # Accept an injected client so tests can pass a fake; build a real one otherwise
        self.llm_client = llm_client or LLMClient()

        # How many of its own turns this agent replays out of agent-local
        # memory. None replays all of them.
        self.history_window = history_window

        # Which SHARED artifacts this agent wants pasted into its prompt.
        # None means "everything on the blackboard"; naming keys keeps a
        # focused agent from being handed deliverables it has no use for.
        self.context_keys = context_keys

        # Where this agent publishes its own output on the blackboard. Left
        # None it publishes nothing - an agent's answer is already returned in
        # the RunResult, so writing it to shared memory is an explicit choice
        # about what later steps should be able to look up by name.
        self.output_key = output_key

    def _build_user_message(
        self, task: Task, shared_memory: Optional[SharedWorkflowMemory]
    ) -> str:
        """Compose this turn's prompt body: shared artifacts, then the task.

        The blackboard goes FIRST and the instruction last, because the last
        thing in the prompt is what a small model weights most heavily - lead
        with a wall of prior deliverables and the actual assignment can get
        buried in it.
        """
        sections: list[str] = []

        # SHARED workflow memory: deliverables other agents produced. Read-only
        # from this agent's point of view unless it has an output_key.
        if shared_memory is not None:
            context = shared_memory.to_prompt_context(self.context_keys)
            if context:
                sections.append(context)

        sections.append(task.description)
        if task.input_data:
            sections.append(f"Input data:\n{task.input_data}")

        return "\n\n".join(sections)

    def execute(
        self,
        task: Task,
        shared_memory: Optional[SharedWorkflowMemory] = None,
    ) -> RunResult:
        user_message = self._build_user_message(task, shared_memory)

        # AGENT-LOCAL memory: this agent's own past turns, replayed as real
        # chat messages rather than flattened into the prompt text, so the
        # model's chat template marks them as history instead of leaving the
        # model to infer it from a transcript. Read BEFORE this turn is
        # recorded, or the prompt would contain a copy of itself.
        history = (
            self.memory.as_chat_messages(self.history_window)
            if self.memory is not None
            else None
        )

        # Passed only when non-empty: a client without a `history` parameter
        # (a fake in a test, say) then keeps working unchanged.
        completion_kwargs = {"history": history} if history else {}

        try:
            output = self.llm_client.complete(
                system_prompt=self.system_prompt,
                user_message=user_message,
                **completion_kwargs,
            )
        except LLMError as exc:
            return RunResult(task_id=task.id, success=False, error=str(exc))

        if self.memory is not None:
            # Both halves of the turn are recorded together, and only once the
            # call succeeded. Recording the prompt on a failed call would leave
            # a user turn with no reply in the history, which the next call
            # would replay as an unanswered question.
            self.memory.add_user_message(user_message)
            self.memory.add_assistant_message(output)

        # Publish the deliverable so later steps - in this workflow or another
        # one - can read it by name instead of having it forwarded through
        # every prompt in between.
        if shared_memory is not None and self.output_key:
            shared_memory.set(self.output_key, output)

        return RunResult(task_id=task.id, success=True, output=output)
