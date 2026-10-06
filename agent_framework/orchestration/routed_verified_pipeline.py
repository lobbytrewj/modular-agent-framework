from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from agent_framework.agents import BaseAgent, LLMAgent, execute_agent
from agent_framework.core.memory import SharedWorkflowMemory, call_with_shared_memory
from agent_framework.core.tasks import RunResult, Task
from agent_framework.observability.tracer import resolve_tracer
from agent_framework.orchestration.hierarchical import OBJECT_FIRST, extract_json
from agent_framework.orchestration.router import (
    DEFAULT_AGENTS_CONFIG,
    DEFAULT_ROUTER_CONFIG,
    RouterOrchestrator,
    create_router_pipeline,
)
from agent_framework.orchestration.sequential import (
    build_agent_kwargs,
    load_agent_config,
)

logger = logging.getLogger(__name__)

_COMPLETE_FIELDS = (
    "is_complete", "complete", "completed", "passed", "satisfied", "done", "ok", "valid",
)
_FEEDBACK_FIELDS = ("feedback", "critique", "reason", "details", "issues", "comments")

NO_FEEDBACK = "(the verifier gave no specific feedback)"


@dataclass
class VerificationVerdict:
    is_complete: bool
    feedback: str
    raw_response: str = ""

    def __repr__(self) -> str:  # readable in a test trace
        return (
            f"VerificationVerdict(is_complete={self.is_complete}, "
            f"feedback={self.feedback[:60]!r})"
        )


class AutoRoutedVerifiedPipeline:
    def __init__(
        self,
        router: RouterOrchestrator,
        verifier_agent: BaseAgent,
        max_attempts: int = 3,
    ):
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        self.router = router
        self.verifier_agent = verifier_agent
        self.max_attempts = max_attempts

        self.history: list[dict] = []
        self.step_results: list[RunResult] = []

        self.verified: bool = False
        self.attempts_used: int = 0

    # --- Prompt construction ---

    def _build_verification_task(self, original_task: Task, workflow_output: str) -> Task:
        """Ask the verifier whether the output actually satisfies the request."""
        description = (
            f"Original user request:\n{original_task.description}\n\n"
            f"Output produced by the workflow:\n{workflow_output}\n\n"
            "Check the output against EVERY requirement in the original "
            "request. Is anything missing, wrong, incomplete, or truncated?\n\n"
            "Reply with ONLY a JSON object, no other text:\n"
            '{"is_complete": <true or false>, "feedback": "<what is missing '
            'or wrong>"}\n\n'
            "Set is_complete to true only if every requirement is fully met. "
            "When it is false, feedback must name the specific gaps and what "
            "would fix them - not general advice."
        )
        return Task(
            description=description,
            assigned_agent=self.verifier_agent.name,
            input_data=original_task.input_data,
        )

    def _build_retry_task(
        self, original_task: Task, feedback: str, previous_output: str
    ) -> Task:
        """Compose the next attempt: the request, what was produced, what was wrong."""
        if not feedback.strip() or feedback.strip() == NO_FEEDBACK:
            critique_block = (
                "A verifier judged that output incomplete but did not say why. "
                "Re-examine the request yourself and produce a more complete, "
                "more specific answer that covers every requirement."
            )
        else:
            critique_block = (
                f"A verifier judged that output INCOMPLETE for this reason:\n{feedback}"
            )

        description = (
            f"{original_task.description}\n\n"
            f"--- A previous attempt at this request ---\n{previous_output}\n\n"
            f"{critique_block}\n\n"
            "Produce a corrected, complete response to the original request "
            "above. Fix everything the verifier objected to, keep whatever was "
            "already right, and reply with the full answer only."
        )
        return Task(
            description=description,
            assigned_agent=original_task.assigned_agent,
            input_data=original_task.input_data,
        )

    # --- Verdict parsing --- (still need to do)
    @staticmethod
    def _coerce_bool(value) -> Optional[bool]:
        """Read a completeness flag that may be a bool, a string, or a number."""
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("true", "yes", "complete", "completed", "pass", "passed", "y", "1"):
                return True
            if lowered in ("false", "no", "incomplete", "fail", "failed", "n", "0"):
                return False
        return None

    def _parse_verdict(self, text: str) -> VerificationVerdict:
        """Turn the verifier's reply into a verdict. Always returns one."""
        return parse_verification_verdict(text)

    # --- Status ---

    def status_summary(self) -> str:
        """One line describing how the run ended."""
        if not self.history:
            return "no attempt was ever verified"

        routes = " -> ".join(entry["chosen_route"] for entry in self.history)
        if self.verified:
            return (
                f"verified on attempt {self.attempts_used} of {self.max_attempts} "
                f"(routes: {routes})"
            )
        return (
            f"NOT verified after {self.attempts_used} attempt(s) "
            f"(routes: {routes}); returning the last output produced"
        )

    # --- The run loop ---

    def run(
        self,
        initial_task: Task,
        shared_memory: Optional[SharedWorkflowMemory] = None,
    ) -> RunResult:
        self.history = []
        self.step_results = []
        self.verified = False
        self.attempts_used = 0

        current_task = initial_task
        latest_output = ""
        tracer = resolve_tracer(shared_memory)

        for attempt in range(1, self.max_attempts + 1):
            self.attempts_used = attempt

            routed = call_with_shared_memory(
                self.router.run, current_task, shared_memory
            )
            self.step_results.append(routed)

            chosen_route = (
                self.router.routing_log[-1]["route"] if self.router.routing_log else "unknown"
            )

            if not routed.success:
                logger.warning(
                    "Routed workflow '%s' failed on attempt %d: %s",
                    chosen_route,
                    attempt,
                    routed.error,
                )
                return RunResult(
                    task_id=initial_task.id,
                    success=False,
                    output=latest_output or None,
                    error=(
                        f"Routed workflow '{chosen_route}' failed on attempt "
                        f"{attempt}: {routed.error}"
                    ),
                )

            latest_output = routed.output or ""

            verification = execute_agent(
                self.verifier_agent,
                self._build_verification_task(initial_task, latest_output),
                shared_memory,
            )
            self.step_results.append(verification)

            if not verification.success:
                logger.warning(
                    "Verifier failed on attempt %d: %s", attempt, verification.error
                )
                return RunResult(
                    task_id=initial_task.id,
                    success=False,
                    output=latest_output,
                    error=(
                        f"Verification failed on attempt {attempt} "
                        f"({verification.error}); returning the unverified output"
                    ),
                )

            verdict = self._parse_verdict(verification.output or "")
            self.history.append(
                {
                    "attempt": attempt,
                    "chosen_route": chosen_route,
                    "workflow_output": latest_output,
                    "verdict": verdict,
                }
            )
            if tracer is not None:
                # The verdict as the pipeline read it, not just the raw text
                # the verifier produced: the agent's own AGENT_CALL event
                # already has the reply, this says what the run made of it.
                tracer.record_verification(
                    self.verifier_agent,
                    is_complete=verdict.is_complete,
                    feedback=verdict.feedback,
                    attempt=attempt,
                    raw_response=verdict.raw_response,
                    workflow="routed_verified",
                    chosen_route=chosen_route,
                    max_attempts=self.max_attempts,
                )

            if verdict.is_complete:
                self.verified = True
                logger.info(
                    "Output verified on attempt %d via route '%s'", attempt, chosen_route
                )
                return RunResult(
                    task_id=initial_task.id,
                    success=True,
                    output=latest_output,
                )

            if attempt == self.max_attempts:
                break

            current_task = self._build_retry_task(
                initial_task, verdict.feedback, latest_output
            )
            if tracer is not None:
                tracer.record_delegation(
                    self.verifier_agent,
                    "router",
                    current_task,
                    reason=f"attempt {attempt} judged incomplete; retrying as attempt {attempt + 1}",
                    workflow="routed_verified",
                    attempt=attempt + 1,
                    feedback=verdict.feedback,
                )

        logger.info("Verification not achieved: %s", self.status_summary())
        return RunResult(
            task_id=initial_task.id,
            success=False,
            output=latest_output,
            error=self.status_summary(),
        )


def parse_verification_verdict(text: str) -> VerificationVerdict:
    """Read a verifier reply of the form {"is_complete": ..., "feedback": ...}.

    Module-level so any workflow that asks an agent for a completeness
    verdict can read the reply the same forgiving way this pipeline does.
    Always returns a verdict; an unreadable reply counts as incomplete.
    """
    text = text or ""
    payload = extract_json(text, OBJECT_FIRST)

    is_complete: Optional[bool] = None
    feedback = ""
    verdict_parsed = isinstance(payload, dict)

    # --- Tier 1: a real JSON object, which is what we asked for ---
    if isinstance(payload, dict):
        for wrapper in ("verdict", "result", "verification", "evaluation"):
            inner = payload.get(wrapper)
            if isinstance(inner, dict):
                payload = inner
                break

        for key in _COMPLETE_FIELDS:
            if key in payload:
                is_complete = AutoRoutedVerifiedPipeline._coerce_bool(payload[key])
                if is_complete is not None:
                    break
        for key in _FEEDBACK_FIELDS:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                feedback = value.strip()
                break
            if isinstance(value, list) and value:
                joined = "\n".join(f"- {item}" for item in value if str(item).strip())
                if joined:
                    feedback = joined
                    break

    # --- Tier 2: regex over the prose ---
    if is_complete is None:
        assignment = re.search(
            r'["\']?\b(?:is_complete|complete|completed|passed|satisfied|done)\b'
            r'["\']?\s*[:=]\s*["\']?(\w+)',
            text,
            re.IGNORECASE,
        )
        if assignment:
            is_complete = AutoRoutedVerifiedPipeline._coerce_bool(assignment.group(1))

        if is_complete is None:
            # Negations are checked first
            if re.search(
                r"\b(?:in|not )complete\b|\bincomplete\b|\bmissing\b|\bfail(?:ed|s)?\b",
                text,
                re.IGNORECASE,
            ):
                is_complete = False
            elif re.search(
                r'\b(?:complete|satisfied|correct)\b(?!\s*["\']?\s*[:=])',
                text,
                re.IGNORECASE,
            ):
                is_complete = True

    if not feedback:
        match = re.search(r'"?feedback"?\s*[:=]\s*"([^"]+)"', text, re.IGNORECASE)
        if match:
            feedback = match.group(1).strip()
        elif not verdict_parsed:
            feedback = text.strip()
        else:
            feedback = NO_FEEDBACK

    if is_complete is None:
        logger.warning(
            "Could not read a verdict from the verifier; treating the output "
            "as unverified. Raw reply: %.120s",
            text.replace("\n", " "),
        )
        is_complete = False

    return VerificationVerdict(
        is_complete=is_complete,
        feedback=feedback,
        raw_response=text,
    )


def create_routed_verified_pipeline(
    agents_config_path: str = DEFAULT_AGENTS_CONFIG,
    router_config_path: str = DEFAULT_ROUTER_CONFIG,
    verifier_key: str = "verifier",
    max_attempts: int = 2,
) -> AutoRoutedVerifiedPipeline:
    """Build a router from the two config files and wrap it in a quality gate."""
    router = create_router_pipeline(agents_config_path, router_config_path)

    agents_config = load_agent_config(agents_config_path)
    if verifier_key not in agents_config:
        raise KeyError(f"No agent configured under key '{verifier_key}' in {agents_config_path}")

    entry = agents_config[verifier_key]
    if not isinstance(entry, dict):
        raise ValueError(
            f"Config entry for '{verifier_key}' in {agents_config_path} must be an "
            f"object, got {type(entry).__name__}"
        )

    kwargs = build_agent_kwargs(entry)
    kwargs.setdefault("name", verifier_key)
    kwargs.setdefault("role", verifier_key)
    kwargs.setdefault(
        "system_prompt",
        "You verify whether output satisfies the user's request and reply with "
        'JSON: {"is_complete": <bool>, "feedback": "<what is missing>"}.',
    )
    verifier_agent = LLMAgent(**kwargs)

    return AutoRoutedVerifiedPipeline(
        router=router,
        verifier_agent=verifier_agent,
        max_attempts=max_attempts,
    )