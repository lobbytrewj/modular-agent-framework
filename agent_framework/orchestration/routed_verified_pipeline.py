from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from agent_framework.agents import BaseAgent, LLMAgent
from agent_framework.core.tasks import RunResult, Task
from agent_framework.orchestration.hierarchical import OBJECT_FIRST, extract_json
from agent_framework.orchestration.router import (
    DEFAULT_AGENTS_CONFIG,
    DEFAULT_ROUTER_CONFIG,
    RouterOrchestrator,
    create_router_pipeline,
)
from agent_framework.orchestration.sequential import (
    AGENT_PARAMS,
    build_llm_client,
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
        """Ask the verifier whether the output actually satisfies the request.

        The original request travels with the output for the same reason it
        does in the evaluator-optimizer: verification is a comparison, and an
        answer can only be judged incomplete relative to what was asked. A
        verifier shown output alone can tell you the code runs; it cannot tell
        you it solves the wrong problem.
        """
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
        """Compose the next attempt: the request, what was produced, what was wrong.

        Two constraints shape this text, and they pull in different directions.

        As a PROMPT it needs the critique, or the next attempt repeats the
        last one - the same feedback-injection argument as the evaluator-
        optimizer's refinement task.

        As a ROUTING KEY it is also the string the router will keyword-match
        on, which the evaluator-optimizer never had to worry about. That is why
        the original request is placed FIRST and verbatim: keyword rules match
        in registration order against the whole description, so leading with
        the original text keeps a retry landing on the same route that the
        first attempt matched. Without that, a critique mentioning "explain" or
        "research" could silently reroute a coding task to a different
        destination - and a retry that changes both the worker and the
        instructions tells you nothing about which change mattered.
        """
        if not feedback.strip() or feedback.strip() == NO_FEEDBACK:
            # An absent critique makes the retry a re-roll rather than a
            # correction, so say something directive instead of pasting the
            # placeholder in as though it were a defect report.
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

    # --- Status --- ( still wokring on it)

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
    def run(self, initial_task: Task) -> RunResult:
        self.history = []
        self.step_results = []
        self.verified = False
        self.attempts_used = 0

        current_task = initial_task
        latest_output = ""

        for attempt in range(1, self.max_attempts + 1):
            self.attempts_used = attempt

            routed = self.router.run(current_task)
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


            verification = self.verifier_agent.execute(
                self._build_verification_task(initial_task, latest_output)
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

        logger.info("Verification not achieved: %s", self.status_summary())
        return RunResult(
            task_id=initial_task.id,
            success=False,
            output=latest_output,
            error=self.status_summary(),
        )

