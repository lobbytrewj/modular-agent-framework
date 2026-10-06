from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from agent_framework.agents import BaseAgent, LLMAgent, execute_agent
from agent_framework.core.memory import SharedWorkflowMemory
from agent_framework.core.tasks import RunResult, Task
from agent_framework.observability.tracer import resolve_tracer
from agent_framework.orchestration.hierarchical import OBJECT_FIRST, extract_json
from agent_framework.orchestration.registry import AgentRegistry
from agent_framework.orchestration.sequential import (
    build_agent_kwargs,
    load_agent_config,
)

logger = logging.getLogger(__name__)

# The grading scale the evaluator is asked to use. Scores are normalised into
# this range before anything reads them, so no caller has to wonder whether a
# given number is out of 10, out of 100, or nonsense.
MIN_SCORE = 0.0
MAX_SCORE = 10.0

_SCORE_FIELDS = ("score", "rating", "grade", "quality", "value")
_PASSED_FIELDS = ("passed", "pass", "accepted", "approved", "ok", "is_valid")
_FEEDBACK_FIELDS = ("feedback", "critique", "comments", "issues", "reason", "notes")

# Recorded when the evaluator returns a well-formed verdict with no critique in
# it - observed live as {"score": 9, "passed": true, "feedback": null}. It is a
# statement of what happened, not an instruction: EvaluationResult records what
# the evaluator SAID, and turning "it said nothing" into "go re-examine your
# draft" would file a directive against rounds that passed and need no further
# work. _build_refinement_task turns it into something actionable, because that
# is the only place it is ever used.
NO_CRITIQUE = "(the reviewer gave no specific critique)"


@dataclass
class EvaluationResult:
    score: float
    passed: bool
    feedback: str
    raw_response: str = ""

    def __repr__(self) -> str:
        return (
            f"EvaluationResult(score={self.score:.1f}, passed={self.passed}, "
            f"feedback={self.feedback[:60]!r})"
        )

@dataclass
class OptimizationStep:
    iteration: int
    generated_output: str
    evaluation: EvaluationResult

    def __repr__(self) -> str:
        return (
            f"OptimizationStep(iteration={self.iteration}, "
            f"score={self.evaluation.score:.1f}, passed={self.evaluation.passed})"
        )



class EvaluatorOptimizerPipeline:
    def __init__(
        self,
        generator_agent: BaseAgent,
        evaluator_agent: BaseAgent,
        min_pass_score: float = 8.0,
        max_retries: int = 3,
    ):
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        if not MIN_SCORE <= min_pass_score <= MAX_SCORE:
            raise ValueError(
                f"min_pass_score must be within {MIN_SCORE}-{MAX_SCORE}, got {min_pass_score}"
            )

        self.generator_agent = generator_agent
        self.evaluator_agent = evaluator_agent
        self.min_pass_score = float(min_pass_score)
        self.max_retries = max_retries

        self.history: list[OptimizationStep] = []
        self.step_results: list[RunResult] = []

        self.passed_threshold: bool = False
        self.best_step: Optional[OptimizationStep] = None

    @classmethod
    def from_registry(
        cls,
        generator_name: str,
        evaluator_name: str,
        registry: AgentRegistry,
        agent_kwargs: Optional[dict[str, dict]] = None,
        min_pass_score: float = 8.0,
        max_retries: int = 3,
    ) -> "EvaluatorOptimizerPipeline":
        """Build the generator/evaluator pair from registered agent classes."""
        agent_kwargs = agent_kwargs or {}

        def instantiate(name: str, default_prompt: str) -> BaseAgent:
            agent_cls = registry.get(name)
            kwargs = dict(agent_kwargs.get(name, {}))
            kwargs.setdefault("name", name)
            kwargs.setdefault("role", name)
            kwargs.setdefault("system_prompt", default_prompt)
            return agent_cls(**kwargs)

        generator = instantiate(
            generator_name,
            f"You are '{generator_name}'; produce the requested work and revise "
            "it precisely as review feedback directs.",
        )
        evaluator = instantiate(
            evaluator_name,
            f"You are '{evaluator_name}'; grade the work you are shown from 1-10 "
            'and reply with JSON: {"score": <n>, "passed": <bool>, "feedback": "<critique>"}.',
        )
        return cls(
            generator,
            evaluator,
            min_pass_score=min_pass_score,
            max_retries=max_retries,
        )


    def _build_evaluation_task(self, draft: str, task: Task) -> Task:
        """Ask the evaluator to grade one draft against the original request.

        The original request is included deliberately. Without it the evaluator
        can only judge the draft on general merit - it has no way to notice
        that a perfectly clean function answers a question nobody asked. Grading
        is a comparison, and it needs both sides.
        """
        description = (
            f"Original request:\n{task.description}\n\n"
            f"Draft to evaluate:\n{draft}\n\n"
            "Grade this draft against the original request. Check it for bugs, "
            "missing edge cases, unmet requirements and style problems.\n\n"
            "Reply with ONLY a JSON object, no other text:\n"
            '{"score": <integer 1-10>, "passed": <true or false>, '
            '"feedback": "<the specific fixes needed>"}\n\n'
            f"Set passed to true only if the score is {self.min_pass_score:g} or "
            "higher. When passed is false, feedback must name concrete defects "
            "and how to fix each one - not general advice."
        )
        return Task(
            description=description,
            assigned_agent=self.evaluator_agent.name,
            input_data=task.input_data,
        )

    def _build_refinement_task(
        self, previous_draft: str, feedback: str, original_task: Task
    ) -> Task:
        """Compose the generator's next turn: its own draft, plus the critique.

        This is the feedback-injection point, and the whole loop turns on it.
        The generator is stateless between calls - it does not remember writing
        the previous draft - so a refinement prompt saying only "apply this
        feedback" would be asking it to patch code it cannot see. All three
        pieces have to travel together:

          * the original request, or the model optimises for the critique and
            quietly drops requirements the critique did not mention;
          * the previous draft, which is the thing being edited;
          * the critique, which is the only NEW information this round.
            Everything else in this prompt was already true last round, so the
            critique is precisely what makes round N+1 differ from round N -
            which is why an empty or generic critique produces a near-identical
            draft and a stalled score.
        """
        # An absent critique therefore has to be replaced with SOMETHING
        # directive, or the round is a guaranteed no-op. Self-review is the
        # weakest useful instruction available, and strictly better than
        # pasting "(the reviewer gave no specific critique)" in as though it
        # were a defect report.
        if not feedback.strip() or feedback.strip() == NO_CRITIQUE:
            critique_block = (
                "The reviewer did not name any specific defect, but the draft "
                "did not meet the quality bar. Re-examine it yourself for bugs, "
                "unhandled edge cases (empty, negative, zero, None, wrong "
                "type), and requirements of the original request it does not "
                "yet satisfy."
            )
        else:
            critique_block = f"A reviewer found these problems with it:\n{feedback}"

        description = (
            f"Original request:\n{original_task.description}\n\n"
            f"Your previous attempt:\n{previous_draft}\n\n"
            f"{critique_block}\n\n"
            "Rewrite your previous attempt so that every problem above is "
            "fixed, while still satisfying the original request. Reply with "
            "the complete corrected version only - no commentary on what you "
            "changed, and do not leave anything out that was already correct."
        )
        return Task(
            description=description,
            assigned_agent=self.generator_agent.name,
            input_data=original_task.input_data,
        )


    @staticmethod
    def _coerce_score(value) -> Optional[float]:
        """Normalise whatever the evaluator called a score onto the 0-10 scale.

        Three real shapes have to survive this:
          "8"      -> 8.0    (a string, because JSON quoting is a coin flip)
          85       -> 8.5    (graded out of 100 despite being asked for 1-10)
          -3 / 240 -> clamped into range rather than believed

        The 100-point rescale matters more than it looks: clamping 85 straight
        to 10.0 would turn a mediocre draft into an instant pass, so a number
        that is obviously percentage-shaped is rescaled rather than truncated.
        """
        if isinstance(value, bool):  # bool is an int subclass; not a score
            return None
        if isinstance(value, str):
            match = re.search(r"-?\d+(?:\.\d+)?", value)
            if not match:
                return None
            value = match.group()
        try:
            score = float(value)
        except (TypeError, ValueError):
            return None

        if MAX_SCORE < score <= 100:
            score /= 10.0
        return max(MIN_SCORE, min(MAX_SCORE, score))

    @staticmethod
    def _coerce_bool(value) -> Optional[bool]:
        """Read a pass/fail flag that may be a bool, a string, or a number."""
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("true", "yes", "pass", "passed", "y", "1"):
                return True
            if lowered in ("false", "no", "fail", "failed", "n", "0"):
                return False
        return None

    def _parse_evaluation(self, text: str) -> EvaluationResult:
        """Turn the evaluator's reply into a verdict. Always returns one.
        """
        text = text or ""
        payload = extract_json(text, OBJECT_FIRST)

        score: Optional[float] = None
        passed: Optional[bool] = None
        feedback = ""

        verdict_parsed = isinstance(payload, dict)
        if isinstance(payload, dict):
            for wrapper in ("evaluation", "result", "assessment"):
                inner = payload.get(wrapper)
                if isinstance(inner, dict):
                    payload = inner
                    break

            for key in _SCORE_FIELDS:
                if key in payload:
                    score = self._coerce_score(payload[key])
                    if score is not None:
                        break
            for key in _PASSED_FIELDS:
                if key in payload:
                    passed = self._coerce_bool(payload[key])
                    if passed is not None:
                        break
            for key in _FEEDBACK_FIELDS:
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    feedback = value.strip()
                    break
                # Some models answer with a list of defects.
                if isinstance(value, list) and value:
                    feedback = "\n".join(f"- {item}" for item in value if str(item).strip())
                    if feedback:
                        break

        if score is None:
            patterns = (
                r"(\d+(?:\.\d+)?)\s*(?:/|out of)\s*10\b",       # "7/10"
                r"(\d+(?:\.\d+)?)\s*(?:/|out of)\s*100\b",      # "70/100"
                r'(?:score|rating|grade)\b\D{0,12}(\d+(?:\.\d+)?)',
            )
            for index, pattern in enumerate(patterns):
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    raw = float(match.group(1))
                    score = self._coerce_score(raw / 10.0 if index == 1 else raw)
                    if score is not None:
                        break

        if passed is None:
            assignment = re.search(
                r'["\']?\b(?:passed|pass|accepted|approved)\b["\']?\s*[:=]\s*["\']?(\w+)',
                text,
                re.IGNORECASE,
            )
            if assignment:
                passed = self._coerce_bool(assignment.group(1))

            if passed is None:
                if re.search(r"\bfail(?:ed|s|ure)?\b|\bnot\s+pass", text, re.IGNORECASE):
                    passed = False
                elif re.search(
                    r'\bpass(?:ed|es)\b(?!\s*["\']?\s*[:=])', text, re.IGNORECASE
                ):
                    passed = True

        if not feedback:
            match = re.search(
                r'"?feedback"?\s*[:=]\s*"([^"]+)"', text, re.IGNORECASE
            )
            if match:
                feedback = match.group(1).strip()
            elif not verdict_parsed:
                # No structured verdict at all, so the reply IS the critique.
                # Not clean feedback, but real information about what the
                # evaluator objected to.
                feedback = text.strip()
            else:
                # A well-formed verdict with an empty or null critique. Echoing
                # the raw JSON back would hand the generator its own score as
                # "the problems with your draft" - no defect to fix, so the next
                # draft returns essentially unchanged and the score stalls.
                feedback = NO_CRITIQUE

        if score is None:
            logger.warning(
                "Could not read a score from the evaluator; treating the draft "
                "as unproven (score %.1f). Raw reply: %.120s",
                MIN_SCORE,
                text.replace("\n", " "),
            )
            score = MIN_SCORE
            # An unreadable grade must never present as a pass, whatever the
            # prose happened to say.
            passed = False

        if passed is None:
            # No explicit flag: the threshold is the definition of passing.
            passed = score >= self.min_pass_score

        return EvaluationResult(
            score=score,
            passed=passed,
            feedback=feedback,
            raw_response=text,
        )


    def _record(self, iteration: int, draft: str, evaluation: EvaluationResult) -> OptimizationStep:
        """Append one graded round to the history and return it."""
        step = OptimizationStep(
            iteration=iteration,
            generated_output=draft,
            evaluation=evaluation,
        )
        self.history.append(step)
        return step

    def _highest_scoring(self) -> Optional[OptimizationStep]:
        """The best draft this run produced.
        """
        best: Optional[OptimizationStep] = None
        for step in self.history:
            if best is None or step.evaluation.score > best.evaluation.score:
                best = step
        return best

    def status_summary(self) -> str:
        """One line describing how the run ended - the 'final status'."""
        if not self.history:
            return "no draft was ever graded"

        best = self.best_step or self._highest_scoring()
        assert best is not None
        rounds = len(self.history)

        if self.passed_threshold:
            return (
                f"passed on round {best.iteration} with score "
                f"{best.evaluation.score:g}/{MAX_SCORE:g} "
                f"(threshold {self.min_pass_score:g}) after {rounds} evaluation(s)"
            )
        return (
            f"did NOT reach the quality threshold {self.min_pass_score:g}/"
            f"{MAX_SCORE:g} in {rounds} round(s); returning the highest-scoring "
            f"draft, from round {best.iteration}, which scored "
            f"{best.evaluation.score:g}"
        )

    def run(
        self,
        initial_task: Task,
        shared_memory: Optional[SharedWorkflowMemory] = None,
    ) -> RunResult:
        # Note what the blackboard is NOT used for here: the draft moves
        # from round to round in the prompt, exactly as before. Refinement
        # depends on the generator seeing its own previous text next to the
        # critique, and routing that through shared state would hide the
        # one thing the loop is actually about. What the blackboard adds is
        # visibility for everyone outside the loop.
        self.history = []
        self.step_results = []
        self.passed_threshold = False
        self.best_step = None

        tracer = resolve_tracer(shared_memory)

        generation = execute_agent(self.generator_agent, initial_task, shared_memory)
        self.step_results.append(generation)
        if not generation.success:

            return generation

        draft = generation.output or ""


        for round_number in range(1, self.max_retries + 1):
            evaluation_result = execute_agent(
                self.evaluator_agent,
                self._build_evaluation_task(draft, initial_task),
                shared_memory,
            )
            self.step_results.append(evaluation_result)

            if not evaluation_result.success:
                logger.warning(
                    "Evaluator failed on round %d: %s", round_number, evaluation_result.error
                )
                return RunResult(
                    task_id=initial_task.id,
                    success=False,
                    output=draft,
                    error=(
                        f"Evaluation failed on round {round_number} "
                        f"({evaluation_result.error}); returning the ungraded draft"
                    ),
                )

            evaluation = self._parse_evaluation(evaluation_result.output or "")
            step = self._record(round_number, draft, evaluation)
            accepted = evaluation.passed or evaluation.score >= self.min_pass_score
            if tracer is not None:
                # The grade as the loop read it; the evaluator's own AGENT_CALL
                # event already carries the raw reply.
                tracer.record_verification(
                    self.evaluator_agent,
                    is_complete=accepted,
                    feedback=evaluation.feedback,
                    attempt=round_number,
                    raw_response=evaluation.raw_response,
                    workflow="evaluator_optimizer",
                    score=evaluation.score,
                    threshold=self.min_pass_score,
                    max_retries=self.max_retries,
                )

            if accepted:
                self.passed_threshold = True
                self.best_step = step
                logger.info(
                    "Draft accepted on round %d with score %.1f", round_number, evaluation.score
                )
                return RunResult(
                    task_id=initial_task.id,
                    success=True,
                    output=draft,
                )

            if round_number == self.max_retries:
                break

            refinement_task = self._build_refinement_task(draft, evaluation.feedback, initial_task)
            if tracer is not None:
                tracer.record_delegation(
                    self.evaluator_agent,
                    self.generator_agent,
                    refinement_task,
                    reason=(
                        f"round {round_number} scored {evaluation.score:g} "
                        f"(threshold {self.min_pass_score:g}); sending feedback back"
                    ),
                    workflow="evaluator_optimizer",
                    attempt=round_number + 1,
                    feedback=evaluation.feedback,
                )
            refinement = execute_agent(self.generator_agent, refinement_task, shared_memory)
            self.step_results.append(refinement)

            if not refinement.success:
                logger.warning(
                    "Refinement failed on round %d: %s", round_number, refinement.error
                )
                break

            draft = refinement.output or draft

        best = self._highest_scoring()
        self.best_step = best
        if best is None:
            return RunResult(
                task_id=initial_task.id,
                success=False,
                output=draft,
                error="No draft was successfully graded; returning the last draft ungraded",
            )

        logger.info("Quality threshold not met: %s", self.status_summary())

        return RunResult(
            task_id=initial_task.id,
            success=False,
            output=best.generated_output,
            error=self.status_summary(),
        )


def create_evaluator_optimizer_pipeline(
    config_path: str,
    generator_key: str,
    evaluator_key: str,
    min_pass_score: float = 8.0,
    max_retries: int = 3,
) -> EvaluatorOptimizerPipeline:
    """Read the agent definitions off disk and wire up a quality-control loop.
    """
    config = load_agent_config(config_path)

    registry = AgentRegistry()
    agent_kwargs: dict[str, dict] = {}

    for key in (generator_key, evaluator_key):
        if key not in config:
            raise KeyError(f"No agent configured under key '{key}' in {config_path}")

        entry = config[key]
        if not isinstance(entry, dict):
            raise ValueError(
                f"Config entry for '{key}' in {config_path} must be an object, "
                f"got {type(entry).__name__}"
            )

        registry.register(key, LLMAgent)
        agent_kwargs[key] = build_agent_kwargs(entry)

    return EvaluatorOptimizerPipeline.from_registry(
        generator_name=generator_key,
        evaluator_name=evaluator_key,
        registry=registry,
        agent_kwargs=agent_kwargs,
        min_pass_score=min_pass_score,
        max_retries=max_retries,
    )
