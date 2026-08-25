"""Live integration test for the Goal 9 evaluator-optimizer quality-control loop.

Runs real local inference through create_evaluator_optimizer_pipeline - no
mocks - so what it proves is that the generate/grade/refine cycle holds up
against an actual small model whose JSON arrives truncated, wrapped in fences,
or not at all.

Required config/agents.json entries (already applied):

    "coder_generator": {
      "name": "CoderGeneratorAgent",
      "role": "generator",
      "system_prompt": "You write and refine code ... On a revision you are
        given your previous attempt plus a reviewer's critique: fix every
        defect the reviewer named, keep everything that was already correct,
        and reply with the complete corrected version only ...",
      "model": "Qwen/Qwen2.5-1.5B-Instruct",
      "device": "mps",
      "temperature": 0.1,
      "max_tokens": 256
    },
    "code_evaluator": {
      "name": "CodeEvaluatorAgent",
      "role": "evaluator",
      "system_prompt": "You are a strict code reviewer ... Reply with ONLY a
        JSON object: {\"score\": <1-10>, \"passed\": <bool>, \"feedback\":
        \"<the specific fixes needed>\"} ...",
      ... same model/device/temperature/max_tokens ...
    }

Use "device": "cpu" instead of "mps" on a machine without Apple Silicon.
"""

from __future__ import annotations

import time
import warnings

from agent_framework.core.tasks import Task
from agent_framework.orchestration.evaluator_optimizer import (
    MAX_SCORE,
    create_evaluator_optimizer_pipeline,
)

# Silences the "To copy construct from a tensor ..." notices transformers emits
# on every pipeline load.
warnings.filterwarnings("ignore", category=UserWarning)

AGENTS_CONFIG = "config/agents.json"
GENERATOR_KEY = "coder_generator"
EVALUATOR_KEY = "code_evaluator"
MIN_PASS_SCORE = 8.0
MAX_RETRIES = 2

# Three requirements that are individually checkable, which is what makes this
# a fair test of a rubric-driven loop: the evaluator can name exactly which one
# is missing, and the generator can fix exactly that.
REQUEST = (
    "Write a Python function to check if a string is a palindrome. "
    "It must include type annotations, a docstring, and handle "
    "case/spacing insensitivity."
)


def snippet(text: str, limit: int = 240) -> str:
    """First `limit` characters of an output, collapsed onto one line."""
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[:limit] + " ..."


def test_real_evaluator_optimizer_pipeline() -> None:
    pipeline = create_evaluator_optimizer_pipeline(
        AGENTS_CONFIG,
        generator_key=GENERATOR_KEY,
        evaluator_key=EVALUATOR_KEY,
        min_pass_score=MIN_PASS_SCORE,
        max_retries=MAX_RETRIES,
    )

    assert pipeline.min_pass_score == MIN_PASS_SCORE
    assert pipeline.max_retries == MAX_RETRIES

    task = Task(description=REQUEST, assigned_agent=pipeline.generator_agent.name)

    started = time.perf_counter()
    result = pipeline.run(task)
    elapsed = time.perf_counter() - started

    # =====================================================================
    # Learning summary: Goal 8 and Goal 9 are both loops, and that is
    # roughly where the similarity ends
    # =====================================================================
    #
    # What circulates is different, and everything else follows from it.
    #
    #   Goal 8, orchestrator-worker - the loop EXPANDS the work.
    #     A supervisor holds a roster of specialists and answers "who should do
    #     what next?". Each turn can add subtasks, hand them to different
    #     agents, and grow the deliverable sideways: research, then code, then
    #     a review of that code. The artifact is a COLLECTION - N worker
    #     outputs that a final turn has to consolidate into one answer. The
    #     loop ends on a judgement call ("is the request met?") that only the
    #     supervisor can make, expressed as a control token. Cost per round is
    #     unbounded in shape: one round might dispatch one subtask or five.
    #
    #   Goal 9, evaluator-optimizer - the loop CONVERGES on one thing.
    #     There is one generator, one evaluator, and one artifact that gets
    #     rewritten in place. Nothing is ever added to the deliverable's scope;
    #     each round replaces the draft with a better draft. The question is
    #     not "what next?" but "is this good enough yet?", and it is answered
    #     against a RUBRIC - a numeric score versus a threshold - rather than
    #     by free judgement. That makes termination checkable by the framework
    #     instead of trusted to the model: `score >= min_pass_score` is
    #     arithmetic, where `WORKFLOW_COMPLETE` is a token the model may simply
    #     forget to emit. Cost per round is fixed: exactly one generate and one
    #     evaluate.
    #
    #   Consequence worth noticing: Goal 9 can report *how good* its answer is
    #   and hand back the best draft it ever produced. Goal 8 cannot - it has
    #   no scale, so a run that ends on the iteration guardrail can only say
    #   "I ran out of turns", not "this is a 7 out of 10". The rubric is what
    #   buys measurable quality, and the price is that it only works when the
    #   deliverable is a single artifact you can grade.
    # =====================================================================

    print("---- Live Evaluator-Optimizer Trace ----")
    print(f"generator: {pipeline.generator_agent.name} ({pipeline.generator_agent.role})")
    print(f"evaluator: {pipeline.evaluator_agent.name} ({pipeline.evaluator_agent.role})")
    print(f"threshold: {MIN_PASS_SCORE:g}/{MAX_SCORE:g}, max_retries: {MAX_RETRIES}")
    print(f"request:   {REQUEST}\n")

    for step in pipeline.history:
        evaluation = step.evaluation
        verdict = "PASS" if evaluation.passed else "FAIL"

        print(f"[Round {step.iteration}] generator draft")
        print(f"  draft:    {snippet(step.generated_output)}")
        print(f"[Round {step.iteration}] evaluator verdict")
        print(f"  score:    {evaluation.score:g}/{MAX_SCORE:g} ({verdict})")
        print(f"  raw JSON: {snippet(evaluation.raw_response, 160)}")
        print(f"  feedback: {snippet(evaluation.feedback, 200)}\n")

    # --- Required assertions ---------------------------------------------

    # NOTE: this asserts a QUALITY outcome, not a structural one. It fails if
    # the model never produces a draft scoring >= MIN_PASS_SCORE within
    # MAX_RETRIES rounds - a legitimate result of the quality gate doing its
    # job, not a defect in the loop. The invariants further down hold either
    # way, so read them first when this line goes red.
    assert result.success is True, (
        f"no draft reached the {MIN_PASS_SCORE:g}/{MAX_SCORE:g} threshold in "
        f"{len(pipeline.history)} round(s) - scores were "
        f"{[step.evaluation.score for step in pipeline.history]}. "
        f"Status: {pipeline.status_summary()}"
    )
    assert result.output, "pipeline returned an empty deliverable"
    assert result.output.strip(), "pipeline returned only whitespace"

    # At least one draft was generated and graded.
    assert len(pipeline.history) >= 1, pipeline.history

    # Every recorded round carries a usable verdict. This is the schema
    # guarantee _parse_evaluation exists to provide: a score that is always a
    # real number in range, and a critique that is never empty - because an
    # empty critique is what makes a refinement round a no-op.
    for step in pipeline.history:
        evaluation = step.evaluation
        assert isinstance(evaluation.score, float), (step.iteration, evaluation.score)
        assert 0.0 <= evaluation.score <= MAX_SCORE, (step.iteration, evaluation.score)
        assert evaluation.feedback.strip(), f"round {step.iteration} had an empty critique"
        assert isinstance(evaluation.passed, bool), (step.iteration, evaluation.passed)
        assert step.generated_output.strip(), f"round {step.iteration} produced an empty draft"

    # The returned deliverable is the highest-scoring draft of the run - never
    # a synthesis, never a summary. run() returns a draft verbatim.
    #
    # (The field is OptimizationStep.generated_output; there is no `.draft`.)
    best = max(pipeline.history, key=lambda step: step.evaluation.score)
    assert result.output == best.generated_output, (
        "returned deliverable is not the highest-scoring draft"
    )
    assert pipeline.best_step is not None
    assert result.output == pipeline.best_step.generated_output

    # On a pass the loop exits immediately, so the accepted draft is also the
    # last one graded. (These come apart only on a threshold miss, where an
    # earlier draft can outscore the final one.)
    assert pipeline.passed_threshold is True
    assert result.output == pipeline.history[-1].generated_output
    assert pipeline.history[-1].evaluation.passed or (
        pipeline.history[-1].evaluation.score >= MIN_PASS_SCORE
    )

    # --- Loop invariants --------------------------------------------------

    assert 1 <= len(pipeline.history) <= MAX_RETRIES

    # Iterations are numbered 1..N with no gaps.
    assert [step.iteration for step in pipeline.history] == list(
        range(1, len(pipeline.history) + 1)
    )

    # Only the final round may pass; had an earlier one passed, run() would
    # already have returned and never generated the drafts that followed.
    for step in pipeline.history[:-1]:
        assert not step.evaluation.passed, step.iteration
        assert step.evaluation.score < MIN_PASS_SCORE, step.iteration

    # Whether each refinement is a genuinely NEW draft is reported, not
    # asserted. A model that ignores the critique returns a byte-identical
    # string, which means the loop spent two calls optimising nothing - worth
    # knowing, but it is a fact about the model, not a defect in the loop.
    # Observed live at max_retries=3: rounds 2 and 3 came back identical while
    # every mechanical invariant above still held, so asserting on it would
    # turn model behaviour into a red test.
    drafts = [step.generated_output for step in pipeline.history]
    if len(set(drafts)) != len(drafts):
        print(
            "  WARNING: a refinement round returned an unchanged draft - the "
            "generator ignored the critique, so that round cost two model "
            "calls and improved nothing.\n"
        )

    # step_results accounts for every agent call: one generation, one
    # evaluation per round, and one refinement between consecutive rounds.
    rounds = len(pipeline.history)
    expected_steps = 1 + rounds + (rounds - 1)
    assert len(pipeline.step_results) == expected_steps, (
        f"expected {expected_steps} agent call(s), got {len(pipeline.step_results)}"
    )
    assert all(step.success for step in pipeline.step_results)

    # --- Summary -----------------------------------------------------------

    revisions = len(pipeline.history) - 1
    print("[termination]")
    print(f"  status:    {pipeline.status_summary()}")
    print(f"  rounds:    {len(pipeline.history)} evaluation(s), {revisions} revision(s)")
    print(f"  calls:     {len(pipeline.step_results)} agent call(s)")
    print(f"  elapsed:   {elapsed:.2f}s total\n")

    # The assertions above check the loop's mechanics and the rubric's
    # bookkeeping. They do NOT check that the code is correct - only that it
    # scored well enough. Reading the deliverable is still on you.
    print("---- Final Accepted Deliverable ----")
    print(result.output)


if __name__ == "__main__":
    test_real_evaluator_optimizer_pipeline()
    print("\nAll assertions passed.")
