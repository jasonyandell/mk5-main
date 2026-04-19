"""Illegal-play retry loop.

Move 3 is diagnostic: we want loud failures, not silent fallbacks. If a model
cannot land a legal move in `max_retries`, the outer rollout decides what to
do (drop the decision, log, fallback). Baking a silent fallback in here would
mask exactly the failure mode Move 3 is designed to measure.
"""

from __future__ import annotations

from typing import Callable

from burl.harness.trace import BurlTrace, TurnStep


class RetryExhausted(RuntimeError):
    """Model failed to produce a legal play within the retry budget.

    The partial `BurlTrace` (with every turn the model actually produced) is
    attached as `.trace` so callers can still diagnose what went wrong. Re-raising
    is still the "loud failure" signal — we just hand the evidence up with it.
    """

    def __init__(self, message: str, trace: "BurlTrace | None" = None):
        super().__init__(message)
        self.trace = trace


StepFn = Callable[[BurlTrace, str | None], TurnStep]
"""Step callable: (trace, rejection_from_last_turn) -> next TurnStep.

The step function owns model calling and tool execution; this module owns only
the legality check and retry bookkeeping.
"""

LegalityFn = Callable[[int], tuple[bool, str]]


def retry_on_illegal(
    step_fn: StepFn,
    is_legal: LegalityFn,
    trace: BurlTrace,
    max_retries: int = 5,
) -> int:
    rejection: str | None = None
    for attempt in range(max_retries + 1):
        step = step_fn(trace, rejection)
        trace.turns.append(step)

        if step.committed_play is None:
            # Pure tool-use turn; not a retry, doesn't count against the budget.
            rejection = None
            continue

        ok, reason = is_legal(step.committed_play)
        if ok:
            trace.final_play = step.committed_play
            return step.committed_play

        step.engine_rejection = reason
        trace.n_retries += 1
        rejection = reason

    raise RetryExhausted(
        f"no legal play after {max_retries} retries "
        f"(state={trace.game_state_key}, turns={len(trace.turns)})",
        trace=trace,
    )
