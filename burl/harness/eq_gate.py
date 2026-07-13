"""EQ-gate — rejection-sampling mechanism for STaR trace collection.

Replaces the yes-biased rationalization step used through iter-1. The gate
fires on legal-but-sub-optimal commits, issues a feedback nudge that never
reveals the ground-truth play, and classifies the resulting re-attempt
chain as ``self_corrected`` / ``forced_flip`` / ``stubborn`` / ``exhausted``
(or ``converged_first_try`` if the rollout already matched bot).

All three public functions are pure; the gate harness integration lives in
``burl/eval/run_move4_star_rollout.py`` (design: see
``scratch/burl_p5_iter2_prep/eq_gate_design.md``; integration stub below).

Contract:
  ``check_commit(...)            -> GateDecision``
  ``gate_feedback_prompt(...)    -> str``
  ``classify_rationalization(..) -> Classification``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


FeedbackVariant = Literal["minimal", "tool-nudge", "social"]

Classification = Literal[
    "converged_first_try",
    "self_corrected",
    "forced_flip",
    "stubborn",
    "exhausted",
]


@dataclass(frozen=True)
class GateDecision:
    fire: bool
    reason: str
    eq_delta: float


@dataclass(frozen=True)
class AttemptSummary:
    """Per-attempt view the classifier operates on.

    Narrow on purpose — keeps ``classify_rationalization`` a pure function
    over primitives. Integrators build one of these per commit attempt
    from their own records (``_build_record`` fields map 1:1).
    """
    committed_play: int | None
    legal: bool
    eq: float | None
    retry_exhausted: bool


# --------------------------------------------------------------------------- #
# Gate firing                                                                  #
# --------------------------------------------------------------------------- #


def check_commit(
    committed_play: int | None,
    legal: bool,
    per_play_eq: dict[int, float],
    bot_play: int,
    bot_eq: float,
    *,
    eq_epsilon: float = 0.0,
) -> GateDecision:
    """Decide whether the EQ-gate should fire on this commit.

    Fires iff the commit is legal, present in ``per_play_eq``, and strictly
    below ``bot_eq`` by at least ``eq_epsilon``. All other paths (no commit,
    illegal commit, missing E[Q], K1 win or tie) yield ``fire=False``.

    ``eq_delta`` is ``burl_eq - bot_eq`` when computable, ``-inf`` when not —
    callers can still sort gated decisions by severity.
    """
    if committed_play is None:
        return GateDecision(False, "no_commit", float("-inf"))
    if not legal:
        return GateDecision(False, "illegal_commit", float("-inf"))
    burl_eq = per_play_eq.get(int(committed_play))
    if burl_eq is None:
        return GateDecision(False, "missing_burl_eq", float("-inf"))
    eq_delta = float(burl_eq) - float(bot_eq)
    if float(burl_eq) + float(eq_epsilon) >= float(bot_eq):
        return GateDecision(False, "burl_at_least_bot", eq_delta)
    return GateDecision(True, "burl_below_bot_by_eps", eq_delta)


# --------------------------------------------------------------------------- #
# Feedback prompts                                                             #
# --------------------------------------------------------------------------- #


_FEEDBACK_MINIMAL = (
    "Your commit was legal, but another of your legal plays has a higher "
    "expected-Q outcome. Take one more look: use your tools to compare "
    "legal plays, then commit again."
)

_FEEDBACK_TOOL_NUDGE = (
    "Your commit was legal, but another of your legal plays looks stronger "
    "under the E[Q] distribution. Before re-committing, call "
    "eq_outcome_distribution on each remaining legal option and compare "
    "mean, stdev, and p_make. You may also call conditional_outcome if a "
    "particular hidden hand would swing it. Then commit."
)

_FEEDBACK_SOCIAL = (
    "Think again — your partner needs this one. A different legal play in "
    "your hand has a higher expected contribution to count. Re-examine the "
    "position (count dominoes loose, trump structure, who is void in what) "
    "and commit to the play that best supports your team."
)

_FEEDBACK_BY_VARIANT: dict[str, str] = {
    "minimal": _FEEDBACK_MINIMAL,
    "tool-nudge": _FEEDBACK_TOOL_NUDGE,
    "social": _FEEDBACK_SOCIAL,
}


def gate_feedback_prompt(variant: FeedbackVariant, attempt_idx: int = 1) -> str:
    """Return the gate's feedback nudge for ``variant``.

    Yes-bias invariant: the string never names a specific domino id, never
    quotes a per-play E[Q] number, and never narrows the answer below two
    legal plays. Enforced by tests (see ``test_eq_gate.py``).

    ``attempt_idx`` (1-based) appends a brevity tail on second-and-later
    nudges so the same paragraph is not repeated verbatim in chat history.
    """
    if variant not in _FEEDBACK_BY_VARIANT:
        raise ValueError(
            f"unknown variant {variant!r}; "
            f"expected one of {sorted(_FEEDBACK_BY_VARIANT)}"
        )
    base = _FEEDBACK_BY_VARIANT[variant]
    if attempt_idx <= 1:
        return base
    return f"{base} (attempt {attempt_idx})"


# --------------------------------------------------------------------------- #
# Classifier                                                                   #
# --------------------------------------------------------------------------- #


def classify_rationalization(
    attempts: list[AttemptSummary],
    bot_play: int,
) -> Classification:
    """Classify a chain of commit attempts under EQ-gate feedback.

    ``attempts[0]`` is the unhinted rollout commit; ``attempts[1:]`` are
    gate-nudged re-attempts. The classifier does not know how many gate
    retries were budgeted — it reads the final attempt and the initial
    attempt and answers.

    Branches:
      - ``converged_first_try``: one attempt, matched bot_play.
      - ``self_corrected``:      >=2 attempts, final matched bot_play.
      - ``forced_flip``:         >=2 attempts, final != initial and != bot.
      - ``stubborn``:            >=2 attempts, final == initial and != bot.
      - ``exhausted``:           final attempt has no legal commit, OR a
                                 single attempt that did not match bot
                                 (no evidence of correction or stubbornness
                                 to report).
    """
    if not attempts:
        raise ValueError("classify_rationalization requires >=1 attempt")

    last = attempts[-1]
    if last.retry_exhausted or last.committed_play is None or not last.legal:
        return "exhausted"

    if len(attempts) == 1:
        if last.committed_play == bot_play:
            return "converged_first_try"
        return "exhausted"

    initial = attempts[0]
    if last.committed_play == bot_play:
        return "self_corrected"
    if initial.committed_play == last.committed_play:
        return "stubborn"
    return "forced_flip"


# --------------------------------------------------------------------------- #
# Integration stub — how the gate hooks into run_move4_star_rollout.py.       #
# --------------------------------------------------------------------------- #
#
# The current Phase B loop calls ``_rationalize_one`` which reveals the
# ground-truth play in the system prompt (see agent_runner_native.py:
# ``_RATIONALIZE_SUFFIX_TEMPLATE``). Iter-2 replaces that with the gate:
#
#   from burl.harness.eq_gate import (
#       AttemptSummary, check_commit, classify_rationalization,
#       gate_feedback_prompt,
#   )
#
#   gate_variant = "tool-nudge"     # iter-2 default — see design doc
#   max_gate_retries = 1            # iter-2 default — see design doc open Q1
#   eq_epsilon = 0.25               # iter-2 default — see design doc open Q5
#
#   for rec in records:
#       if rec["category"] != "legal_loss":
#           continue
#       decision = _decision_for(rec)           # existing lookup
#       gate = check_commit(
#           committed_play=rec["final_play"],
#           legal=rec["final_play_legal"],
#           per_play_eq=decision.per_play_eq,
#           bot_play=int(decision.bot_play),
#           bot_eq=float(decision.bot_eq),
#           eq_epsilon=eq_epsilon,
#       )
#       if not gate.fire:
#           continue                            # K1 pass / illegal / unresolvable
#
#       attempts = [AttemptSummary(
#           committed_play=rec["final_play"],
#           legal=rec["final_play_legal"],
#           eq=rec["burl_eq"],
#           retry_exhausted=bool(rec.get("retry_exhausted", False)),
#       )]
#       for gi in range(1, max_gate_retries + 1):
#           nudge = gate_feedback_prompt(gate_variant, attempt_idx=gi)
#           nudged_trace, exhausted = _run_with_feedback(
#               decision, model_fn, nudge, max_turns, max_retries,
#           )
#           nudged_rec = _build_record(decision, nudged_trace, exhausted, 0.0)
#           attempts.append(AttemptSummary(
#               committed_play=nudged_rec["final_play"],
#               legal=nudged_rec["final_play_legal"],
#               eq=nudged_rec["burl_eq"],
#               retry_exhausted=bool(nudged_rec.get("retry_exhausted", False)),
#           ))
#           if attempts[-1].committed_play == int(decision.bot_play):
#               break
#           if attempts[-1].retry_exhausted:
#               break
#
#       verdict = classify_rationalization(attempts, int(decision.bot_play))
#       gate_records.append({
#           "decision": decision,
#           "attempts": attempts,
#           "verdict": verdict,
#           "trace": nudged_trace,
#       })
#
#   # Phase C: only `self_corrected` verdicts are composed into the SFT
#   # corpus (tagged ``source="eq_gate_self_correct"``). ``forced_flip``
#   # and ``stubborn`` are dropped by default — see design doc open Q3.
#
# ``_run_with_feedback`` is a thin wrapper on ``NativeHarness.run`` that
# appends a ``role="user"`` message containing ``nudge`` after the initial
# user turn (or, equivalently, passes an ``extra_user_messages`` kwarg if
# we take the 3-line NativeHarness tweak). The harness itself does not
# need to know about the gate.
