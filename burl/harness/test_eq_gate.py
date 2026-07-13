"""Unit tests for ``burl.harness.eq_gate``.

Covers every branch of ``check_commit`` and ``classify_rationalization``
plus the yes-bias invariants on ``gate_feedback_prompt``. Tests are pure —
no harness, no Modal, no game state.
"""

from __future__ import annotations

import pytest

from burl.harness.eq_gate import (
    AttemptSummary,
    GateDecision,
    check_commit,
    classify_rationalization,
    gate_feedback_prompt,
)


# --------------------------------------------------------------------------- #
# check_commit                                                                 #
# --------------------------------------------------------------------------- #


def test_check_commit_fires_on_legal_loss():
    per_play_eq = {14: -5.4, 21: +10.3}
    gate = check_commit(
        committed_play=14, legal=True, per_play_eq=per_play_eq,
        bot_play=21, bot_eq=10.3,
    )
    assert gate.fire is True
    assert gate.reason == "burl_below_bot_by_eps"
    assert gate.eq_delta == pytest.approx(-15.7)


def test_check_commit_skips_k1_win():
    per_play_eq = {14: +12.0, 21: +10.3}
    gate = check_commit(
        committed_play=14, legal=True, per_play_eq=per_play_eq,
        bot_play=21, bot_eq=10.3,
    )
    assert gate.fire is False
    assert gate.reason == "burl_at_least_bot"
    assert gate.eq_delta == pytest.approx(+1.7)


def test_check_commit_ties_skip_under_default_epsilon():
    per_play_eq = {14: +10.3, 21: +10.3}
    gate = check_commit(
        committed_play=14, legal=True, per_play_eq=per_play_eq,
        bot_play=21, bot_eq=10.3,
    )
    assert gate.fire is False
    assert gate.reason == "burl_at_least_bot"


def test_check_commit_epsilon_raises_threshold():
    per_play_eq = {14: +10.1, 21: +10.3}
    gate_zero_eps = check_commit(
        committed_play=14, legal=True, per_play_eq=per_play_eq,
        bot_play=21, bot_eq=10.3,
    )
    assert gate_zero_eps.fire is True

    gate_quarter_eps = check_commit(
        committed_play=14, legal=True, per_play_eq=per_play_eq,
        bot_play=21, bot_eq=10.3, eq_epsilon=0.25,
    )
    assert gate_quarter_eps.fire is False
    assert gate_quarter_eps.reason == "burl_at_least_bot"


def test_check_commit_illegal_commit_does_not_fire():
    per_play_eq = {14: -5.4, 21: +10.3}
    gate = check_commit(
        committed_play=14, legal=False, per_play_eq=per_play_eq,
        bot_play=21, bot_eq=10.3,
    )
    assert gate.fire is False
    assert gate.reason == "illegal_commit"
    assert gate.eq_delta == float("-inf")


def test_check_commit_no_commit_does_not_fire():
    gate = check_commit(
        committed_play=None, legal=False, per_play_eq={21: 10.3},
        bot_play=21, bot_eq=10.3,
    )
    assert gate.fire is False
    assert gate.reason == "no_commit"


def test_check_commit_missing_per_play_eq_does_not_fire():
    gate = check_commit(
        committed_play=14, legal=True, per_play_eq={21: 10.3},
        bot_play=21, bot_eq=10.3,
    )
    assert gate.fire is False
    assert gate.reason == "missing_burl_eq"
    assert gate.eq_delta == float("-inf")


# --------------------------------------------------------------------------- #
# gate_feedback_prompt                                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("variant", ["minimal", "tool-nudge", "social"])
def test_feedback_prompt_returns_nonempty(variant):
    out = gate_feedback_prompt(variant)
    assert isinstance(out, str)
    assert len(out) >= 40


def test_feedback_prompts_are_distinct():
    variants = ["minimal", "tool-nudge", "social"]
    outputs = {v: gate_feedback_prompt(v) for v in variants}
    assert len(set(outputs.values())) == len(variants)


@pytest.mark.parametrize("variant", ["minimal", "tool-nudge", "social"])
def test_feedback_prompt_never_leaks_bot_play(variant):
    prompt = gate_feedback_prompt(variant)
    # No variant may contain any specific domino id or "domino_id=" anchor.
    for did in range(28):
        assert f"domino_id={did}" not in prompt
        assert f"play({did})" not in prompt
        assert f"commit_play({did})" not in prompt
    assert "domino_id=" not in prompt
    assert "bot_play" not in prompt
    assert "correct play" not in prompt
    assert "right play" not in prompt


def test_feedback_prompt_attempt_idx_tail():
    first = gate_feedback_prompt("minimal", attempt_idx=1)
    second = gate_feedback_prompt("minimal", attempt_idx=2)
    third = gate_feedback_prompt("minimal", attempt_idx=3)
    assert first == gate_feedback_prompt("minimal")  # default is attempt 1
    assert "(attempt 2)" in second
    assert "(attempt 3)" in third
    assert first != second


def test_feedback_prompt_unknown_variant_raises():
    with pytest.raises(ValueError, match="unknown variant"):
        gate_feedback_prompt("shouty", attempt_idx=1)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# classify_rationalization                                                     #
# --------------------------------------------------------------------------- #


def _attempt(play: int | None, *, legal: bool = True, exhausted: bool = False,
             eq: float | None = 0.0) -> AttemptSummary:
    return AttemptSummary(
        committed_play=play, legal=legal, eq=eq, retry_exhausted=exhausted,
    )


def test_classify_converged_first_try():
    attempts = [_attempt(21)]
    assert classify_rationalization(attempts, bot_play=21) == "converged_first_try"


def test_classify_single_attempt_nonmatch_is_exhausted():
    # Single attempt where the classifier has no re-try evidence collapses
    # to "exhausted" by convention (caller should have gated but didn't).
    attempts = [_attempt(14)]
    assert classify_rationalization(attempts, bot_play=21) == "exhausted"


def test_classify_self_corrected():
    attempts = [_attempt(14), _attempt(21)]
    assert classify_rationalization(attempts, bot_play=21) == "self_corrected"


def test_classify_self_corrected_through_intermediate_flip():
    attempts = [_attempt(14), _attempt(7), _attempt(21)]
    assert classify_rationalization(attempts, bot_play=21) == "self_corrected"


def test_classify_forced_flip():
    attempts = [_attempt(14), _attempt(7)]
    assert classify_rationalization(attempts, bot_play=21) == "forced_flip"


def test_classify_stubborn():
    attempts = [_attempt(14), _attempt(14)]
    assert classify_rationalization(attempts, bot_play=21) == "stubborn"


def test_classify_exhausted_retry_on_last_attempt():
    attempts = [_attempt(14), _attempt(None, legal=False, exhausted=True)]
    assert classify_rationalization(attempts, bot_play=21) == "exhausted"


def test_classify_exhausted_illegal_last_commit():
    # Last attempt produced a commit but the engine rejected it after the
    # retry loop gave up: legal=False without exhausted flag still counts
    # as exhausted for gate purposes (can't teach from a protocol failure).
    attempts = [_attempt(14), _attempt(99, legal=False)]
    assert classify_rationalization(attempts, bot_play=21) == "exhausted"


def test_classify_empty_list_raises():
    with pytest.raises(ValueError, match=">=1 attempt"):
        classify_rationalization([], bot_play=21)


def test_gate_decision_is_immutable_dataclass():
    gd = GateDecision(fire=True, reason="test", eq_delta=-1.0)
    with pytest.raises(Exception):
        gd.fire = False  # type: ignore[misc]


def test_attempt_summary_is_immutable_dataclass():
    a = _attempt(14)
    with pytest.raises(Exception):
        a.committed_play = 21  # type: ignore[misc]
