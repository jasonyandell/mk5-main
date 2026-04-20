"""Tests for ``burl.tools.meta_tools.what_would_change_my_mind``.

Three levels, ordered fastest to slowest:

1. ``test_rationale_and_tag`` — pure string helpers, no oracle.
2. ``test_registration`` — harness advertises the tool.
3. ``test_basic`` / ``test_shift_ordering`` / ``test_mean_consistency`` —
   one real oracle call on a held-out seed.

Run:
    source .venv/bin/activate && \
    python -u -m pytest burl/tools/test_meta_tools.py -v
"""

from __future__ import annotations

import random

import pytest

from burl.tools.meta_tools import (
    _format_rationale,
    _qualitative_tag,
    what_would_change_my_mind,
)


# --------------------------------------------------------------------------- #
# 1. Pure helpers                                                              #
# --------------------------------------------------------------------------- #


def test_rationale_and_tag() -> None:
    # Big positive shift -> "big shift".
    assert _qualitative_tag(0.0, 10.0, 10.0) == "big shift"
    # Tiny shift -> "confirms current outlook".
    assert _qualitative_tag(5.0, 5.2, 0.2) == "confirms current outlook"
    # Deep negative conditional mean AND big negative shift -> "disaster flag".
    assert _qualitative_tag(0.0, -15.0, -15.0) == "disaster flag"
    # Moderate shift -> "moderate shift".
    assert _qualitative_tag(0.0, 3.0, 3.0) == "moderate shift"

    # Format sanity: includes numbers and the tag.
    text = _format_rationale(0.0, 10.0, 10.0)
    assert "big shift" in text
    assert "+10.0" in text
    assert "flips mean from" in text


# --------------------------------------------------------------------------- #
# 2. Registration                                                              #
# --------------------------------------------------------------------------- #


def test_registration_xml_registry() -> None:
    """Tool must be in the XML-path registry."""
    from burl.harness.agent_runner import build_tool_registry

    registry = build_tool_registry()
    assert "what_would_change_my_mind" in registry


def test_registration_native_schema() -> None:
    """Tool schema must appear in the native-path TOOL_SCHEMAS."""
    from burl.harness.agent_runner_native import TOOL_SCHEMAS

    names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    assert "what_would_change_my_mind" in names

    schema = next(
        s for s in TOOL_SCHEMAS
        if s["function"]["name"] == "what_would_change_my_mind"
    )
    params = schema["function"]["parameters"]
    assert params["type"] == "object"
    assert "play" in params["properties"]
    assert "play" in params["required"]


# --------------------------------------------------------------------------- #
# 3. Live oracle integration                                                   #
# --------------------------------------------------------------------------- #


def _build_midgame_state(seed: int = 900013, n_plays: int = 12):
    """Roll forward a random game to give the meta-tool a non-trivial state."""
    from forge.zeb.game import apply_action, legal_actions, new_game

    state = new_game(seed=seed, skip_bidding=True)
    rng = random.Random(seed)
    for _ in range(n_plays):
        slots = legal_actions(state)
        if not slots:
            break
        state = apply_action(state, rng.choice(slots))
    return state


def _pick_legal_play(state) -> int:
    """First legal play for the current player."""
    from burl.tools.engine import is_legal
    from burl.tools.meta_tools import _me_abs

    me = _me_abs(state)
    hand = [d for d in state.hands[me] if d not in state.played]
    legal = [d for d in hand if is_legal(state, d)[0]]
    assert legal, "seed should have legal plays at this point"
    return int(legal[0])


@pytest.mark.slow
def test_basic() -> None:
    """Schema + top_k bound + per-assumption field shape."""
    from burl.tools.eq_distribution import load_eq_oracle

    state = _build_midgame_state()
    play = _pick_legal_play(state)

    oracle = load_eq_oracle()
    result = what_would_change_my_mind(
        state, play, n_samples_per_probe=5, top_k=5, oracle=oracle,
    )

    assert set(result.keys()) == {"play", "unconditional_mean", "assumptions"}
    assert result["play"] == play
    assert isinstance(result["unconditional_mean"], float)
    assert len(result["assumptions"]) <= 5

    for a in result["assumptions"]:
        assert set(a.keys()) == {
            "player", "holds", "conditional_mean", "shift", "rationale",
        }
        assert a["player"] in {"left_opp", "partner", "right_opp"}, a
        assert 0 <= a["holds"] < 28
        assert isinstance(a["conditional_mean"], float)
        assert isinstance(a["shift"], float)
        assert isinstance(a["rationale"], str) and a["rationale"]
        # Rationale should echo the numbers (mechanical).
        assert "flips mean from" in a["rationale"]


@pytest.mark.slow
def test_shift_ordering() -> None:
    """Assumptions returned must be sorted by |shift| descending."""
    from burl.tools.eq_distribution import load_eq_oracle

    state = _build_midgame_state()
    play = _pick_legal_play(state)

    oracle = load_eq_oracle()
    result = what_would_change_my_mind(
        state, play, n_samples_per_probe=5, top_k=5, oracle=oracle,
    )

    shifts = [abs(a["shift"]) for a in result["assumptions"]]
    assert shifts == sorted(shifts, reverse=True), shifts


@pytest.mark.slow
def test_identifies_impactful_assumption() -> None:
    """The top-ranked assumption's |shift| should be >10 Q on a mid-game
    state with a meaningfully ambiguous play — signalling the probe loop
    actually finds an assumption that would move the mean, not just noise.

    This is the Task-2 correctness check from ITER4_PLAN §2 Candidate C.
    """
    from burl.tools.eq_distribution import load_eq_oracle

    state = _build_midgame_state()
    play = _pick_legal_play(state)
    oracle = load_eq_oracle()
    result = what_would_change_my_mind(
        state, play, n_samples_per_probe=5, top_k=5, oracle=oracle,
    )
    assert result["assumptions"], "expected at least one reachable probe"
    top_shift = abs(result["assumptions"][0]["shift"])
    assert top_shift > 10.0, (
        f"top |shift|={top_shift:.2f} — probe loop only found noise-sized "
        f"swings. full result: {result}"
    )


@pytest.mark.slow
def test_mean_consistency() -> None:
    """The baseline ``unconditional_mean`` must match a direct
    ``eq_outcome_distribution`` call under the same n_samples.

    Both calls share the SAME ``_WorldSamplerMRV`` default seeding path —
    small-sample variance is the only source of disagreement, so we allow
    a generous tolerance. The goal is "same order of magnitude", not
    "identical to the last decimal".
    """
    from burl.tools.eq_distribution import eq_outcome_distribution, load_eq_oracle

    state = _build_midgame_state()
    play = _pick_legal_play(state)
    oracle = load_eq_oracle()

    meta = what_would_change_my_mind(
        state, play, n_samples_per_probe=5, top_k=3, oracle=oracle,
    )
    baseline = eq_outcome_distribution(
        state, play, n_samples=5, oracle=oracle, suggest_counterfactuals=False,
    )

    # Sampling is stochastic; tolerate a few Q-units.
    assert abs(meta["unconditional_mean"] - float(baseline.mean)) < 5.0, (
        meta["unconditional_mean"], baseline.mean,
    )
