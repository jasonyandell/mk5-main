"""Tests for the candlewax extension of ``eq_outcome_distribution``.

Four tests, ordered fastest -> slowest:

1. ``test_detect_modes_unimodal`` — synthetic PDF, pure helper.
2. ``test_detect_modes_bimodal`` — synthetic PDF, pure helper.
3. ``test_outcome_distribution_dataclass_defaults`` — schema / backward compat.
4. ``test_integration_real_oracle`` — one real oracle call on a held-out seed,
   asserts the candlewax fields are populated and internally consistent.

Run: ``source .venv/bin/activate && python -u -m pytest burl/tools/test_eq_distribution_candlewax.py -v``
"""
from __future__ import annotations

import numpy as np
import pytest

from burl.tools.eq_distribution import (
    N_BINS,
    OutcomeDistribution,
    Q_VALUES,
    _detect_modes,
    _rationale_for_mode,
    _seat_label,
)


# --------------------------------------------------------------------------- #
# 1. Unimodal synthetic PDF                                                    #
# --------------------------------------------------------------------------- #


def test_detect_modes_unimodal() -> None:
    """A tight Gaussian-ish bump around Q=0 should be unimodal."""
    pdf = np.zeros(N_BINS, dtype=np.float32)
    center = 42          # bin 42 -> Q = 0
    for offset, w in zip(range(-4, 5), [0.02, 0.05, 0.10, 0.18, 0.30, 0.18, 0.10, 0.05, 0.02]):
        pdf[center + offset] = w
    pdf /= pdf.sum()

    shape, modes = _detect_modes(pdf)
    assert shape == "unimodal", f"expected unimodal, got {shape}"
    assert len(modes) == 1
    assert abs(modes[0]["center"] - 0.0) < 1.0, modes[0]
    # Mass in window ±3 around the peak should capture the bulk of the bump.
    assert modes[0]["mass"] > 0.8


# --------------------------------------------------------------------------- #
# 2. Bimodal synthetic PDF                                                    #
# --------------------------------------------------------------------------- #


def test_detect_modes_bimodal() -> None:
    """Two well-separated peaks should be labelled bimodal with the correct
    centers and mass ordering."""
    pdf = np.zeros(N_BINS, dtype=np.float32)
    # Positive mode at bin 65 -> Q = 23. Heavier.
    for offset, w in zip(range(-3, 4), [0.03, 0.06, 0.12, 0.20, 0.12, 0.06, 0.03]):
        pdf[65 + offset] = w
    # Negative mode at bin 20 -> Q = -22. Lighter.
    for offset, w in zip(range(-3, 4), [0.01, 0.03, 0.06, 0.10, 0.06, 0.03, 0.01]):
        pdf[20 + offset] = w
    pdf /= pdf.sum()

    shape, modes = _detect_modes(pdf)
    assert shape == "bimodal", f"expected bimodal, got {shape}"
    assert len(modes) == 2
    # Top mode is the heavier positive one.
    assert modes[0]["center"] > 0.0
    assert modes[1]["center"] < 0.0
    assert modes[0]["mass"] > modes[1]["mass"]
    gap = abs(modes[0]["center"] - modes[1]["center"])
    assert gap > 30.0, f"expected wide gap, got {gap}"


def test_rationale_text_is_direction_aware() -> None:
    # Top (dominant) mode — rationale should "confirm" the scenario.
    top_pos = _rationale_for_mode(22.9, mode_mass=0.6, is_top_mode=True)
    top_neg = _rationale_for_mode(-18.2, mode_mass=0.6, is_top_mode=True)
    assert "confirms" in top_pos and "winning" in top_pos, top_pos
    assert "confirms" in top_neg and "losing" in top_neg, top_neg

    # Non-top (tail) mode — rationale should "collapse" the tail.
    tail_pos = _rationale_for_mode(22.9, mode_mass=0.3, is_top_mode=False)
    tail_neg = _rationale_for_mode(-18.2, mode_mass=0.3, is_top_mode=False)
    assert "collapses" in tail_pos and "tail" in tail_pos, tail_pos
    assert "collapses" in tail_neg and "tail" in tail_neg, tail_neg

    # Near-zero sanity (no crash, non-empty string).
    assert isinstance(_rationale_for_mode(0.0, 0.5, True), str)
    assert _rationale_for_mode(0.0, 0.5, False)


def test_seat_label_mapping() -> None:
    # me=0: left_opp=1, partner=2, right_opp=3.
    assert _seat_label(1, me=0) == "left_opp"
    assert _seat_label(2, me=0) == "partner"
    assert _seat_label(3, me=0) == "right_opp"
    # me=2: wrap-around.
    assert _seat_label(3, me=2) == "left_opp"
    assert _seat_label(0, me=2) == "partner"
    assert _seat_label(1, me=2) == "right_opp"


# --------------------------------------------------------------------------- #
# 3. Schema / backward compat                                                  #
# --------------------------------------------------------------------------- #


def test_outcome_distribution_dataclass_defaults() -> None:
    """Adding candlewax fields must not break construction with only the old
    positional args — they default to unimodal / empty."""
    pdf = np.zeros(N_BINS, dtype=np.float32)
    pdf[42] = 1.0
    od = OutcomeDistribution(
        play=7,
        pdf_bins=pdf,
        mean=0.0,
        stdev=0.0,
        p_make=0.5,
        n_samples=10,
        min_q=-1.0,
        max_q=1.0,
        percentiles={10: 0.0, 25: 0.0, 50: 0.0, 75: 0.0, 90: 0.0},
        is_offense=True,
    )
    assert od.distribution_shape == "unimodal"
    assert od.modes == []
    assert od.gap_between_modes == 0.0
    assert od.suggested_counterfactuals == []


# --------------------------------------------------------------------------- #
# 4. Integration: one real oracle call                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_integration_real_oracle() -> None:
    """Exercise the full path on a held-out seed. Asserts:

    - candlewax fields are present on the returned dataclass,
    - modes (if any) have centers inside the Q range,
    - gap_between_modes is consistent with the top-2 mode centers,
    - if shape != unimodal, suggested_counterfactuals returns at most 2
      ``{player, holds, rationale}`` dicts with valid seat labels.
    """
    import random as _r
    from burl.tools.eq_distribution import eq_outcome_distribution, load_eq_oracle
    from burl.tools.engine import is_legal
    from forge.zeb.game import apply_action, legal_actions, new_game

    oracle = load_eq_oracle()
    seed = 900013
    state = new_game(seed=seed, skip_bidding=True)
    rng = _r.Random(seed)
    for _ in range(12):
        slots = legal_actions(state)
        if not slots:
            break
        state = apply_action(state, rng.choice(slots))

    me = (state.trick_leader + len(state.current_trick)) % 4
    my_hand = [d for d in state.hands[me] if d not in state.played]
    legal = [d for d in my_hand if is_legal(state, d)[0]]
    assert legal, "smoke seed should have legal plays at trick 3"

    play = legal[0]
    od = eq_outcome_distribution(state, play, n_samples=10, oracle=oracle)

    # Schema sanity.
    assert od.distribution_shape in {"unimodal", "bimodal", "multimodal"}
    for m in od.modes:
        assert -42.0 <= m["center"] <= 42.0, m
        assert 0.0 <= m["mass"] <= 1.0 + 1e-6, m
    if len(od.modes) >= 2:
        expected_gap = abs(od.modes[0]["center"] - od.modes[1]["center"])
        assert abs(od.gap_between_modes - expected_gap) < 1e-6
    else:
        assert od.gap_between_modes == 0.0

    for cf in od.suggested_counterfactuals:
        assert set(cf.keys()) == {"player", "holds", "rationale"}, cf
        assert cf["player"] in {"left_opp", "partner", "right_opp"}, cf
        assert 0 <= cf["holds"] < 28, cf
        assert isinstance(cf["rationale"], str) and cf["rationale"], cf

    assert len(od.suggested_counterfactuals) <= 2
