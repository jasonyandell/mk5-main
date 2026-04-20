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


# --------------------------------------------------------------------------- #
# 5. enumerate=auto / True / False                                             #
# --------------------------------------------------------------------------- #


def _advance_state(seed: int, n_plays: int):
    """Helper: make a state with ``n_plays`` plays already on the table."""
    import random as _r
    from forge.zeb.game import apply_action, legal_actions, new_game

    state = new_game(seed=seed, skip_bidding=True)
    rng = _r.Random(seed)
    for _ in range(n_plays):
        slots = legal_actions(state)
        if not slots:
            break
        state = apply_action(state, rng.choice(slots))
    return state


@pytest.mark.slow
def test_enumerate_true_marks_sampling_mode() -> None:
    """At trick 6 (pool ~= 6) ``enumerate=True`` should return
    ``sampling_mode='enumerated'`` and ``n_samples`` equal to the exact
    enumerated world count."""
    from burl.tools.eq_distribution import (
        _enumerated_worlds_tensor,
        _state_to_game_state_tensor,
        eq_outcome_distribution,
        load_eq_oracle,
    )
    from burl.tools.engine import is_legal
    import torch

    oracle = load_eq_oracle()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Trick 6 starts after 20 plays. Advance to 21 so we're at position 2 in
    # trick 6 (pool stays very small). With 20 plays the pool size is
    # typically <= 8.
    state = _advance_state(seed=900013, n_plays=20)
    me = (state.trick_leader + len(state.current_trick)) % 4
    my_hand = [d for d in state.hands[me] if d not in state.played]
    legal = [d for d in my_hand if is_legal(state, d)[0]]
    assert legal, "smoke seed should have legal plays at trick 6"
    play = legal[0]

    # Cross-check: enumerated worlds tensor shape matches what we'd expect
    # from raw enumerate_worlds_cpu output.
    gst = _state_to_game_state_tensor(state, device)
    worlds_tensor, raw = _enumerated_worlds_tensor(gst, state, device)
    assert worlds_tensor.shape[1] == len(raw)
    assert worlds_tensor.shape[1] > 0, "trick-6 state should have >=1 world"

    od = eq_outcome_distribution(
        state, play, n_samples=10, oracle=oracle, enumerate=True,
    )
    assert od.sampling_mode == "enumerated"
    assert od.n_samples == len(raw)


@pytest.mark.slow
def test_enumerate_true_rejects_oversized_pool() -> None:
    """``enumerate=True`` must refuse at the start of the game when the pool
    is the full 21 dominoes (>> _ENUMERATE_HARD_POOL_CAP)."""
    from burl.tools.eq_distribution import eq_outcome_distribution, load_eq_oracle
    from burl.tools.engine import is_legal

    oracle = load_eq_oracle()
    state = _advance_state(seed=900013, n_plays=0)  # start of hand
    me = (state.trick_leader + len(state.current_trick)) % 4
    my_hand = [d for d in state.hands[me] if d not in state.played]
    legal = [d for d in my_hand if is_legal(state, d)[0]]
    assert legal
    play = legal[0]

    with pytest.raises(ValueError, match="enumerate=True refused"):
        eq_outcome_distribution(
            state, play, n_samples=10, oracle=oracle, enumerate=True,
        )


@pytest.mark.slow
def test_enumerate_auto_picks_correctly_by_pool_size() -> None:
    """At pool_size <= 12 'auto' should enumerate; at pool_size >= 15 it
    should sample. Use trick 6 (small pool) and trick 1 (large pool)."""
    from burl.tools.eq_distribution import eq_outcome_distribution, load_eq_oracle
    from burl.tools.engine import is_legal

    oracle = load_eq_oracle()

    # Trick 6 (20 plays on table) -> pool <= 8.
    state_late = _advance_state(seed=900013, n_plays=20)
    me = (state_late.trick_leader + len(state_late.current_trick)) % 4
    my_hand = [d for d in state_late.hands[me] if d not in state_late.played]
    legal = [d for d in my_hand if is_legal(state_late, d)[0]]
    assert legal
    play = legal[0]

    od_late = eq_outcome_distribution(
        state_late, play, n_samples=10, oracle=oracle, enumerate="auto",
    )
    assert od_late.sampling_mode == "enumerated"

    # Trick 1 (~0 plays) -> pool ~= 21.
    state_early = _advance_state(seed=900013, n_plays=0)
    me = (state_early.trick_leader + len(state_early.current_trick)) % 4
    my_hand = [d for d in state_early.hands[me] if d not in state_early.played]
    legal = [d for d in my_hand if is_legal(state_early, d)[0]]
    assert legal
    play = legal[0]

    od_early = eq_outcome_distribution(
        state_early, play, n_samples=10, oracle=oracle, enumerate="auto",
    )
    assert od_early.sampling_mode == "sampled"


@pytest.mark.slow
def test_enumerate_is_deterministic_and_sensible() -> None:
    """Enumerate=True must be deterministic (same state, same result) and
    produce a PDF that sums to 1.0 with a mean inside the enumerated Q
    range.

    We do NOT compare against the sampler here: WorldSamplerMRV biases
    toward the most-constrained draws, so even at trick 6 with only ~12
    unique worlds the sampled mean can differ from the uniform
    (enumerated) mean by a few Q. Enumeration is the ground truth; the
    sampler is the estimator. See SESSION_NOTES 2026-04-19.
    """
    from burl.tools.eq_distribution import (
        Q_VALUES,
        eq_outcome_distribution,
        load_eq_oracle,
    )
    from burl.tools.engine import is_legal

    oracle = load_eq_oracle()
    state = _advance_state(seed=900013, n_plays=20)
    me = (state.trick_leader + len(state.current_trick)) % 4
    my_hand = [d for d in state.hands[me] if d not in state.played]
    legal = [d for d in my_hand if is_legal(state, d)[0]]
    assert legal
    play = legal[0]

    od1 = eq_outcome_distribution(
        state, play, oracle=oracle, enumerate=True,
        suggest_counterfactuals=False,
    )
    od2 = eq_outcome_distribution(
        state, play, oracle=oracle, enumerate=True,
        suggest_counterfactuals=False,
    )
    # Determinism: same enumerated worlds -> identical mean and PDF.
    assert od1.sampling_mode == "enumerated"
    assert od1.n_samples == od2.n_samples
    assert abs(od1.mean - od2.mean) < 1e-5
    assert (abs(od1.pdf_bins - od2.pdf_bins) < 1e-6).all()

    # PDF sums to 1.0, mean inside Q range.
    import numpy as np
    assert abs(od1.pdf_bins.sum() - 1.0) < 1e-5
    assert -42.0 <= od1.mean <= 42.0
    # PDF-implied mean matches the stored mean to <1 Q (bin rounding).
    implied = float((od1.pdf_bins * Q_VALUES).sum())
    assert abs(implied - od1.mean) < 0.6


@pytest.mark.slow
def test_enumerate_auto_benchmark() -> None:
    """Informational: print enumerate-auto latency at trick 6 (should be
    <20ms per call on M5 Max CPU). Fails only if it blows past 200ms (well
    clear of the sampling baseline of ~6ms, but a safety net for regressions).
    """
    import time
    from burl.tools.eq_distribution import eq_outcome_distribution, load_eq_oracle
    from burl.tools.engine import is_legal

    oracle = load_eq_oracle()
    state = _advance_state(seed=900013, n_plays=20)
    me = (state.trick_leader + len(state.current_trick)) % 4
    my_hand = [d for d in state.hands[me] if d not in state.played]
    legal = [d for d in my_hand if is_legal(state, d)[0]]
    assert legal
    play = legal[0]

    # Benchmark the raw enumerate path without the (slow, recursive)
    # counterfactual probe — that probe reruns conditional_outcome N times
    # and isn't what "sampling vs enumeration" is trying to measure.
    # Warm up.
    _ = eq_outcome_distribution(
        state, play, oracle=oracle, enumerate="auto",
        suggest_counterfactuals=False,
    )
    # Measure.
    N = 5
    t0 = time.perf_counter()
    for _ in range(N):
        od = eq_outcome_distribution(
            state, play, oracle=oracle, enumerate="auto",
            suggest_counterfactuals=False,
        )
    dt_ms = 1000.0 * (time.perf_counter() - t0) / N
    print(
        f"[enumerate-auto trick-6 bench] n_worlds={od.n_samples} "
        f"mode={od.sampling_mode} avg_ms={dt_ms:.1f}"
    )
    assert od.sampling_mode == "enumerated"
    # Target: <20ms per call on M5 Max CPU. 100ms ceiling for stragglers.
    assert dt_ms < 100.0, f"regression: enumerate-auto took {dt_ms:.0f} ms"
