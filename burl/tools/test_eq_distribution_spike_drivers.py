"""Tests for the spike-driver extension of ``eq_outcome_distribution``.

Spike drivers surface the (seat, domino) assignments that are empirically
over-represented in the worlds whose Q-values landed in a given mode. They
answer "when this play lands in the disaster branch, who tends to be holding
what?" — the data-grounded complement to the hypothetical
``suggested_counterfactuals`` probe.

Two fast unit tests exercise the pure helper; two slower integration tests
(marked ``slow``) run the full pipeline against the real Stage-1 oracle on
a held-out seed.

Run: ``source .venv/bin/activate && python -u -m pytest
burl/tools/test_eq_distribution_spike_drivers.py -v``
"""
from __future__ import annotations

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# 1. Synthetic bimodal: catalyst only in disaster worlds                       #
# --------------------------------------------------------------------------- #


def test_compute_spike_drivers_bimodal_catalyst() -> None:
    """Construct 10 synthetic worlds where:

    - 4 worlds have Q=-22 (disaster mode) and right_opp holds domino 21
    - 6 worlds have Q=+20 (winning mode) and right_opp holds domino 5

    Expect disaster spike's top catalyst to be (right_opp=3, domino=21) with
    freq_in_spike=1.0 and baseline=0.4, i.e. lift=2.5.

    Me is seat 0. With the standard seat convention
    (``_seat_label(abs_seat, me)``), absolute seat 3 maps to ``right_opp``.
    """
    from burl.tools.eq_distribution import _compute_spike_drivers

    me = 0
    disaster_world = {0: {0, 1}, 1: {10, 11}, 2: {20, 22}, 3: {21, 23}}
    winning_world = {0: {0, 1}, 1: {10, 11}, 2: {20, 22}, 3: {5, 23}}
    worlds = [disaster_world] * 4 + [winning_world] * 6
    q_values = np.array([-22.0] * 4 + [20.0] * 6, dtype=np.float32)
    modes = [
        {"center": 20.0, "mass": 0.6},     # dominant (winning)
        {"center": -22.0, "mass": 0.4},    # tail (disaster)
    ]

    out = _compute_spike_drivers(worlds, q_values, modes, me, shape="bimodal")
    assert len(out) == 2, out

    # Sorted by center ascending: disaster first, winning second.
    assert out[0]["mode_center"] == -22.0
    assert out[0]["n_worlds_in_spike"] == 4
    disaster_catalysts = out[0]["catalysts"]
    assert disaster_catalysts, "expected at least one catalyst for disaster spike"
    top = disaster_catalysts[0]
    assert top["seat"] == "right_opp"
    assert top["domino"] == 21
    assert top["freq_in_spike"] == pytest.approx(1.0, abs=1e-3)
    assert top["baseline"] == pytest.approx(0.4, abs=1e-3)
    assert top["lift"] == pytest.approx(2.5, abs=1e-3)

    # Winning spike should have its own catalyst (right_opp holds 5 in all
    # winning worlds but 0 disaster worlds).
    winning_catalysts = out[1]["catalysts"]
    winning_top_pairs = {(c["seat"], c["domino"]) for c in winning_catalysts}
    assert ("right_opp", 5) in winning_top_pairs


# --------------------------------------------------------------------------- #
# 2. Unimodal skip                                                             #
# --------------------------------------------------------------------------- #


def test_compute_spike_drivers_unimodal_returns_empty() -> None:
    """Unimodal distributions have no spike structure — spike_drivers must be
    an empty list regardless of world content."""
    from burl.tools.eq_distribution import _compute_spike_drivers

    worlds = [{0: {0}, 1: {10}, 2: {20}, 3: {21}}] * 8
    q_values = np.full(8, 5.0, dtype=np.float32)
    modes = [{"center": 5.0, "mass": 1.0}]

    out = _compute_spike_drivers(worlds, q_values, modes, me=0, shape="unimodal")
    assert out == []


# --------------------------------------------------------------------------- #
# 3. Me-seat catalysts are skipped                                             #
# --------------------------------------------------------------------------- #


def test_compute_spike_drivers_skips_self_seat() -> None:
    """The model already knows its own hand; spike-driver catalysts on the
    me-seat are not informative."""
    from burl.tools.eq_distribution import _compute_spike_drivers

    me = 2
    disaster_world = {me: {99}, 0: {0, 1}, 1: {10}, 3: {21}}
    winning_world = {me: {99}, 0: {0, 1}, 1: {10}, 3: {5}}
    worlds = [disaster_world] * 3 + [winning_world] * 5
    q_values = np.array([-20.0] * 3 + [20.0] * 5, dtype=np.float32)
    modes = [
        {"center": 20.0, "mass": 0.62},
        {"center": -20.0, "mass": 0.38},
    ]

    out = _compute_spike_drivers(worlds, q_values, modes, me=me, shape="bimodal")
    for spike in out:
        for cat in spike["catalysts"]:
            assert cat["seat"] != "self", cat
            assert cat["domino"] != 99, cat


# --------------------------------------------------------------------------- #
# 4. Small-spike filter: <_SPIKE_MIN_WORLDS worlds in a mode -> skipped        #
# --------------------------------------------------------------------------- #


def test_compute_spike_drivers_drops_tiny_modes() -> None:
    """A mode with fewer than ``_SPIKE_MIN_WORLDS`` (=2) worlds assigned to it
    is too small to produce a reliable catalyst list and should be skipped."""
    from burl.tools.eq_distribution import _compute_spike_drivers

    me = 0
    worlds = [
        {0: {0}, 1: {10}, 2: {20}, 3: {21}},      # disaster (Q=-20), only 1
        {0: {0}, 1: {10}, 2: {20}, 3: {5}},       # winning
        {0: {0}, 1: {10}, 2: {20}, 3: {5}},       # winning
        {0: {0}, 1: {10}, 2: {20}, 3: {5}},       # winning
    ]
    q_values = np.array([-20.0, 20.0, 20.0, 20.0], dtype=np.float32)
    modes = [
        {"center": 20.0, "mass": 0.75},
        {"center": -20.0, "mass": 0.25},
    ]
    out = _compute_spike_drivers(worlds, q_values, modes, me=me, shape="bimodal")
    # Disaster spike has 1 world -> dropped. Winning spike has 3 -> kept only
    # if it produces at least one over-baseline catalyst; right_opp=5 appears
    # in 3/3 winning worlds and 0/1 disaster worlds -> baseline=0.75,
    # freq_in_spike=1.0, lift~1.33 which is BELOW the 1.5 threshold, so it's
    # filtered too. Expect empty output.
    centers = [s["mode_center"] for s in out]
    assert -20.0 not in centers, f"tiny disaster spike leaked through: {out}"


# --------------------------------------------------------------------------- #
# 5. Dataclass default                                                         #
# --------------------------------------------------------------------------- #


def test_outcome_distribution_spike_drivers_default_empty() -> None:
    """``OutcomeDistribution.spike_drivers`` defaults to []."""
    from burl.tools.eq_distribution import N_BINS, OutcomeDistribution

    od = OutcomeDistribution(
        play=0,
        pdf_bins=np.zeros(N_BINS, dtype=np.float32),
        mean=0.0,
        stdev=0.0,
        p_make=0.0,
        n_samples=0,
        min_q=0.0,
        max_q=0.0,
        percentiles={10: 0.0, 25: 0.0, 50: 0.0, 75: 0.0, 90: 0.0},
        is_offense=True,
    )
    assert od.spike_drivers == []


# --------------------------------------------------------------------------- #
# 6. Integration — real oracle at seed 900013                                  #
# --------------------------------------------------------------------------- #


def _advance_state(seed: int, n_plays: int):
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
def test_integration_spike_drivers_on_enumerated_state() -> None:
    """Enumerate path at trick 6 — run eq_outcome_distribution on every legal
    play and confirm spike_drivers is:

    - Always a list.
    - Empty for unimodal distributions.
    - Non-empty only when the distribution is bi/multimodal AND at least one
      mode hosts enough worlds to pass the catalyst filter.
    - Each catalyst entry has the expected keys with valid ranges.

    This is the 'nuts and bolts' validation on real data: if we don't see
    any bimodal play across the legal set, the integration run prints the
    per-play shape so we can diagnose, but still passes (the unit tests
    already cover the bimodal pathway directly).
    """
    from burl.tools.engine import is_legal
    from burl.tools.eq_distribution import eq_outcome_distribution, load_eq_oracle

    oracle = load_eq_oracle()
    state = _advance_state(seed=900013, n_plays=20)
    me = (state.trick_leader + len(state.current_trick)) % 4
    my_hand = [d for d in state.hands[me] if d not in state.played]
    legal = [d for d in my_hand if is_legal(state, d)[0]]
    assert legal, "seed 900013 @ n_plays=20 should yield legal plays"

    shapes_seen = []
    spike_driver_hit = False

    for play in legal:
        od = eq_outcome_distribution(
            state, play, oracle=oracle, enumerate=True,
            suggest_counterfactuals=False,
        )
        shapes_seen.append((play, od.distribution_shape, len(od.spike_drivers)))

        assert isinstance(od.spike_drivers, list)

        if od.distribution_shape == "unimodal":
            assert od.spike_drivers == [], (
                f"play={play}: unimodal should have empty spike_drivers, got "
                f"{od.spike_drivers}"
            )
            continue

        # Non-unimodal: may or may not surface catalysts (depends on whether
        # any catalyst clears the filter). When it does, schema must be
        # strict.
        for spike in od.spike_drivers:
            assert {"mode_center", "mode_mass", "n_worlds_in_spike",
                    "catalysts"} <= spike.keys()
            assert isinstance(spike["catalysts"], list)
            for cat in spike["catalysts"]:
                assert {"seat", "domino", "freq_in_spike", "baseline",
                        "lift"} <= cat.keys()
                assert cat["seat"] in {"left_opp", "partner", "right_opp"}, (
                    cat
                )
                assert 0 <= cat["domino"] < 28, cat
                assert 0.0 <= cat["freq_in_spike"] <= 1.0, cat
                assert 0.0 <= cat["baseline"] <= 1.0, cat
                assert cat["lift"] >= 1.5 or cat["lift"] == float("inf"), cat
                spike_driver_hit = True

    print(f"[spike-drivers seed 900013 trick-6] per-play: {shapes_seen} "
          f"any_catalysts_hit={spike_driver_hit}")


@pytest.mark.slow
def test_integration_spike_drivers_disabled_in_conditional() -> None:
    """``conditional_outcome`` must not populate ``spike_drivers`` — the
    distribution is already restricted to an assumption, so per-spike
    catalyst analysis on that slice is circular.
    """
    from burl.tools.engine import is_legal
    from burl.tools.eq_distribution import (
        ConditionUnreachable,
        conditional_outcome,
        load_eq_oracle,
    )

    oracle = load_eq_oracle()
    state = _advance_state(seed=900013, n_plays=20)
    me = (state.trick_leader + len(state.current_trick)) % 4
    my_hand = [d for d in state.hands[me] if d not in state.played]
    legal = [d for d in my_hand if is_legal(state, d)[0]]
    assert legal

    # Try a handful of {player, holds} assumptions until we find one that's
    # reachable — we don't need a particular outcome, just a non-crashing
    # conditional call to check the disabled-drivers invariant.
    for play in legal:
        for opp_offset in (1, 2, 3):
            seat = (me + opp_offset) % 4
            for dom in range(28):
                if dom in my_hand or dom in state.played:
                    continue
                try:
                    od = conditional_outcome(
                        state, play,
                        {"player": seat, "holds": dom},
                        n_samples=10, max_sampling_tries=5,
                        oracle=oracle,
                    )
                except (ConditionUnreachable, ValueError):
                    continue
                assert od.spike_drivers == [], (
                    "conditional_outcome must not populate spike_drivers; "
                    f"got {od.spike_drivers}"
                )
                return

    pytest.skip("No reachable conditional assumption found at seed 900013 "
                "trick 6 — not a failure, just a thin held-out slice.")
