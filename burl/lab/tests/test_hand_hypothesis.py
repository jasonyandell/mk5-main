from __future__ import annotations

import sys
from types import SimpleNamespace

from burl.lab.tools import hand_hypothesis


def test_simulate_hand_impact_accepts_burls_invented_aliases() -> None:
    play, seat, holds = hand_hypothesis._extract_args(
        {
            "play": 19,
            "opponent_focus": "Seat 2",
            "hypothetical_hand_to_test": {"hand_id": 12},
        }
    )

    assert play == 19
    assert seat == "Seat 2"
    assert holds == 12


def test_simulate_hand_impact_renders_probability_and_conditional_shift(monkeypatch) -> None:
    baseline = SimpleNamespace(
        mean=-13.9,
        p_make=0.55,
        stdev=22.8,
        distribution_shape="multimodal",
        n_samples=20,
    )
    conditional = SimpleNamespace(
        mean=29.0,
        p_make=0.95,
        stdev=4.0,
        distribution_shape="unimodal",
        n_samples=20,
    )

    class FakeConditionUnreachable(RuntimeError):
        pass

    def fake_conditional_outcome(game_state, *, play, assumption, n_samples, max_sampling_tries, oracle):
        assert play == 19
        assert assumption == {"player": 2, "holds": 12}
        assert n_samples == 20
        assert max_sampling_tries == 100
        assert oracle == "oracle"
        assert game_state == "state"
        return conditional

    fake_eq_module = SimpleNamespace(
        ConditionUnreachable=FakeConditionUnreachable,
        conditional_outcome=fake_conditional_outcome,
    )
    monkeypatch.setitem(sys.modules, "burl.tools.eq_distribution", fake_eq_module)
    monkeypatch.setattr(
        hand_hypothesis,
        "_belief_probability",
        lambda ctx, abs_seat, holds: (
            0.403,
            {"source": "test", "relative_seat": "left_opp"},
        ),
    )

    ctx = SimpleNamespace(
        me_abs=1,
        oracle="oracle",
        game_state="state",
        get_or_build=lambda play: SimpleNamespace(dist=baseline),
    )

    result = hand_hypothesis.SIMULATE_HAND_IMPACT.impl(
        ctx, {"play_id": 19, "seat": "left_opp", "holds": 12}
    )

    assert "Plausibility: 40% (medium)" in result.evidence["prose"]
    assert "shift:       +42.9 Q" in result.evidence["prose"]
    structured = result.evidence["structured"]
    assert structured["hypothesis"]["seat_abs"] == 2
    assert structured["hypothesis"]["holds_label"] == "4-2"
    assert structured["shift_in_mean"] == 42.9
    assert structured["probability_weighted_shift"] == 17.29
