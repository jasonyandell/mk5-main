from __future__ import annotations

import json
import random
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np

from burl.harness.agent_runner import _current_player
from burl.tools import engine as engine_tools
from burl.tools.eq_distribution import OutcomeDistribution
from burl.wax_museum.harness import run_decision_waxed
from burl.wax_museum.tools import (
    WaxContext,
    build_registry,
    precompute_tool_lattice,
)
from forge.zeb.game import apply_action, legal_actions, new_game


def _mid_game_state(seed: int = 2026):
    state = new_game(seed=seed, skip_bidding=True)
    rng = random.Random(seed)
    while len(state.play_history) < 20:
        slots = legal_actions(state)
        if not slots:
            break
        state = apply_action(state, rng.choice(slots))
    return state


def _legal_play(state) -> int:
    me = _current_player(state)
    for d in state.hands[me]:
        if d not in state.played and engine_tools.is_legal(state, d)[0]:
            return int(d)
    raise RuntimeError("no legal play")


def _dist(play: int, mean: float, shape: str = "bimodal") -> OutcomeDistribution:
    spikes = []
    modes = [{"center": mean - 8.0, "mass": 0.35}, {"center": mean + 5.0, "mass": 0.65}]
    if shape != "unimodal":
        spikes = [
            {
                "mode_center": mean - 8.0,
                "mode_mass": 0.35,
                "n_worlds_in_spike": 7,
                "catalysts": [
                    {
                        "seat": "left_opp",
                        "domino": 5,
                        "freq_in_spike": 0.8,
                        "baseline": 0.3,
                        "lift": 2.7,
                    }
                ],
            },
            {
                "mode_center": mean + 5.0,
                "mode_mass": 0.65,
                "n_worlds_in_spike": 13,
                "catalysts": [
                    {
                        "seat": "partner",
                        "domino": 11,
                        "freq_in_spike": 0.9,
                        "baseline": 0.25,
                        "lift": 3.6,
                    }
                ],
            },
        ]
    return OutcomeDistribution(
        play=int(play),
        pdf_bins=np.zeros(85, dtype=np.float32),
        mean=float(mean),
        stdev=4.0,
        p_make=0.55,
        n_samples=20,
        min_q=-12.0,
        max_q=18.0,
        percentiles={10: -6.0, 25: -2.0, 50: mean, 75: mean + 3.0, 90: mean + 6.0},
        is_offense=True,
        distribution_shape=shape,
        modes=modes,
        gap_between_modes=13.0,
        spike_drivers=spikes,
    )


def _patch_tool_backends() -> ExitStack:
    stack = ExitStack()

    def fake_eq(_game_state, play: int, **_kw):
        return _dist(play, mean=float(int(play) % 13) - 3.0)

    def fake_conditional(_game_state, play: int, assumption: dict, **_kw):
        seat = int(assumption["player"])
        mean = (float(int(play) % 13) - 3.0) + (2.0 if seat % 2 == 0 else -2.0)
        return _dist(play, mean=mean, shape="unimodal")

    def fake_belief(_game_state, **_kw):
        return {
            "prose": "BELIEF: deterministic fake payload",
            "posterior_by_domino": {"5": {"left": 0.8, "partner": 0.1, "right": 0.1}},
            "top_shifts_since_last": [],
            "gus_value_estimate": 1.25,
        }

    stack.enter_context(patch("burl.wax_museum.tools.eq_outcome_distribution", fake_eq))
    stack.enter_context(patch("burl.wax_museum.tools.conditional_outcome", fake_conditional))
    stack.enter_context(patch("burl.wax_museum.tools.call_belief_trajectory", fake_belief))
    return stack


def _json_equivalent(payload) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


def test_eager_tool_cache_matches_lazy_tool_payloads():
    state = _mid_game_state()
    me = _current_player(state)
    play = _legal_play(state)

    with _patch_tool_backends():
        lazy_ctx = WaxContext(game_state=state, me_abs=me, oracle=None)
        lazy = build_registry(lazy_ctx)

        eager_ctx = WaxContext(game_state=state, me_abs=me, oracle=None)
        precompute_tool_lattice(eager_ctx)
        eager = build_registry(eager_ctx, use_eager_cache=True)

        calls = [
            ("belief_trajectory", {}),
            ("explore_game", {"play": play}),
            ("probe_best_case", {"play": play}),
            ("probe_worst_case", {"play": play}),
            ("ask_rule", {"topic": "trick_winner"}),
        ]
        for name, args in calls:
            assert _json_equivalent(eager[name](**args)) == _json_equivalent(lazy[name](**args))

        stats = eager_ctx.tool_cache_stats.as_dict()
        assert stats["legal_play_count"] >= 1
        assert stats["precomputed_count"] >= 5
        assert stats["hit_count"] == len(calls)
        assert stats["saved_tool_wall_s"] >= 0.0


def test_eager_tool_cache_does_not_change_harness_result():
    state = _mid_game_state()
    target = _legal_play(state)

    def make_stub():
        script = iter([
            "Plan.\n" + "x" * 300 + f"\n<|tool_call>call:explore_game{{play:{target}}}<tool_call|>",
            f"Probe worst.\n<|tool_call>call:probe_worst_case{{play:{target}}}<tool_call|>",
            f"Commit.\n<|tool_call>call:commit_play{{domino_id:{target}}}<tool_call|>",
        ])

        def stub(_messages, _tool_schemas):
            return next(script)

        return stub

    with _patch_tool_backends():
        lazy = run_decision_waxed(state, make_stub(), max_turns=5, oracle=None)
        eager = run_decision_waxed(
            state, make_stub(), max_turns=5, oracle=None, eager_tool_cache=True
        )

    assert eager.trace.final_play == lazy.trace.final_play == target
    assert eager.tool_call_sequence == lazy.tool_call_sequence
    assert eager.probed == lazy.probed
    assert eager.tool_cache_stats["hit_count"] == 2
    assert eager.tool_cache_stats["wasted_count"] >= 0
