from __future__ import annotations

import json
import math
from pathlib import Path

from forge.zeb.game import legal_actions

from w42.world_sampler_audit.audit import (
    FixtureSpec,
    advance_fixture,
    analytic_mrv_distribution,
    analyze_fixture,
    canonical_world,
    full_deal_for_world,
    rank_actions,
)


def test_analytic_mrv_is_uniform_when_every_assignment_is_symmetric() -> None:
    pool = (0, 1, 2, 3)
    hand_sizes = (2, 1, 1)
    all_candidates = sum(1 << domino_id for domino_id in pool)
    distribution, invalid = analytic_mrv_distribution(
        pool,
        hand_sizes,
        (all_candidates, all_candidates, all_candidates),
    )

    assert invalid == 0.0
    assert len(distribution) == 12
    assert all(math.isclose(probability, 1.0 / 12.0) for probability in distribution.values())


def test_analytic_mrv_detects_unequal_completion_counts() -> None:
    # Seat 0 can take A/B and seat 1 can take A/C.  If seat 0 takes A there
    # is one completion; if it takes B there are two.  MRV nevertheless gives
    # each first choice 1/2, producing probabilities 1/2, 1/4, 1/4 instead of
    # the exact-uniform 1/3, 1/3, 1/3.
    pool = (0, 1, 2)
    distribution, invalid = analytic_mrv_distribution(
        pool,
        (1, 1, 1),
        ((1 << 0) | (1 << 1), (1 << 0) | (1 << 2), (1 << 0) | (1 << 1) | (1 << 2)),
    )

    assert invalid == 0.0
    assert len(distribution) == 3
    assert sorted(distribution.values()) == [0.25, 0.25, 0.5]


def test_historical_fixture_preserves_defect_and_validates_repair() -> None:
    spec = FixtureSpec(
        name="historical",
        deal_seed=900013,
        n_plays=20,
        rollout_seed=900013,
    )
    result, _state, _worlds, _uniform = analyze_fixture(
        spec,
        n_samples=20_000,
        empirical_seed=20260711,
    )

    assert result["exact_world_count"] == 12
    assert math.isclose(result["analytic_mrv_dead_end_probability"], 1.0 / 3.0)
    assert math.isclose(result["analytic_mrv_invalid_mass"], 1.0 / 3.0)
    assert result["live_sampler_algorithm"] == "uniform-completion-dp-v1"
    assert result["empirical_invalid_fraction"] == 0.0
    assert not result["legacy_mrv_conformance"]["passes"]
    assert result["uniform_conformance"]["passes"]


def test_full_deal_reconstruction_restores_every_played_domino() -> None:
    spec = FixtureSpec("historical", 900013, 20, 900013)
    result, state, worlds, _uniform = analyze_fixture(
        spec,
        n_samples=100,
        empirical_seed=7,
    )
    assert result["exact_world_count"] > 0

    deal = full_deal_for_world(state, worlds[0])
    assert all(len(hand) == 7 for hand in deal)
    assert sorted(domino for hand in deal for domino in hand) == list(range(28))
    for player, domino_id in state.play_history:
        assert domino_id in deal[player]


def test_rank_actions_changes_only_world_weights() -> None:
    spec = FixtureSpec("historical", 900013, 20, 900013)
    _result, state, worlds, uniform = analyze_fixture(
        spec,
        n_samples=100,
        empirical_seed=7,
    )
    legal = tuple(int(slot) for slot in legal_actions(state))
    assert legal

    # Synthetic rows pin the reweighting arithmetic without loading a model.
    q_rows = {world: [0.0] * 7 for world in worlds}
    for index, world in enumerate(worlds):
        row = q_rows[world]
        for slot in legal:
            row[slot] = float(index + slot)
    mrv = dict(uniform)
    empirical = dict(uniform)
    ranked = rank_actions(state, worlds, q_rows, uniform, mrv, empirical)

    assert not ranked["argmax_flipped"]
    assert ranked["max_abs_action_shift_q"] < 1e-12
    assert all(
        abs(action["uniform_eq"] - action["analytic_mrv_eq"]) < 1e-12
        for action in ranked["actions"]
    )


def test_canonical_world_ignores_padding_and_slot_order() -> None:
    assert canonical_world([[3, 1, -1], [5], [4, 2]]) == (
        (1, 3),
        (5,),
        (2, 4),
    )


def test_reviewed_summary_is_self_consistent() -> None:
    summary_path = Path(__file__).with_name("summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["status"] == "complete"
    assert summary["verdict"]["uniform_sampler_claim"] == "falsified"
    assert summary["aggregate"]["live_sampler_matches_analytic_mrv"]
    assert not summary["aggregate"]["all_sampled_worlds_valid"]
    assert summary["verdict"]["argmax_flips_in_three_fixture_panel"] == 0
    historical = summary["fixtures"][0]
    assert historical["name"] == "historical-seed-900013-p20"
    assert math.isclose(historical["analytic_invalid_mass"], 1.0 / 3.0)
    assert historical["uniform_best"] == historical["mrv_best"] == "21"


def test_repair_summary_closes_exact_fixture_gate() -> None:
    repair_path = Path(__file__).with_name("repair_summary.json")
    repair = json.loads(repair_path.read_text(encoding="utf-8"))

    assert repair["status"] == "complete"
    assert repair["algorithm"] == "uniform-completion-dp-v1"
    exact = repair["exact_fixture_validation"]
    assert exact["all_live_outputs_valid"]
    assert exact["all_live_fixtures_match_uniform_within_seven_sigma"]
    assert exact["historical_seed_900013_p20"]["invalid_fraction"] == 0.0
    rejected = repair["rejected_intermediate"]
    assert rejected["algorithm"] == "uniform-rejection-v1"
    assert rejected["status"] == "rejected"
    hard = rejected["falsifying_state"]
    assert hard["exact_valid_assignment_count"] == 924
    assert hard["all_labeled_seat_partition_count"] == 17_153_136
    assert hard["worlds_found"] < hard["requested_worlds"]
    assert repair["direct_count_checks"]["unconstrained_7_7_7_root_count"] == 399_072_960
