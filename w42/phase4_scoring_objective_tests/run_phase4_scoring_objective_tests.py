#!/usr/bin/env python3
"""Phase-4 scoring-objective and tournament tests for W42 Chapter 10 claims.

This runner extends the earlier deterministic scoring transform with generated
hand and match traces. It is deliberately conservative: trace rows are evidence
about the scoring objectives and the bundled heuristic policy population, not a
claim that the policies are optimal Texas 42 play.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forge.oracle.declarations import DECL_ID_TO_NAME, N_DECLS
from forge.oracle.rng import deal_from_seed
from forge.oracle.tables import (
    DOMINO_COUNT_POINTS,
    DOMINOES,
    can_follow,
    led_suit_for_lead_domino,
    resolve_trick,
    trick_rank,
)
from forge.zeb.types import BidState, GamePhase, ZebGameState
from forge.zeb.game import apply_action, current_player, legal_actions

OUT_DIR = ROOT / "w42/phase4_scoring_objective_tests"
BEAD_ID = "t42-br7n.3"
POLICIES = ("weak_count_dumper", "random", "control_greedy", "count_saver")
POLICY_PAIRS = (
    ("control_greedy", "random"),
    ("control_greedy", "weak_count_dumper"),
    ("count_saver", "random"),
    ("random", "weak_count_dumper"),
)
CLAIM_IDS = (
    "ch10-score-mode-objective",
    "ch10-early-terminal-under-marks",
    "ch10-nonbidder-partial-points-erased",
    "ch10-set-severity-compression",
    "ch10-special-bid-mark-multiplier",
    "ch10-low-bid-score-distortion",
    "ch10-point-system-skill-signal",
    "ch10-tournament-speed-tradeoff",
    "ch10-timed-marks-advancement-objective",
)


@dataclass(frozen=True)
class HandResult:
    seed: int
    matchup: str
    team0_policy: str
    team1_policy: str
    bidder: int
    bidder_team: int
    bid: int
    bid_class: str
    decl_id: int
    decl_name: str
    team0_points: int
    team1_points: int
    bidder_points: int
    defender_points: int
    made: bool
    point_team0_score: int
    point_team1_score: int
    mark_team0_score: int
    mark_team1_score: int
    mark_multiplier: int
    early_terminal_trick: int
    early_terminal_status: str
    tricks_saved: int
    partial_defender_points_erased: int
    set_severity_points: int
    full_play_history: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--hands-per-pair", type=int, default=1000)
    parser.add_argument("--match-count", type=int, default=160)
    parser.add_argument("--timed-trials", type=int, default=160)
    parser.add_argument("--timed-trick-budget", type=int, default=84)
    parser.add_argument("--base-seed", type=int, default=2026050303)
    parser.add_argument("--example-limit", type=int, default=6)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def git_status_short() -> str:
    try:
        return subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def dom_label(domino_id: int) -> str:
    high, low = DOMINOES[domino_id]
    return f"{high}-{low}"


def mark_multiplier(bid: int) -> int:
    if bid >= 84:
        return bid // 42
    return 1


def bid_class(bid: int) -> str:
    if bid < 42:
        return "ordinary"
    if bid == 42:
        return "all_42"
    return "mark_ladder"


def made_contract(bid: int, bidder_points: int) -> bool:
    if bid < 42:
        return bidder_points >= bid
    return bidder_points == 42


def score_hand_points(bid: int, bidder_team: int, team_points: tuple[int, int]) -> tuple[int, int]:
    bidder_points = team_points[bidder_team]
    defender_team = 1 - bidder_team
    defender_points = team_points[defender_team]
    made = made_contract(bid, bidder_points)
    scores = [0, 0]
    if bid < 42:
        if made:
            scores[bidder_team] = bidder_points
            scores[defender_team] = defender_points
        else:
            scores[bidder_team] = 0
            scores[defender_team] = defender_points + bid
    else:
        if made:
            scores[bidder_team] = bid
            scores[defender_team] = 0
        else:
            scores[bidder_team] = 0
            scores[defender_team] = bid
    return scores[0], scores[1]


def score_hand_marks(bid: int, bidder_team: int, team_points: tuple[int, int]) -> tuple[int, int]:
    scores = [0, 0]
    winner_team = bidder_team if made_contract(bid, team_points[bidder_team]) else 1 - bidder_team
    scores[winner_team] = mark_multiplier(bid)
    return scores[0], scores[1]


def initial_state(seed: int, bidder: int, bid: int, decl_id: int) -> ZebGameState:
    hands = tuple(tuple(hand) for hand in deal_from_seed(seed))
    return ZebGameState(
        hands=hands,
        dealer=0,
        phase=GamePhase.PLAYING,
        bid_state=BidState(bids=(0, 0, 0, 0), high_bidder=bidder, high_bid=bid),
        decl_id=decl_id,
        bidder=bidder,
        played=frozenset(),
        play_history=(),
        current_trick=(),
        trick_leader=bidder,
        team_points=(0, 0),
    )


def candidate_snapshot(state: ZebGameState, slot: int, player: int) -> dict[str, Any]:
    domino_id = state.hands[player][slot]
    count_points = DOMINO_COUNT_POINTS[domino_id]
    current = state.current_trick + (domino_id,)
    completes = len(current) == 4
    winner_team_now: int | None = None
    would_win_complete = False
    current_trick_count = sum(DOMINO_COUNT_POINTS[d] for d in current)

    if state.current_trick:
        led_suit = led_suit_for_lead_domino(state.current_trick[0], state.decl_id)
        ranks = [trick_rank(d, led_suit, state.decl_id) for d in current]
        best_offset = max(range(len(ranks)), key=lambda idx: ranks[idx])
        winner_team_now = ((state.trick_leader + best_offset) % 4) % 2
    else:
        winner_team_now = player % 2

    if completes:
        outcome = resolve_trick(current[0], current, state.decl_id)
        winner = (state.trick_leader + outcome.winner_offset) % 4
        winner_team_now = winner % 2
        would_win_complete = winner == player

    return {
        "slot": slot,
        "domino_id": domino_id,
        "label": dom_label(domino_id),
        "count_points": count_points,
        "completes_trick": completes,
        "would_win_complete": would_win_complete,
        "winner_team_now": winner_team_now,
        "current_trick_count": current_trick_count,
        "rank_key": _candidate_rank_key(state, domino_id),
    }


def _candidate_rank_key(state: ZebGameState, domino_id: int) -> int:
    if not state.current_trick:
        led_suit = led_suit_for_lead_domino(domino_id, state.decl_id)
    else:
        led_suit = led_suit_for_lead_domino(state.current_trick[0], state.decl_id)
    return trick_rank(domino_id, led_suit, state.decl_id)


def choose_action(state: ZebGameState, policy: str, rng: random.Random) -> int:
    player = current_player(state)
    legal = list(legal_actions(state))
    if not legal:
        raise ValueError("no legal actions")
    if policy == "random":
        return rng.choice(legal)

    snaps = [candidate_snapshot(state, slot, player) for slot in legal]
    team = player % 2
    if policy == "weak_count_dumper":
        return max(snaps, key=lambda s: (s["count_points"], -s["rank_key"], -s["domino_id"]))["slot"]

    if policy == "count_saver":
        def score_saver(s: dict[str, Any]) -> tuple[float, int]:
            team_controls = s["winner_team_now"] == team
            value = 0.0
            if team_controls:
                value += 6.0 + s["current_trick_count"]
            else:
                value -= 2.0 * s["count_points"]
            value -= 0.08 * s["rank_key"]
            return value, -s["domino_id"]

        return max(snaps, key=score_saver)["slot"]

    if policy == "control_greedy":
        def score_control(s: dict[str, Any]) -> tuple[float, int]:
            team_controls = s["winner_team_now"] == team
            value = 0.0
            if team_controls:
                value += 10.0 + (1.5 * s["current_trick_count"]) + (0.05 * s["rank_key"])
            else:
                value -= 2.5 * s["count_points"]
                value -= 0.03 * s["rank_key"]
            if s["completes_trick"] and s["would_win_complete"]:
                value += 4.0
            return value, -s["domino_id"]

        return max(snaps, key=score_control)["slot"]

    raise ValueError(f"unknown policy: {policy}")


def mark_terminal_status(
    bid: int,
    bidder_team: int,
    team_points: tuple[int, int],
    completed_tricks: int,
) -> tuple[str, int]:
    bidder_points = team_points[bidder_team]
    defender_points = team_points[1 - bidder_team]
    captured_total = team_points[0] + team_points[1]
    remaining_total = 42 - captured_total
    if bid < 42:
        if bidder_points >= bid:
            return "made", 7 - completed_tricks
        if bidder_points + remaining_total < bid:
            return "set", 7 - completed_tricks
        return "live", 0
    if defender_points > 0:
        return "set", 7 - completed_tricks
    if captured_total == 42 and bidder_points == 42:
        return "made", 0
    return "live", 0


def deterministic_contract(seed: int) -> tuple[int, int, int]:
    rng = random.Random(seed ^ 0x42C10)
    bidder = rng.randrange(4)
    roll = rng.random()
    if roll < 0.74:
        bid = rng.randint(30, 41)
    elif roll < 0.90:
        bid = 42
    elif roll < 0.97:
        bid = 84
    elif roll < 0.992:
        bid = 126
    else:
        bid = 168
    decl_id = rng.randrange(N_DECLS)
    return bidder, bid, decl_id


def play_hand(seed: int, team0_policy: str, team1_policy: str) -> HandResult:
    bidder, bid, decl_id = deterministic_contract(seed)
    state = initial_state(seed, bidder=bidder, bid=bid, decl_id=decl_id)
    rng = random.Random(seed ^ 0xBADC0DE)
    early_trick = 7
    early_status = "full_hand"
    tricks_saved = 0

    while state.phase == GamePhase.PLAYING:
        player = current_player(state)
        policy = team0_policy if player % 2 == 0 else team1_policy
        before_tricks = len(state.play_history) // 4
        action = choose_action(state, policy, rng)
        state = apply_action(state, action)
        after_tricks = len(state.play_history) // 4
        if after_tricks > before_tricks and early_status == "full_hand":
            status, saved = mark_terminal_status(state.bid_state.high_bid, state.bidder % 2, state.team_points, after_tricks)
            if status != "live":
                early_trick = after_tricks
                early_status = status
                tricks_saved = saved

    team_points = state.team_points
    bidder_team = bidder % 2
    bidder_points = team_points[bidder_team]
    defender_points = team_points[1 - bidder_team]
    made = made_contract(bid, bidder_points)
    point_scores = score_hand_points(bid, bidder_team, team_points)
    mark_scores = score_hand_marks(bid, bidder_team, team_points)
    point_by_team = [0, 0]
    point_by_team[0], point_by_team[1] = point_scores
    marks_by_team = [0, 0]
    marks_by_team[0], marks_by_team[1] = mark_scores
    history = " ".join(f"P{p}:{dom_label(d)}" for p, d in state.play_history)
    set_severity = 0
    if not made:
        set_severity = bid if bid >= 42 else bid + defender_points

    return HandResult(
        seed=seed,
        matchup=f"{team0_policy}_vs_{team1_policy}",
        team0_policy=team0_policy,
        team1_policy=team1_policy,
        bidder=bidder,
        bidder_team=bidder_team,
        bid=bid,
        bid_class=bid_class(bid),
        decl_id=decl_id,
        decl_name=DECL_ID_TO_NAME[decl_id],
        team0_points=team_points[0],
        team1_points=team_points[1],
        bidder_points=bidder_points,
        defender_points=defender_points,
        made=made,
        point_team0_score=point_scores[0],
        point_team1_score=point_scores[1],
        mark_team0_score=mark_scores[0],
        mark_team1_score=mark_scores[1],
        mark_multiplier=mark_multiplier(bid),
        early_terminal_trick=early_trick,
        early_terminal_status=early_status,
        tricks_saved=tricks_saved,
        partial_defender_points_erased=(defender_points if made and bid < 42 else 0),
        set_severity_points=set_severity,
        full_play_history=history,
    )


def hand_to_row(hand: HandResult) -> dict[str, Any]:
    return {
        "seed": hand.seed,
        "matchup": hand.matchup,
        "team0_policy": hand.team0_policy,
        "team1_policy": hand.team1_policy,
        "bidder": hand.bidder,
        "bidder_team": hand.bidder_team,
        "bid": hand.bid,
        "bid_class": hand.bid_class,
        "decl_id": hand.decl_id,
        "decl_name": hand.decl_name,
        "team0_points": hand.team0_points,
        "team1_points": hand.team1_points,
        "bidder_points": hand.bidder_points,
        "defender_points": hand.defender_points,
        "made": hand.made,
        "point_team0_score": hand.point_team0_score,
        "point_team1_score": hand.point_team1_score,
        "mark_team0_score": hand.mark_team0_score,
        "mark_team1_score": hand.mark_team1_score,
        "mark_multiplier": hand.mark_multiplier,
        "early_terminal_trick": hand.early_terminal_trick,
        "early_terminal_status": hand.early_terminal_status,
        "tricks_saved": hand.tricks_saved,
        "partial_defender_points_erased": hand.partial_defender_points_erased,
        "set_severity_points": hand.set_severity_points,
        "full_play_history": hand.full_play_history,
    }


def generated_hands(args: argparse.Namespace) -> list[HandResult]:
    hands_per_pair = 80 if args.smoke else args.hands_per_pair
    rows: list[HandResult] = []
    for pair_idx, (team0_policy, team1_policy) in enumerate(POLICY_PAIRS):
        for offset in range(hands_per_pair):
            seed = args.base_seed + (pair_idx * 10_000_000) + offset
            rows.append(play_hand(seed, team0_policy, team1_policy))
    return rows


def summarize_early_terminal(hands: list[HandResult]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str], list[HandResult]] = defaultdict(list)
    for hand in hands:
        buckets[(hand.matchup, hand.bid_class, hand.early_terminal_status)].append(hand)
    rows: list[dict[str, Any]] = []
    for (matchup, bclass, status), bucket in sorted(buckets.items()):
        rows.append(
            {
                "matchup": matchup,
                "bid_class": bclass,
                "early_terminal_status": status,
                "hands": len(bucket),
                "mean_terminal_trick": round(mean(h.early_terminal_trick for h in bucket), 6),
                "mean_tricks_saved": round(mean(h.tricks_saved for h in bucket), 6),
                "saved_trick_rate": round(sum(h.tricks_saved for h in bucket) / (7 * len(bucket)), 6),
            }
        )
    return rows


def summarize_objective_compression(hands: list[HandResult]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, bool], list[HandResult]] = defaultdict(list)
    for hand in hands:
        buckets[(hand.bid_class, hand.made)].append(hand)
    rows: list[dict[str, Any]] = []
    for (bclass, made), bucket in sorted(buckets.items()):
        erased = [h.partial_defender_points_erased for h in bucket if h.partial_defender_points_erased > 0]
        set_severity = [h.set_severity_points for h in bucket if h.set_severity_points > 0]
        rows.append(
            {
                "bid_class": bclass,
                "made": made,
                "hands": len(bucket),
                "partial_erasure_hands": len(erased),
                "partial_erasure_rate": round(len(erased) / len(bucket), 6),
                "mean_erased_defender_points": round(mean(erased), 6) if erased else 0.0,
                "set_severity_hands": len(set_severity),
                "distinct_set_severity_values": len(set(set_severity)),
                "min_set_severity_points": min(set_severity) if set_severity else 0,
                "max_set_severity_points": max(set_severity) if set_severity else 0,
                "mark_values_in_bucket": ",".join(str(v) for v in sorted({h.mark_multiplier for h in bucket})),
            }
        )
    return rows


def summarize_policy_population(hands: list[HandResult]) -> list[dict[str, Any]]:
    buckets: dict[str, list[HandResult]] = defaultdict(list)
    for hand in hands:
        buckets[hand.matchup].append(hand)
    rows: list[dict[str, Any]] = []
    for matchup, bucket in sorted(buckets.items()):
        point_net = [h.point_team0_score - h.point_team1_score for h in bucket]
        mark_net = [h.mark_team0_score - h.mark_team1_score for h in bucket]
        point_wins = sum(1 for v in point_net if v > 0)
        mark_wins = sum(1 for v in mark_net if v > 0)
        sign_disagreements = sum(1 for p, m in zip(point_net, mark_net, strict=True) if (p > 0) != (m > 0))
        rows.append(
            {
                "matchup": matchup,
                "hands": len(bucket),
                "mean_point_net_team0": round(mean(point_net), 6),
                "mean_mark_net_team0": round(mean(mark_net), 6),
                "point_team0_win_rate": round(point_wins / len(bucket), 6),
                "mark_team0_win_rate": round(mark_wins / len(bucket), 6),
                "point_mark_hand_winner_disagreement_rate": round(sign_disagreements / len(bucket), 6),
                "point_net_std": round(stddev(point_net), 6),
                "mark_net_std": round(stddev(mark_net), 6),
            }
        )
    return rows


def stddev(values: list[float] | list[int]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return math.sqrt(sum((v - center) ** 2 for v in values) / (len(values) - 1))


def point_match_winner(scores: list[int], last_bidder_team: int) -> int | None:
    if scores[0] < 250 and scores[1] < 250:
        return None
    if scores[0] >= 250 and scores[1] >= 250:
        return last_bidder_team
    return 0 if scores[0] >= 250 else 1


def simulate_match(seed: int, team0_policy: str, team1_policy: str) -> dict[str, Any]:
    point_scores = [0, 0]
    mark_scores = [0, 0]
    point_done: dict[str, Any] | None = None
    mark_done: dict[str, Any] | None = None
    hand_idx = 0
    full_tricks_seen = 0
    mark_played_tricks_seen = 0
    while (point_done is None or mark_done is None) and hand_idx < 80:
        hand = play_hand(seed + hand_idx, team0_policy, team1_policy)
        full_tricks_seen += 7
        if mark_done is None:
            mark_played_tricks_seen += 7 - hand.tricks_saved
        if point_done is None:
            point_scores[0] += hand.point_team0_score
            point_scores[1] += hand.point_team1_score
            winner = point_match_winner(point_scores, hand.bidder_team)
            if winner is not None:
                point_done = {
                    "point_match_hands": hand_idx + 1,
                    "point_match_full_tricks": full_tricks_seen,
                    "point_match_winner": winner,
                    "point_score_team0": point_scores[0],
                    "point_score_team1": point_scores[1],
                }
        if mark_done is None:
            mark_scores[0] += hand.mark_team0_score
            mark_scores[1] += hand.mark_team1_score
            if mark_scores[0] >= 7 or mark_scores[1] >= 7:
                winner = 0 if mark_scores[0] >= 7 else 1
                mark_done = {
                    "mark_match_hands": hand_idx + 1,
                    "mark_match_played_tricks_early_stop": mark_played_tricks_seen,
                    "mark_match_full_trick_equivalent": (hand_idx + 1) * 7,
                    "mark_match_winner": winner,
                    "mark_score_team0": mark_scores[0],
                    "mark_score_team1": mark_scores[1],
                }
        hand_idx += 1
    if point_done is None:
        point_done = {
            "point_match_hands": hand_idx,
            "point_match_full_tricks": full_tricks_seen,
            "point_match_winner": -1,
            "point_score_team0": point_scores[0],
            "point_score_team1": point_scores[1],
        }
    if mark_done is None:
        mark_done = {
            "mark_match_hands": hand_idx,
            "mark_match_played_tricks_early_stop": mark_played_tricks_seen,
            "mark_match_full_trick_equivalent": hand_idx * 7,
            "mark_match_winner": -1,
            "mark_score_team0": mark_scores[0],
            "mark_score_team1": mark_scores[1],
        }
    return {
        "seed": seed,
        "matchup": f"{team0_policy}_vs_{team1_policy}",
        **point_done,
        **mark_done,
        "mark_vs_point_winner_disagree": point_done["point_match_winner"] != mark_done["mark_match_winner"],
        "mark_tricks_saved_vs_full_mark_match": mark_done["mark_match_full_trick_equivalent"]
        - mark_done["mark_match_played_tricks_early_stop"],
    }


def simulate_matches(args: argparse.Namespace) -> list[dict[str, Any]]:
    match_count = 16 if args.smoke else args.match_count
    rows: list[dict[str, Any]] = []
    for pair_idx, (team0_policy, team1_policy) in enumerate(POLICY_PAIRS):
        for match_idx in range(match_count):
            seed = args.base_seed + 100_000_000 + (pair_idx * 1_000_000) + (match_idx * 1000)
            rows.append(simulate_match(seed, team0_policy, team1_policy))
    return rows


def summarize_matches(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row["matchup"]].append(row)
    summary_rows: list[dict[str, Any]] = []
    for matchup, bucket in sorted(buckets.items()):
        summary_rows.append(
            {
                "matchup": matchup,
                "matches": len(bucket),
                "mean_point_match_hands": round(mean(row["point_match_hands"] for row in bucket), 6),
                "mean_mark_match_hands": round(mean(row["mark_match_hands"] for row in bucket), 6),
                "mean_point_full_tricks": round(mean(row["point_match_full_tricks"] for row in bucket), 6),
                "mean_mark_played_tricks_early_stop": round(
                    mean(row["mark_match_played_tricks_early_stop"] for row in bucket), 6
                ),
                "mean_mark_tricks_saved_vs_full_mark_match": round(
                    mean(row["mark_tricks_saved_vs_full_mark_match"] for row in bucket), 6
                ),
                "winner_disagreement_rate": round(
                    sum(bool(row["mark_vs_point_winner_disagree"]) for row in bucket) / len(bucket), 6
                ),
            }
        )
    return summary_rows


def simulate_timed_trials(args: argparse.Namespace) -> list[dict[str, Any]]:
    trial_count = 16 if args.smoke else args.timed_trials
    policy_order = list(POLICIES)
    pairings = ((0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2))
    rows: list[dict[str, Any]] = []
    for trial in range(trial_count):
        rng = random.Random(args.base_seed + 200_000_000 + trial)
        rng.shuffle(policy_order)
        mark_scores = [0, 0, 0, 0]
        point_scores = [0, 0, 0, 0]
        hands_played = [0, 0, 0, 0]
        tricks_used = [0, 0, 0, 0]
        for round_idx, (a, b) in enumerate(pairings):
            budget = args.timed_trick_budget
            used = 0
            hand_idx = 0
            while used < budget:
                seed = args.base_seed + 300_000_000 + (trial * 1_000_000) + (round_idx * 50_000) + hand_idx
                hand = play_hand(seed, policy_order[a], policy_order[b])
                played_tricks = 7 - hand.tricks_saved
                if used + played_tricks > budget:
                    break
                used += played_tricks
                hands_played[a] += 1
                hands_played[b] += 1
                tricks_used[a] += played_tricks
                tricks_used[b] += played_tricks
                mark_scores[a] += hand.mark_team0_score
                mark_scores[b] += hand.mark_team1_score
                point_scores[a] += hand.point_team0_score
                point_scores[b] += hand.point_team1_score
                hand_idx += 1

        top_by_marks = max(range(4), key=lambda idx: (mark_scores[idx], point_scores[idx], -idx))
        top_by_points = max(range(4), key=lambda idx: (point_scores[idx], mark_scores[idx], -idx))
        rows.append(
            {
                "trial": trial,
                "timed_trick_budget_per_pairing": args.timed_trick_budget,
                "team0_policy": policy_order[0],
                "team1_policy": policy_order[1],
                "team2_policy": policy_order[2],
                "team3_policy": policy_order[3],
                "marks": "|".join(str(v) for v in mark_scores),
                "points": "|".join(str(v) for v in point_scores),
                "hands_played": "|".join(str(v) for v in hands_played),
                "tricks_used": "|".join(str(v) for v in tricks_used),
                "top_by_marks": top_by_marks,
                "top_by_points": top_by_points,
                "top_by_marks_policy": policy_order[top_by_marks],
                "top_by_points_policy": policy_order[top_by_points],
                "advancement_disagreement": top_by_marks != top_by_points,
                "mark_leader_point_rank": rank_of(top_by_marks, point_scores),
                "point_leader_mark_rank": rank_of(top_by_points, mark_scores),
            }
        )
    return rows


def rank_of(index: int, scores: list[int]) -> int:
    ordered = sorted(range(len(scores)), key=lambda idx: (scores[idx], -idx), reverse=True)
    return ordered.index(index) + 1


def summarize_timed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    disagreements = [row for row in rows if row["advancement_disagreement"]]
    mark_rank = [int(row["mark_leader_point_rank"]) for row in rows]
    point_rank = [int(row["point_leader_mark_rank"]) for row in rows]
    return [
        {
            "timed_trials": len(rows),
            "advancement_disagreement_trials": len(disagreements),
            "advancement_disagreement_rate": round(len(disagreements) / len(rows), 6) if rows else 0.0,
            "mean_mark_leader_point_rank": round(mean(mark_rank), 6) if mark_rank else 0.0,
            "mean_point_leader_mark_rank": round(mean(point_rank), 6) if point_rank else 0.0,
        }
    ]


def deterministic_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    terminal_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    for bid in list(range(30, 42)) + [42, 84, 126, 168]:
        for bidder_points in range(43):
            made = made_contract(bid, bidder_points)
            team_points = (bidder_points, 42 - bidder_points)
            point_scores = score_hand_points(bid, 0, team_points)
            mark_scores = score_hand_marks(bid, 0, team_points)
            terminal_rows.append(
                {
                    "bid": bid,
                    "bid_class": bid_class(bid),
                    "bidder_capture_points": bidder_points,
                    "defender_capture_points": 42 - bidder_points,
                    "made": made,
                    "point_bidder_score": point_scores[0],
                    "point_defender_score": point_scores[1],
                    "point_net": point_scores[0] - point_scores[1],
                    "mark_multiplier": mark_multiplier(bid),
                    "mark_bidder_score": mark_scores[0],
                    "mark_defender_score": mark_scores[1],
                    "mark_net": mark_scores[0] - mark_scores[1],
                    "partial_defender_points_erased": 42 - bidder_points if made and bid < 42 else 0,
                    "set_severity_points": 0 if made else (bid if bid >= 42 else bid + (42 - bidder_points)),
                }
            )
    for bid in [30, 35, 41, 42, 84, 126, 168]:
        threshold_rows.append(
            {
                "bid": bid,
                "bid_class": bid_class(bid),
                "mark_multiplier": mark_multiplier(bid),
                "mark_break_even_make_probability": 0.5,
                "ordinary_point_objective_note": "point threshold depends on defender retained points"
                if bid < 42
                else "symmetric sweep bid under deterministic score transform",
            }
        )
    return terminal_rows, threshold_rows


def claim_summary_rows(
    hands: list[HandResult],
    match_summary: list[dict[str, Any]],
    timed_summary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    early = [h for h in hands if h.tricks_saved > 0]
    ordinary_made = [h for h in hands if h.bid_class == "ordinary" and h.made]
    partial = [h for h in ordinary_made if h.partial_defender_points_erased > 0]
    ordinary_sets = [h for h in hands if h.bid_class == "ordinary" and not h.made]
    mark_match_trick_mean = mean(row["mean_mark_played_tricks_early_stop"] for row in match_summary)
    point_match_trick_mean = mean(row["mean_point_full_tricks"] for row in match_summary)
    timed_disagreement = timed_summary[0]["advancement_disagreement_rate"] if timed_summary else 0.0
    rows = [
        {
            "claim_id": "ch10-score-mode-objective",
            "status": "supported",
            "evidence": "deterministic terminal transform plus generated hands carry divergent point and mark labels",
            "metric": "generated_hands",
            "value": len(hands),
            "caveat": "does not prove objective-aware optimal policies",
        },
        {
            "claim_id": "ch10-early-terminal-under-marks",
            "status": "supported",
            "evidence": "generated traces reached mark-terminal states before trick 7",
            "metric": "early_terminal_rate",
            "value": round(len(early) / len(hands), 6),
            "caveat": "trace frequency depends on synthetic bid mix and heuristic policies",
        },
        {
            "claim_id": "ch10-nonbidder-partial-points-erased",
            "status": "supported",
            "evidence": "made ordinary contracts with defender points score zero defender marks",
            "metric": "partial_erasure_rate_among_made_ordinary",
            "value": round(len(partial) / len(ordinary_made), 6) if ordinary_made else 0.0,
            "caveat": "terminal accounting evidence, not live-state chase/avoidance advice",
        },
        {
            "claim_id": "ch10-set-severity-compression",
            "status": "supported",
            "evidence": "ordinary failed bids collapse distinct point penalties into one mark",
            "metric": "distinct_ordinary_set_severity_values",
            "value": len({h.set_severity_points for h in ordinary_sets}),
            "caveat": "high-mark bids still scale by multiplier",
        },
        {
            "claim_id": "ch10-special-bid-mark-multiplier",
            "status": "supported",
            "evidence": "deterministic multiplier table covers 84/126/168",
            "metric": "max_multiplier_tested",
            "value": 4,
            "caveat": "bid selection and make probability remain separate questions",
        },
        {
            "claim_id": "ch10-low-bid-score-distortion",
            "status": "supported",
            "evidence": "generated match winners sometimes disagree under point-to-250 vs marks-to-7 labels",
            "metric": "mean_match_winner_disagreement_rate",
            "value": round(mean(row["winner_disagreement_rate"] for row in match_summary), 6),
            "caveat": "frequency is generated-policy dependent",
        },
        {
            "claim_id": "ch10-point-system-skill-signal",
            "status": "context-limited",
            "evidence": "heuristic policy-population rows expose different separation/noise under point and mark labels",
            "metric": "policy_population_rows",
            "value": len(POLICY_PAIRS),
            "caveat": "needs oracle or human policy population before promotion beyond context-limited",
        },
        {
            "claim_id": "ch10-tournament-speed-tradeoff",
            "status": "supported-for-generated-trace-proxy",
            "evidence": "mark matches used fewer played tricks than point-to-250 matches in generated traces",
            "metric": "mean_trick_reduction_points_minus_marks",
            "value": round(point_match_trick_mean - mark_match_trick_mean, 6),
            "caveat": "tricks are a time proxy; no wall-clock human table data",
        },
        {
            "claim_id": "ch10-timed-marks-advancement-objective",
            "status": "context-limited",
            "evidence": "timed synthetic pools sometimes select a different leader by marks than by points",
            "metric": "advancement_disagreement_rate",
            "value": timed_disagreement,
            "caveat": "synthetic round-robin with trick budget, not a real bracket or human time control",
        },
    ]
    return rows


def build_examples(hands: list[HandResult], timed_rows: list[dict[str, Any]], limit: int) -> dict[str, Any]:
    examples: dict[str, Any] = {claim: [] for claim in CLAIM_IDS}
    for hand in hands:
        row = hand_to_row(hand)
        compact = {k: row[k] for k in row if k != "full_play_history"}
        if hand.tricks_saved > 0 and len(examples["ch10-early-terminal-under-marks"]) < limit:
            compact["full_play_history"] = hand.full_play_history
            examples["ch10-early-terminal-under-marks"].append(compact)
        if hand.partial_defender_points_erased > 0 and len(examples["ch10-nonbidder-partial-points-erased"]) < limit:
            examples["ch10-nonbidder-partial-points-erased"].append(compact)
        if hand.set_severity_points > 0 and len(examples["ch10-set-severity-compression"]) < limit:
            examples["ch10-set-severity-compression"].append(compact)
        if hand.bid >= 84 and len(examples["ch10-special-bid-mark-multiplier"]) < limit:
            examples["ch10-special-bid-mark-multiplier"].append(compact)
        if hand.point_team0_score - hand.point_team1_score != hand.mark_team0_score - hand.mark_team1_score and len(
            examples["ch10-score-mode-objective"]
        ) < limit:
            examples["ch10-score-mode-objective"].append(compact)
        if hand.bid == 30 and hand.made and hand.defender_points >= 10 and len(examples["ch10-low-bid-score-distortion"]) < limit:
            examples["ch10-low-bid-score-distortion"].append(compact)
    for row in timed_rows:
        if row["advancement_disagreement"] and len(examples["ch10-timed-marks-advancement-objective"]) < limit:
            examples["ch10-timed-marks-advancement-objective"].append(row)
    examples["ch10-point-system-skill-signal"] = [
        "See policy_population_signal.csv for heuristic policy separation under point and mark labels."
    ]
    examples["ch10-tournament-speed-tradeoff"] = [
        "See match_mode_summary.csv for point-to-250 full-trick proxy versus marks-to-7 early-stop proxy."
    ]
    return examples


def blocker_rows() -> list[dict[str, Any]]:
    return [
        {
            "claim_id": "ch10-point-system-skill-signal",
            "missing_field": "oracle_or_human_policy_population",
            "why_needed": "heuristic policies can show score-label separation, but not actual skill honing",
            "next_generator_or_data": "paired policy-population arena with E[Q]/Gus/Burl/human traces under point and mark objectives",
        },
        {
            "claim_id": "ch10-tournament-speed-tradeoff",
            "missing_field": "wall_clock_table_time",
            "why_needed": "tricks saved are a proxy; shuffling, bidding, table talk, and laydowns change real speed",
            "next_generator_or_data": "human table logs or calibrated per-trick/per-hand timing model",
        },
        {
            "claim_id": "ch10-timed-marks-advancement-objective",
            "missing_field": "real_tournament_format_and_tiebreakers",
            "why_needed": "synthetic round-robin marks-vs-points disagreement does not identify actual bracket EV",
            "next_generator_or_data": "bracket simulator with published format, clock, total-mark tiebreak, and opponent population priors",
        },
    ]


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    hands = generated_hands(args)
    hand_rows = [hand_to_row(hand) for hand in hands]
    early_summary = summarize_early_terminal(hands)
    compression_summary = summarize_objective_compression(hands)
    policy_summary = summarize_policy_population(hands)
    match_rows = simulate_matches(args)
    match_summary = summarize_matches(match_rows)
    timed_rows = simulate_timed_trials(args)
    timed_summary = summarize_timed(timed_rows)
    deterministic_terminal, deterministic_thresholds = deterministic_rows()
    claim_rows = claim_summary_rows(hands, match_summary, timed_summary)
    examples = build_examples(hands, timed_rows, args.example_limit)
    blockers = blocker_rows()

    write_csv(args.output_dir / "generated_hand_traces.csv", hand_rows)
    write_csv(args.output_dir / "early_terminal_summary.csv", early_summary)
    write_csv(args.output_dir / "objective_compression_summary.csv", compression_summary)
    write_csv(args.output_dir / "policy_population_signal.csv", policy_summary)
    write_csv(args.output_dir / "match_mode_rows.csv", match_rows)
    write_csv(args.output_dir / "match_mode_summary.csv", match_summary)
    write_csv(args.output_dir / "timed_advancement_rows.csv", timed_rows)
    write_csv(args.output_dir / "timed_advancement_summary.csv", timed_summary)
    write_csv(args.output_dir / "deterministic_terminal_transform.csv", deterministic_terminal)
    write_csv(args.output_dir / "deterministic_thresholds.csv", deterministic_thresholds)
    write_csv(args.output_dir / "claim_summary.csv", claim_rows)
    write_csv(args.output_dir / "blockers.csv", blockers)
    write_json(args.output_dir / "examples.json", examples)

    summary = {
        "schema": "w42.phase4_scoring_objective_tests.v1",
        "bead": BEAD_ID,
        "repo_commit": git_sha(),
        "git_status_short": git_status_short(),
        "config": {
            "hands_per_pair": 80 if args.smoke else args.hands_per_pair,
            "match_count_per_pair": 16 if args.smoke else args.match_count,
            "timed_trials": 16 if args.smoke else args.timed_trials,
            "timed_trick_budget": args.timed_trick_budget,
            "base_seed": args.base_seed,
            "policy_pairs": [f"{a}_vs_{b}" for a, b in POLICY_PAIRS],
        },
        "coverage": {
            "generated_hands": len(hands),
            "match_rows": len(match_rows),
            "timed_trials": len(timed_rows),
            "deterministic_terminal_rows": len(deterministic_terminal),
            "claims_reported": len(claim_rows),
            "blockers": len(blockers),
        },
        "headline_metrics": {
            "early_terminal_rate": round(sum(1 for h in hands if h.tricks_saved > 0) / len(hands), 6),
            "mean_tricks_saved_per_generated_hand": round(mean(h.tricks_saved for h in hands), 6),
            "partial_erasure_hands": sum(1 for h in hands if h.partial_defender_points_erased > 0),
            "distinct_ordinary_set_severity_values": len(
                {h.set_severity_points for h in hands if h.bid_class == "ordinary" and h.set_severity_points > 0}
            ),
            "mean_match_winner_disagreement_rate": round(
                mean(row["winner_disagreement_rate"] for row in match_summary), 6
            ),
            "timed_advancement_disagreement_rate": timed_summary[0]["advancement_disagreement_rate"],
        },
        "claim_statuses": {row["claim_id"]: row["status"] for row in claim_rows},
        "source_pages": [
            "wiki/AGENTS.md",
            "wiki/experiments/winning42-ch10-tournament-scoring.md",
            "wiki/experiments/w42-scoring-objective-drift-claim-validation.md",
            "scratch/winning42/winning42.with_figures.md lines 4243-4331",
        ],
        "caveats": [
            "Generated play uses deterministic heuristic policies, not oracle-optimal play.",
            "Trick count is a tournament time proxy; no wall-clock human timing data is used.",
            "Timed advancement simulation is a synthetic four-team round-robin with a fixed trick budget.",
            "No wiki or bead files are edited by this worker artifact.",
        ],
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
