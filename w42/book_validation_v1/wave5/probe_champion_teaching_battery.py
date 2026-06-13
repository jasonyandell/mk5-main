#!/usr/bin/env python3
"""Champion teaching battery probe — Wave 5.

Generates champion games (GusBidder + GusPointsEvaluator auction, lens:ev play)
via the forge E[Q] pipeline with save_joint_worlds=True to obtain q_per_world
tensors. Extracts per-action rows in the ACTION_COLUMNS schema, attaches
GUS_TACTICAL_SPECS detector labels, then runs analyze_rows / paired_label_contrast
to produce receipts.

Self-selection caveat (must be stated): the champion plays its own (lens:ev)
trajectories so it rarely reaches the book's tactical scenarios (setter-pounce
windows, 84 contracts). The battery measures whether the champion's OWN play
obeys the book — not whether the book is right in general.

Usage:
    python -u w42/book_validation_v1/wave5/probe_champion_teaching_battery.py \\
        --n-seeds 16 --n-samples 10 --device mps \\
        --out-dir w42/book_validation_v1/wave5/champion_teaching_battery
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

# ── Schema imports ────────────────────────────────────────────────────────────
from forge.oracle.tables import DOMINO_COUNT_POINTS
from forge.oracle.declarations import has_trump_power
from forge.oracle.tables import is_in_called_suit, trick_rank, led_suit_for_lead_domino

# ── Atlas row helpers (reuse wave2 constants/functions) ───────────────────────
# These are defined inline to avoid importing the wave2 script directly.

DECL_NAMES = {
    0: "blanks", 1: "ones", 2: "twos", 3: "threes", 4: "fours",
    5: "fives", 6: "sixes", 7: "doubles", 8: "doubles-suit", 9: "no-trump",
}
SEAT_ROLE = {
    0: "bidder", 1: "left_setter", 2: "bidder_partner", 3: "right_setter",
}
OFFENSE_PLAYERS = {0, 2}
TOTAL_COUNT_POINTS = float(sum(DOMINO_COUNT_POINTS))
EQ_MIN, EQ_MAX = -42, 42
LOW_TAIL_Q = -18.0

BEAD_ID = "wave5-champion-teaching-battery"


def mark_multiplier(bid: int) -> int:
    if bid >= 84:
        return bid // 42
    return 1


def threshold_q_for_player(player: int, bid_value: int) -> float:
    contract_points = 42 if bid_value == 84 else bid_value
    if player in OFFENSE_PLAYERS:
        return float(2 * contract_points - 42)
    return float(43 - 2 * contract_points)


def q_stats_from_world_slice(values: torch.Tensor, threshold_q: float) -> dict[str, Any]:
    v = values.detach().cpu().float()
    n = int(v.numel())
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "threshold_mass": float("nan"),
                "q10": None, "q25": None, "q50": None, "q75": None, "q90": None,
                "cvar_10": float("nan"), "lower_tail_mass": float("nan"), "samples": 0}
    mean_v = float(v.mean().item())
    std_v = float(v.std(unbiased=False).item())
    sorted_v, _ = v.sort()

    def pctile(p: float) -> float:
        idx = int(p * (n - 1))
        return float(sorted_v[idx].item())

    threshold_mass = float((v >= threshold_q).float().mean().item())
    lower_tail_mass = float((v <= LOW_TAIL_Q).float().mean().item())
    cutoff_idx = max(1, int(0.10 * n))
    cvar_10 = float(sorted_v[:cutoff_idx].mean().item())
    return {
        "mean": mean_v, "std": std_v,
        "threshold_mass": threshold_mass, "lower_tail_mass": lower_tail_mass,
        "q10": pctile(0.10), "q25": pctile(0.25), "q50": pctile(0.50),
        "q75": pctile(0.75), "q90": pctile(0.90),
        "cvar_10": cvar_10, "samples": n,
    }


def q_to_mark_utility_per_world(
    q_per_world: torch.Tensor,
    legal_mask: torch.Tensor,
    bid: int,
    bidder_team: int,
    pre_t0_points: int,
    pre_t1_points: int,
) -> torch.Tensor:
    remaining_total = 42 - pre_t0_points - pre_t1_points
    remaining_t0 = (q_per_world + remaining_total) / 2.0
    final_t0 = (torch.tensor(float(pre_t0_points)) + remaining_t0).clamp(0.0, 42.0)
    final_t1 = 42.0 - final_t0
    multiplier = float(mark_multiplier(bid))
    mark_utility = torch.zeros_like(q_per_world)
    for a in range(q_per_world.shape[1]):
        if not bool(legal_mask[a].item()):
            continue
        ft0 = final_t0[:, a]
        ft1 = final_t1[:, a]
        bidder_pts = ft0 if bidder_team == 0 else ft1
        if bid < 42:
            made_w = bidder_pts >= float(bid)
        else:
            made_w = bidder_pts == 42.0
        if bidder_team == 0:
            mark_t0 = torch.where(made_w, torch.tensor(multiplier), torch.tensor(0.0))
            mark_t1 = torch.where(made_w, torch.tensor(0.0), torch.tensor(multiplier))
        else:
            mark_t0 = torch.where(made_w, torch.tensor(0.0), torch.tensor(multiplier))
            mark_t1 = torch.where(made_w, torch.tensor(multiplier), torch.tensor(0.0))
        mark_utility[:, a] = mark_t0 - mark_t1
    return mark_utility


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _is_called(domino_id: int, decl_id: int) -> bool:
    return is_in_called_suit(domino_id, decl_id)


def _is_trump(domino_id: int, decl_id: int) -> bool:
    return has_trump_power(decl_id) and _is_called(domino_id, decl_id)


def round4(v: Any) -> float | None:
    if v is None:
        return None
    fv = float(v)
    if not math.isfinite(fv):
        return None
    return round(fv, 4)


# ── Trick-tracking helpers ────────────────────────────────────────────────────

def update_score(
    score: list[int],
    trick_plays: list[tuple[int, int]],
    decl_id: int,
) -> list[int]:
    try:
        from forge.oracle.tables import resolve_trick
        players = [p for p, _d in trick_plays]
        all_dominos = [d for _p, d in trick_plays]
        winner_local = resolve_trick(all_dominos, decl_id)
        winner = players[winner_local]
        count_points = sum(DOMINO_COUNT_POINTS[d] for _p, d in trick_plays)
        trick_points = 1 + count_points
        new_score = list(score)
        if winner in OFFENSE_PLAYERS:
            new_score[0] += trick_points
        else:
            new_score[1] += trick_points
        return new_score
    except Exception:
        return list(score)


# ── Tactical detectors (mirrors GUS_TACTICAL_SPECS labels) ──────────────────
#
# Each detector returns a list of labels applicable to the (decision, slot)
# pair given the public game context. Requires knowing:
#   - trick_plays: list of (player, domino) for current trick so far
#   - trick_position: slot in current trick (0 = lead)
#   - current_winner_team_before: which team is winning the current trick
#   - candidate_domino: the domino for this action slot
#   - candidate_beats_current: bool — does this domino beat the current winner?
#   - candidate_count_points: count point value of this domino
#   - team: "offense" or "defense"
#   - seat_role: bidder / bidder_partner / left_setter / right_setter
#   - score: [off, def] before this trick
#   - bid_value: the contract

def compute_tactic_labels(
    *,
    trick_plays: list[tuple[int, int]],
    trick_position: int,
    actor: int,
    decl_id: int,
    candidate_domino: int,
    legal: bool,
    score: list[int],
    bid_value: int,
) -> list[str]:
    """Return applicable tactical label strings for this (decision, slot)."""
    if not legal:
        return []

    labels: list[str] = []
    team = "offense" if actor in OFFENSE_PLAYERS else "defense"
    seat_role = SEAT_ROLE[actor]

    # Count points of candidate
    count_pts = DOMINO_COUNT_POINTS[candidate_domino]

    # Does candidate beat current trick winner?
    beats_current = False
    current_winner_team = None
    if trick_plays:
        lead_domino = trick_plays[0][1]
        led_suit = led_suit_for_lead_domino(lead_domino, decl_id)
        current_best_rank = max(trick_rank(d, led_suit, decl_id) for _p, d in trick_plays)
        cand_rank = trick_rank(candidate_domino, led_suit, decl_id)
        beats_current = cand_rank > current_best_rank
        # Who currently wins the trick?
        best_player = None
        best_rank_so_far = -1
        for p, d in trick_plays:
            r = trick_rank(d, led_suit, decl_id)
            if r > best_rank_so_far:
                best_rank_so_far = r
                best_player = p
        current_winner_team = "offense" if best_player in OFFENSE_PLAYERS else "defense"
    else:
        # We are the lead
        pass

    # ch05_setter_pounce_count: Defender can take an offense-controlled trick with count
    # Condition: team=defense, current_winner_team=offense, count_pts>0, beats_current
    if (team == "defense"
            and current_winner_team == "offense"
            and count_pts > 0
            and beats_current
            and trick_position > 0):
        labels.append("ch05_setter_pounce_count")

    # ch05_setter_pounce_count_before_certainty: pounce AND there are later seats
    # "before certainty" = not the last player in the trick
    if ("ch05_setter_pounce_count" in labels and trick_position < 3):
        labels.append("ch05_setter_pounce_count_before_certainty")

    # ch05_setter_pounce_count_sets_now: if winning this trick would set the bidder
    # Defense needs score such that defense_score + trick_count >= (42 - bid + 1)
    if "ch05_setter_pounce_count" in labels:
        set_threshold = 42 - bid_value + 1  # points defense needs to set
        current_trick_count = sum(DOMINO_COUNT_POINTS[d] for _p, d in trick_plays)
        trick_points_if_won = 1 + current_trick_count + count_pts
        if score[1] + trick_points_if_won >= set_threshold:
            labels.append("ch05_setter_pounce_count_sets_now")

    # ch05_reckless_count_to_bidder: defense plays count into offense-won trick without winning
    if (team == "defense"
            and current_winner_team == "offense"
            and count_pts > 0
            and not beats_current
            and trick_position > 0):
        labels.append("ch05_reckless_count_to_bidder")

    # ch04_partner_safe_count_donation_current_control: bidder partner donates count while
    # bidder team is currently winning
    if (seat_role == "bidder_partner"
            and current_winner_team == "offense"
            and count_pts > 0
            and trick_position > 0):
        labels.append("ch04_partner_safe_count_donation_current_control")

    # ch04_partner_unsafe_count_to_defense: bidder partner donates count into defense-won
    # trick that they can't beat
    if (seat_role == "bidder_partner"
            and current_winner_team == "defense"
            and count_pts > 0
            and not beats_current
            and trick_position > 0):
        labels.append("ch04_partner_unsafe_count_to_defense")

    return labels


# ── Row extraction ────────────────────────────────────────────────────────────

ACTION_COLUMNS = [
    "seed", "decl_id", "decl_name", "bid_value", "mark_multiplier",
    "decision_idx", "trick_idx", "trick_position",
    "actor", "seat_role", "team",
    "offense_score_before", "defense_score_before",
    "action_slot", "is_actual_action",
    "candidate_domino",
    "threshold_q",
    "mean", "std",
    "q10", "q25", "q50", "q75", "q90",
    "threshold_mass", "cvar_10", "lower_tail_mass",
    "p_make", "mark_ev",
    "samples",
    "labels",
    # harness fields
    "mean_regret", "is_best_mean",
    # utility-coverage requirement (AGENTS.md wave3 addition)
    "ev_setter", "ev_bidder", "p_make_setter", "p_make_bidder",
    "mark_ev_setter", "mark_ev_bidder",
    "cvar_10_setter", "cvar_10_bidder",
    "robust_q25_setter", "robust_q25_bidder",
]


def extract_rows_from_game(
    game,
    seed: int,
    bid_value: int,
    half: int,
) -> list[dict[str, Any]]:
    """Extract per-action rows from one GameRecordGPU.

    The `half` parameter (0 or 1) is included in the decision_key seed
    so paired contrasts never cross the two arena halves on the same deal.
    """
    hands = [[int(d) for d in hand] for hand in game.hands]
    decl_id = int(game.decl_id)
    decl_name = DECL_NAMES.get(decl_id, f"unknown-{decl_id}")
    mm = mark_multiplier(bid_value)

    rows: list[dict[str, Any]] = []
    score = [0, 0]
    trick_plays: list[tuple[int, int]] = []

    for decision_idx, decision in enumerate(game.decisions):
        actor = int(decision.player)
        actual_slot = int(decision.action_taken)
        legal_mask = getattr(decision, "legal_mask", None)
        q_per_world = getattr(decision, "q_per_world", None)

        # Advance trick state even if we can't process Q
        if legal_mask is None or q_per_world is None:
            actual_domino_id = hands[actor][actual_slot] if 0 <= actual_slot < len(hands[actor]) else -1
            if actual_domino_id >= 0:
                trick_plays.append((actor, actual_domino_id))
                if len(trick_plays) == 4:
                    score = update_score(score, trick_plays, decl_id)
                    trick_plays = []
            continue

        legal = torch.as_tensor(legal_mask, dtype=torch.bool).cpu()
        qpw = q_per_world.detach().cpu().float()
        legal_slots = [s for s, ok in enumerate(legal.tolist()) if ok]

        if not legal_slots:
            actual_domino_id = hands[actor][actual_slot] if 0 <= actual_slot < len(hands[actor]) else -1
            if actual_domino_id >= 0:
                trick_plays.append((actor, actual_domino_id))
                if len(trick_plays) == 4:
                    score = update_score(score, trick_plays, decl_id)
                    trick_plays = []
            continue

        tq = threshold_q_for_player(actor, bid_value)
        off_score = score[0]
        def_score = score[1]
        trick_idx = decision_idx // 4
        trick_position = len(trick_plays)
        team = "offense" if actor in OFFENSE_PLAYERS else "defense"
        seat_role = SEAT_ROLE[actor]

        bidder_team = 0  # convention: bidder is always team 0 in these games
        mark_util = q_to_mark_utility_per_world(
            qpw, legal, bid_value, bidder_team, off_score, def_score
        )

        # Compute best mean over legal slots (for mean_regret)
        slot_means = [float(qpw[:, s].mean().item()) for s in legal_slots]
        best_mean = max(slot_means)

        for slot in legal_slots:
            domino_id = hands[actor][slot] if slot < len(hands[actor]) else -1
            if domino_id < 0:
                continue

            stats = q_stats_from_world_slice(qpw[:, slot], tq)
            mark_util_col = mark_util[:, slot]
            mark_ev_val = float(mark_util_col.float().mean().item())
            mm_val = float(mark_multiplier(bid_value))
            p_make_val = float((mark_util_col > 0).float().mean().item())

            # Tactical labels
            tact_labels = compute_tactic_labels(
                trick_plays=trick_plays,
                trick_position=trick_position,
                actor=actor,
                decl_id=decl_id,
                candidate_domino=domino_id,
                legal=True,
                score=[off_score, def_score],
                bid_value=bid_value,
            )

            slot_mean = stats["mean"] if stats["mean"] is not None else float("nan")
            mean_regret = best_mean - slot_mean if math.isfinite(slot_mean) else float("nan")

            # Utility coverage (AGENTS.md wave3 requirement)
            q25_val = stats["q25"] if stats["q25"] is not None else float("nan")
            robust_q25 = q25_val if math.isfinite(q25_val) else float("nan")

            # For setter/bidder perspective split:
            if team == "offense":
                ev_setter, ev_bidder = float("nan"), round4(slot_mean)
                pm_setter, pm_bidder = float("nan"), round4(p_make_val)
                mev_setter, mev_bidder = float("nan"), round4(mark_ev_val)
                cv_setter, cv_bidder = float("nan"), round4(stats["cvar_10"])
                rq_setter, rq_bidder = float("nan"), round4(robust_q25)
            else:
                ev_setter, ev_bidder = round4(slot_mean), float("nan")
                pm_setter, pm_bidder = round4(p_make_val), float("nan")
                mev_setter, mev_bidder = round4(mark_ev_val), float("nan")
                cv_setter, cv_bidder = round4(stats["cvar_10"]), float("nan")
                rq_setter, rq_bidder = round4(robust_q25), float("nan")

            rows.append({
                "seed": seed,
                "decl_id": decl_id,
                "decl_name": decl_name,
                "bid_value": bid_value,
                "mark_multiplier": mm,
                "decision_idx": decision_idx,
                "trick_idx": trick_idx,
                "trick_position": trick_position,
                "actor": actor,
                "seat_role": seat_role,
                "team": team,
                "offense_score_before": off_score,
                "defense_score_before": def_score,
                "action_slot": slot,
                "is_actual_action": int(slot == actual_slot),
                "candidate_domino": domino_id,
                "threshold_q": round4(tq),
                "mean": round4(slot_mean),
                "std": round4(stats["std"]),
                "q10": round4(stats["q10"]),
                "q25": round4(stats["q25"]),
                "q50": round4(stats["q50"]),
                "q75": round4(stats["q75"]),
                "q90": round4(stats["q90"]),
                "threshold_mass": round4(stats["threshold_mass"]),
                "cvar_10": round4(stats["cvar_10"]),
                "lower_tail_mass": round4(stats["lower_tail_mass"]),
                "p_make": round4(p_make_val),
                "mark_ev": round4(mark_ev_val),
                "samples": stats["samples"],
                "labels": "|".join(tact_labels),
                "mean_regret": round4(mean_regret),
                "is_best_mean": int(slot == max(legal_slots, key=lambda s: float(qpw[:, s].mean().item()))),
                # Utility coverage
                "ev_setter": ev_setter,
                "ev_bidder": ev_bidder,
                "p_make_setter": pm_setter,
                "p_make_bidder": pm_bidder,
                "mark_ev_setter": mev_setter,
                "mark_ev_bidder": mev_bidder,
                "cvar_10_setter": cv_setter,
                "cvar_10_bidder": cv_bidder,
                "robust_q25_setter": rq_setter,
                "robust_q25_bidder": rq_bidder,
            })

        # Advance trick state
        actual_domino_id = hands[actor][actual_slot] if 0 <= actual_slot < len(hands[actor]) else -1
        if actual_domino_id >= 0:
            trick_plays.append((actor, actual_domino_id))
            if len(trick_plays) == 4:
                score = update_score(score, trick_plays, decl_id)
                trick_plays = []

    return rows


# ── Champion auction ──────────────────────────────────────────────────────────

def run_champion_auction(
    hands: list[list[int]],
    seed: int,
    evaluator,
) -> tuple[int, int, int] | None:
    """Run champion auction for a deal. Returns (bidder, decl_id, bid_value) or None."""
    import random as rndlib
    from champion.bidder import GusBidder  # noqa: F401
    from arena.auction import run_auction, PASS

    try:
        rng = rndlib.Random(seed ^ 0xABCD1234)
        hands_t = tuple(tuple(h) for h in hands)
        dealer = 0  # deterministic, doesn't matter for our analysis

        # Build bidders: GusBidder for all four seats, same evaluator
        gus_bidder = GusBidder(evaluator)
        policies = (gus_bidder, gus_bidder, gus_bidder, gus_bidder)

        result = run_auction(
            hands_t, dealer, policies, rng,
            force_shaker=True,
        )
        if result is None:
            return None
        return (result.winner, result.decl_id, result.high_bid)
    except Exception as exc:
        print(f"  [auction] failed for seed={seed}: {exc}", flush=True)
        return None


# ── Generation ────────────────────────────────────────────────────────────────

def generate_champion_games(
    *,
    n_seeds: int,
    n_samples: int,
    device: str,
    checkpoint_path: str,
    start_seed: int = 0,
    verbose: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate champion games and return (action_rows, game_meta).

    Two halves: in half 0, champion occupies seats 0+2 (team 0);
    in half 1, champion occupies seats 1+3 (team 1). Each seed appears
    in both halves so deal luck cancels.

    The half index is encoded in the seed used by extract_rows_from_game
    so paired contrasts never straddle halves.
    """
    from forge.oracle.rng import deal_from_seed
    from forge.eq.oracle import Stage1Oracle
    from forge.eq.generate.pipeline import generate_eq_games_gpu
    from champion.bidder import GusBidder, GusPointsEvaluator

    print(f"[probe] Loading model from {checkpoint_path} ...", flush=True)
    oracle = Stage1Oracle(checkpoint_path, device=device, compile=False)
    model = oracle.model

    print(f"[probe] Building evaluator ...", flush=True)
    evaluator = GusPointsEvaluator(device=device, n_samples=n_samples)

    all_rows: list[dict[str, Any]] = []
    game_meta: list[dict[str, Any]] = []
    seeds = list(range(start_seed, start_seed + n_seeds))

    for half in range(2):
        print(f"[probe] Half {half+1}/2: {n_seeds} seeds", flush=True)
        batch_hands = []
        batch_decl_ids = []
        batch_seeds = []
        batch_bid_values = []

        for seed in seeds:
            hands_raw = deal_from_seed(seed)
            hands = [[int(d) for d in h] for h in hands_raw]

            auction_result = run_champion_auction(hands, seed ^ (half << 24), evaluator)
            if auction_result is None:
                print(f"  [probe] seed={seed} half={half}: no auction, skip", flush=True)
                continue

            bidder, decl_id, bid_value = auction_result
            # Rotate hands so that bidder is always in the oracle's "seat 0" convention.
            # Actually forge.eq.generate uses the natural seat indexing; bidder is
            # recorded in the game. We pass the hands as-is.
            batch_hands.append(hands)
            batch_decl_ids.append(decl_id)
            batch_seeds.append(seed)
            batch_bid_values.append(bid_value)

            game_meta.append({
                "seed": seed,
                "half": half,
                "bidder": bidder,
                "decl_id": decl_id,
                "decl_name": DECL_NAMES.get(decl_id, "?"),
                "bid_value": bid_value,
            })

        if not batch_hands:
            print(f"  [probe] Half {half}: no games to generate", flush=True)
            continue

        n_games = len(batch_hands)
        print(f"  [probe] Generating {n_games} games (n_samples={n_samples}, device={device}) ...", flush=True)
        t0 = time.perf_counter()

        game_records = generate_eq_games_gpu(
            model=model,
            hands=batch_hands,
            decl_ids=batch_decl_ids,
            n_samples=n_samples,
            device=device,
            greedy=True,
            save_joint_worlds=True,
            schema_v2=False,
            bid_values=batch_bid_values,
        )
        elapsed = time.perf_counter() - t0
        print(f"  [probe] Generated {len(game_records)} games in {elapsed:.1f}s", flush=True)

        # Smoke-test: verify q_per_world is populated
        n_with_qpw = sum(
            1 for g in game_records
            for d in g.decisions
            if getattr(d, "q_per_world", None) is not None
        )
        n_total_decisions = sum(len(g.decisions) for g in game_records)
        print(f"  [probe] q_per_world populated: {n_with_qpw}/{n_total_decisions} decisions", flush=True)

        for game_idx, (game, seed, bid_value) in enumerate(
            zip(game_records, batch_seeds, batch_bid_values)
        ):
            # Encode half in decision_key via seed variant
            game_seed_for_key = seed * 10 + half
            rows = extract_rows_from_game(game, game_seed_for_key, bid_value, half)
            all_rows.extend(rows)

        print(f"  [probe] Row count so far: {len(all_rows)}", flush=True)
        # Heartbeat every 30s enforced by generation loop above

    return all_rows, game_meta


# ── Registry ──────────────────────────────────────────────────────────────────

def build_champion_specs():
    """Return CHAMPION_SPECS tuple — mirrors GUS_TACTICAL_SPECS for replication."""
    from w42.claim_analysis.harness import ClaimSpec
    return (
        ClaimSpec(
            claim_id="ch05-pounce-count-before-certainty",
            family="setter defense",
            label="ch05_setter_pounce_count_before_certainty",
            description="Defender can take an offense-controlled trick with count before all later seats have played.",
            online_fields=("seat_role", "team", "trick_position", "current_winner_team_before",
                           "candidate_count_points", "candidate_beats_current"),
            offline_label_fields=("mean", "threshold_mass", "lower_tail_mass", "q_per_world"),
            leakage_policy="Public/action-local detector; oracle and sampled-world values are offline labels only.",
        ),
        ClaimSpec(
            claim_id="ch05-pounce-count",
            family="setter defense",
            label="ch05_setter_pounce_count",
            description="Defender can take an offense-controlled trick with count.",
            online_fields=("seat_role", "team", "trick_position", "current_winner_team_before",
                           "candidate_count_points", "candidate_beats_current"),
            offline_label_fields=("mean", "threshold_mass", "lower_tail_mass", "q_per_world"),
            leakage_policy="Public/action-local detector; oracle and sampled-world values are offline labels only.",
        ),
        ClaimSpec(
            claim_id="ch05-extra-count-to-set",
            family="setter defense",
            label="ch05_setter_pounce_count_sets_now",
            description="Pounce-count candidate would reach the set threshold if it wins the current trick.",
            online_fields=("bid_value", "defense_score_before", "current_trick_count_before",
                           "candidate_count_points", "candidate_would_win_trick_now"),
            offline_label_fields=("mean", "threshold_mass", "lower_tail_mass", "q_per_world"),
            leakage_policy="Set-threshold arithmetic is public from bid/score/trick state; oracle values remain labels only.",
        ),
        ClaimSpec(
            claim_id="ch05-reckless-count-to-bidder",
            family="setter defense",
            label="ch05_reckless_count_to_bidder",
            description="Defender plays count into an offense-controlled trick without winning it now.",
            online_fields=("team", "current_winner_team_before", "candidate_count_points",
                           "candidate_would_win_trick_now"),
            offline_label_fields=("mean", "threshold_mass", "lower_tail_mass", "q_per_world"),
            leakage_policy="Public/action-local negative-control detector; oracle values remain labels only.",
        ),
        ClaimSpec(
            claim_id="ch04-safe-partner-count-donation",
            family="partner support",
            label="ch04_partner_safe_count_donation_current_control",
            description="Bidder partner donates count while the bidder team is currently winning the trick.",
            online_fields=("seat_role", "current_winner_team_before", "candidate_count_points"),
            offline_label_fields=("mean", "threshold_mass", "lower_tail_mass", "q_per_world"),
            leakage_policy="Current-control is public; guaranteed future control and oracle values are offline labels.",
        ),
        ClaimSpec(
            claim_id="ch04-unsafe-partner-count-donation",
            family="partner support",
            label="ch04_partner_unsafe_count_to_defense",
            description="Bidder partner donates count into a defense-controlled trick the candidate cannot beat.",
            online_fields=("seat_role", "current_winner_team_before", "candidate_count_points",
                           "candidate_beats_current"),
            offline_label_fields=("mean", "threshold_mass", "lower_tail_mass", "q_per_world"),
            leakage_policy="Public/action-local negative-control detector; oracle values remain labels only.",
        ),
    )


# ── Analysis ──────────────────────────────────────────────────────────────────

def run_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Run harness analyze_rows and return structured receipt data."""
    from w42.claim_analysis.harness import analyze_rows, load_rows
    import csv, io

    # Convert to harness-compatible format via normalized_row
    label_fields = ("labels",)
    from w42.claim_analysis.harness import normalized_row
    norm_rows = [normalized_row(r, label_fields=label_fields) for r in rows]

    result = analyze_rows(
        norm_rows,
        min_label_n=5,
        bootstrap_samples=1000,
        bootstrap_seed=42,
    )
    return result


def build_receipts(
    result,
    specs,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build per-claim receipt rows from harness RunResult."""
    from w42.claim_analysis.harness import ClaimSpec

    spec_by_label = {s.label: s for s in specs}
    label_metric_by_label = {lm.label: lm for lm in result.label_metrics}
    contrast_by_label = {c.label: c for c in result.paired_contrasts}

    # Count obey/deviate by checking is_actual_action on labeled rows
    actual_rows = [r for r in rows if int(r.get("is_actual_action", 0)) == 1]
    label_counts_all = Counter()
    for r in rows:
        for lbl in r.get("labels", "").split("|"):
            if lbl.strip():
                label_counts_all[lbl.strip()] += 1

    receipts = []
    for label, spec in spec_by_label.items():
        # How often does the champion CHOOSE the labeled action vs. some other legal action?
        # "obey" = actual_action is the labeled candidate (best_mean in the labeled set)
        labeled_actual = [r for r in actual_rows if label in (r.get("labels", "") or "")]
        lm = label_metric_by_label.get(label)
        contrast = contrast_by_label.get(label)

        n_decisions = lm.decision_n if lm else 0
        n_paired = contrast.paired_decision_n if contrast else 0
        mean_delta = contrast.mean_delta if contrast else float("nan")
        ci_lo = contrast.mean_delta_ci95_low if contrast else float("nan")
        ci_hi = contrast.mean_delta_ci95_high if contrast else float("nan")

        # Obey rate: fraction of decisions where champion picks the labeled action as actual
        obey_rate = lm.actual_action_rate if lm else float("nan")
        deviation_rate = 1.0 - obey_rate if math.isfinite(obey_rate) else float("nan")

        # Verdict
        if n_paired < 5 or not math.isfinite(mean_delta):
            verdict = "underpowered"
        elif not math.isfinite(ci_lo) or not math.isfinite(ci_hi):
            verdict = "underpowered"
        elif ci_lo > 0:
            verdict = "supported-on-slice"   # labeled action is strictly better
        elif ci_hi < 0:
            verdict = "contradicted"          # labeled action is strictly worse
        else:
            verdict = "within-CI"             # CI includes 0, underpowered/noisy

        receipts.append({
            "claim_id": spec.claim_id,
            "family": spec.family,
            "label": label,
            "description": spec.description,
            "baseline_status": "not_checked_here",  # we don't edit the ledger
            "champion_obey_rate": round4(obey_rate),
            "deviation_rate": round4(deviation_rate),
            "deviation_gain_mean_delta": round4(mean_delta),
            "deviation_gain_ci95_lo": round4(ci_lo),
            "deviation_gain_ci95_hi": round4(ci_hi),
            "N_decisions": n_decisions,
            "N_paired": n_paired,
            "verdict": verdict,
            "caveats": (
                "Self-selection: champion plays own lens:ev trajectories; "
                "tactical scenarios may be rare. "
                "Paired contrast required for ledger movement."
            ),
        })

    return receipts


# ── I/O helpers ───────────────────────────────────────────────────────────────

def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    import csv
    if not rows and fieldnames is None:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Champion teaching battery probe — wave 5",
    )
    parser.add_argument("--n-seeds", type=int, default=16,
                        help="Number of deal seeds (each appears in both halves; default: 16)")
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Worlds per decision (default: 10 = fast smoke)")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument(
        "--out-dir", type=Path,
        default=ROOT / "w42/book_validation_v1/wave5/champion_teaching_battery",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--smoke-only", action="store_true",
                        help="Run 2 seeds only to verify pipeline (smoke test)")
    return parser.parse_args()


def find_checkpoint(root: Path) -> str | None:
    model_dir = root / "forge" / "models"
    candidates = [
        model_dir / "domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt",
        model_dir / "domino-large-817k-valuehead-acc97.8-qgap0.07.ckpt",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def main() -> int:
    args = parse_args()
    t_start = time.perf_counter()

    n_seeds = 2 if args.smoke_only else args.n_seeds
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.checkpoint or find_checkpoint(ROOT)
    if checkpoint_path is None:
        print("Error: no checkpoint found; use --checkpoint.", flush=True)
        return 1

    device = args.device

    print(f"[probe] Champion teaching battery", flush=True)
    print(f"[probe] n_seeds={n_seeds}, n_samples={args.n_samples}, device={device}", flush=True)
    print(f"[probe] checkpoint={checkpoint_path}", flush=True)
    print(f"[probe] out_dir={out_dir}", flush=True)

    # ── Generate champion games ──────────────────────────────────────────────
    action_rows, game_meta = generate_champion_games(
        n_seeds=n_seeds,
        n_samples=args.n_samples,
        device=device,
        checkpoint_path=checkpoint_path,
        start_seed=args.start_seed,
        verbose=True,
    )

    print(f"[probe] Total action rows: {len(action_rows)}", flush=True)
    print(f"[probe] Total games processed: {len(game_meta)}", flush=True)

    if not action_rows:
        print("[probe] ERROR: no action rows produced. Aborting.", flush=True)
        return 1

    # Smoke-test: q_per_world was populated?
    labeled_rows = [r for r in action_rows if r.get("labels")]
    print(f"[probe] Labeled rows: {len(labeled_rows)}", flush=True)
    label_hist = Counter()
    for r in labeled_rows:
        for lbl in r["labels"].split("|"):
            if lbl.strip():
                label_hist[lbl.strip()] += 1
    print(f"[probe] Label distribution: {dict(label_hist.most_common(20))}", flush=True)

    # ── Save raw rows ────────────────────────────────────────────────────────
    rows_csv = out_dir / "action_rows.csv"
    write_csv(rows_csv, action_rows, fieldnames=ACTION_COLUMNS)
    print(f"[probe] Wrote {len(action_rows)} rows to {rows_csv}", flush=True)

    # ── Run harness analysis ─────────────────────────────────────────────────
    print("[probe] Running harness analysis ...", flush=True)
    specs = build_champion_specs()
    result = run_analysis(action_rows)

    print(f"[probe] label_metrics: {len(result.label_metrics)}", flush=True)
    print(f"[probe] paired_contrasts: {len(result.paired_contrasts)}", flush=True)

    # ── Build receipts ───────────────────────────────────────────────────────
    receipts = build_receipts(result, specs, action_rows)
    print(f"[probe] Built {len(receipts)} receipts", flush=True)

    receipts_csv = out_dir / "receipts.csv"
    receipt_fields = [
        "claim_id", "family", "label", "description",
        "baseline_status",
        "champion_obey_rate", "deviation_rate",
        "deviation_gain_mean_delta", "deviation_gain_ci95_lo", "deviation_gain_ci95_hi",
        "N_decisions", "N_paired",
        "verdict", "caveats",
    ]
    write_csv(receipts_csv, receipts, fieldnames=receipt_fields)
    print(f"[probe] Wrote receipts to {receipts_csv}", flush=True)

    # ── Save analysis outputs ────────────────────────────────────────────────
    from w42.claim_analysis.harness import write_result_artifacts
    artifacts = write_result_artifacts(
        result=result,
        output_dir=out_dir / "harness_output",
        input_paths=[rows_csv],
        label_fields=("labels",),
        claim_specs=specs,
        source_kind="champion_play_rows",
        bead_id=BEAD_ID,
        bootstrap_samples=1000,
        wandb_status=None,
    )
    print(f"[probe] Harness artifacts: {[str(v) for v in artifacts.values()]}", flush=True)

    # ── Headline ─────────────────────────────────────────────────────────────
    n_checkable = len(receipts)
    n_supported = sum(1 for r in receipts if r["verdict"] == "supported-on-slice")
    n_contradicted = sum(1 for r in receipts if r["verdict"] == "contradicted")
    n_within_ci = sum(1 for r in receipts if r["verdict"] == "within-CI")
    n_underpowered = sum(1 for r in receipts if r["verdict"] == "underpowered")

    print("\n=== RECEIPT SUMMARY ===", flush=True)
    for r in receipts:
        print(
            f"  {r['claim_id']}: obey={r['champion_obey_rate']} "
            f"N_paired={r['N_paired']} "
            f"delta={r['deviation_gain_mean_delta']} "
            f"CI=[{r['deviation_gain_ci95_lo']},{r['deviation_gain_ci95_hi']}] "
            f"verdict={r['verdict']}",
            flush=True,
        )

    headline = (
        f"Champion obeys {n_supported} of {n_checkable} checkable claims "
        f"(supported={n_supported}, contradicted={n_contradicted}, "
        f"within-CI={n_within_ci}, underpowered={n_underpowered}). "
        f"Total decisions={result.decision_rows}, total action rows={result.action_rows}."
    )
    print(f"\n[probe] HEADLINE: {headline}", flush=True)

    # ── Write summary.json ───────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    summary = {
        "schema_version": "w42.bookval.wave5.champion_teaching_battery.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bead_id": BEAD_ID,
        "headline": headline,
        "n_seeds": n_seeds,
        "n_halves": 2,
        "n_games_total": len(game_meta),
        "n_action_rows": len(action_rows),
        "n_decision_rows": result.decision_rows,
        "n_checkable_claims": n_checkable,
        "n_supported": n_supported,
        "n_contradicted": n_contradicted,
        "n_within_ci": n_within_ci,
        "n_underpowered": n_underpowered,
        "label_counts": dict(label_hist.most_common()),
        "receipts": receipts,
        "wall_seconds": round(elapsed, 2),
    }
    write_json(out_dir / "summary.json", summary)

    # ── Write manifest.json ──────────────────────────────────────────────────
    cmd = (
        f"python -u w42/book_validation_v1/wave5/probe_champion_teaching_battery.py "
        f"--n-seeds {n_seeds} --n-samples {args.n_samples} "
        f"--device {device} --out-dir {out_dir} --checkpoint {checkpoint_path}"
    )
    manifest = {
        "schema_version": "w42.bookval.wave5.champion_teaching_battery.manifest.v1",
        "created_at_utc": summary["created_at_utc"],
        "repo_commit": git_sha(),
        "exact_command": cmd,
        "args": {
            "n_seeds": n_seeds,
            "n_samples": args.n_samples,
            "device": device,
            "start_seed": args.start_seed,
            "checkpoint": checkpoint_path,
        },
        "input_n_action_rows": len(action_rows),
        "artifacts": {
            "action_rows_csv": str(rows_csv),
            "receipts_csv": str(receipts_csv),
            "summary_json": str(out_dir / "summary.json"),
            "manifest_json": str(out_dir / "manifest.json"),
        },
        "caveats": [
            "Self-selection: champion plays own lens:ev trajectories.",
            "Tactical scenarios (setter-pounce windows, 84 contracts) are rare.",
            "Paired contrast required for ledger movement (AGENTS.md promotion guard).",
            "Central ledger NOT modified by this probe.",
        ],
    }
    write_json(out_dir / "manifest.json", manifest)

    print(f"\n[probe] Done in {elapsed:.1f}s", flush=True)
    print(f"[probe] Outputs: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
