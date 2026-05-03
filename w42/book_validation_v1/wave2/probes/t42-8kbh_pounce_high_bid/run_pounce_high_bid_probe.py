"""High-bid pounce probe — Wave 2.E.2 (bids 35/36/39/42).

Tests `ch12-setter-pounce-high-bid-off`:
  At high bids (35/36/39/42), when the bidder's team has played a count domino
  in the most recent completed trick and the setter is now FOLLOWING (not
  leading), does the oracle prefer to "pounce" (win the current trick) over
  "decline" (play a non-winning tile)?

Wave 2.E showed pounce was wrong-by-EV at bid=30 (decline better in 65.4%).
Wave 2.B.2 aggregate Q-delta evidence shows the signal flips in book direction
at bid >= 39. This probe confirms with the same paired-contrast methodology but
on high-bid snapshots from the bid-aware atlas.

Pounce definition:
  A (pounce):  lowest-cost winning play (the legal tile that wins the trick
               with the smallest pip-rank, to avoid burning trump unnecessarily)
  B (decline): lowest-cost non-winning play

Metrics (per pair, from setter = Team 1 perspective):
  - EV delta:        E[Q](A) - E[Q](B) from Team 0 view; negative = pounce better for setter
  - p_set delta:     P(set | pounce) - P(set | decline); positive = pounce better for setter
  - mark_ev delta:   mark_ev(pounce) - mark_ev(decline); negative = pounce better for setter
  - CVaR delta:      CVaR_10(pounce) - CVaR_10(decline)

Sliced by:
  - Bid bucket (35/36/39/42)
  - Phase (early: tricks 0-2, mid: 3-5, late: 6-7)
  - Count-points-at-stake (5 vs 10)
  - Setter team led vs bidder team led

Usage:
    python run_pounce_high_bid_probe.py \\
        [--atlas-dir w42/book_validation_v1/wave2/bid_aware_atlas] \\
        [--checkpoint forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt] \\
        [--samples 100] [--device mps] [--max-per-bid 500]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

SNAPSHOTS_DIR = PROJECT_ROOT / "w42/book_validation_v1/wave2/snapshots"
sys.path.insert(0, str(SNAPSHOTS_DIR))

import numpy as np
import torch

from forge.oracle.tables import (
    DOMINO_COUNT_POINTS,
    DOMINO_HIGH,
    DOMINO_LOW,
    DOMINO_IS_DOUBLE,
    led_suit_for_lead_domino,
    resolve_trick,
)
from corpus_replayer import GameState, get_trump_set

SETTER_PLAYERS = {1, 3}
BIDDER_TEAM = {0, 2}
COUNT_THRESHOLD = 5
HIGH_BIDS = [35, 36, 39, 42]

# Mark multiplier (bid=84 -> 2, else 1)
def mark_multiplier(bid: int) -> int:
    if bid >= 84:
        return bid // 42
    return 1


# ---------------------------------------------------------------------------
# Trick-rank helper (mirrors corpus_replayer logic)
# ---------------------------------------------------------------------------

def trick_rank(domino_id: int, led_suit: int, decl_id: int) -> int:
    """Return rank for trick-winning comparison (higher = better)."""
    from forge.oracle.declarations import has_trump_power
    is_double = bool(DOMINO_IS_DOUBLE[domino_id])
    high = int(DOMINO_HIGH[domino_id])
    low = int(DOMINO_LOW[domino_id])
    pip_sum = high + low
    domino_suit = led_suit_for_lead_domino(domino_id, decl_id)
    is_trump = (domino_suit == 7)

    rank_in_suit = 14 if is_double else pip_sum

    if is_trump and has_trump_power(decl_id):
        return (2 << 4) + rank_in_suit
    if led_suit == 7:
        if is_trump:
            return (1 << 4) + high
        return 0
    has_pip = (led_suit == high) or (led_suit == low)
    if has_pip and not is_trump:
        return (1 << 4) + rank_in_suit
    return 0


def current_trick_winner_rank(trick_plays_so_far: list[tuple[int, int]], decl_id: int) -> int:
    """Return rank of the current trick winner (the tile leading the trick matters)."""
    if not trick_plays_so_far:
        return -1
    lead_domino = trick_plays_so_far[0][1]
    led_suit = led_suit_for_lead_domino(lead_domino, decl_id)
    best = 0
    for _player, did in trick_plays_so_far:
        r = trick_rank(did, led_suit, decl_id)
        if r > best:
            best = r
    return best


def can_win_current_trick(
    domino_id: int,
    trick_plays_so_far: list[tuple[int, int]],
    decl_id: int,
) -> bool:
    """Return True if playing domino_id would win the current trick."""
    if not trick_plays_so_far:
        return True  # lead is always a win
    lead_domino = trick_plays_so_far[0][1]
    led_suit = led_suit_for_lead_domino(lead_domino, decl_id)
    my_rank = trick_rank(domino_id, led_suit, decl_id)
    current_best = current_trick_winner_rank(trick_plays_so_far, decl_id)
    return my_rank > current_best


# ---------------------------------------------------------------------------
# Atlas snapshot mining
# ---------------------------------------------------------------------------

class MockGame:
    """Wraps a GameRecordGPU-like atlas game for corpus_replayer."""
    def __init__(self, game_obj):
        self.hands = game_obj.hands
        self.decl_id = game_obj.decl_id


def count_pts_in_current_trick(
    trick_plays_so_far: list[tuple[int, int]],
) -> int:
    """Total count points from bidder-team plays in the current trick."""
    return sum(DOMINO_COUNT_POINTS[did] for p, did in trick_plays_so_far if p in BIDDER_TEAM)


def _last_completed_trick(state: GameState) -> list[tuple[int, int]]:
    """Return [(player, domino_id)] for the most recently completed trick."""
    completed = state.history_count // 4
    if completed == 0:
        return []
    start = (completed - 1) * 4
    return [(state.history[start + i][0], state.history[start + i][1]) for i in range(4)]


def mine_pounce_snapshots_from_pt(
    pt_path: Path,
    bid_value: int,
    max_per_bid: int = 500,
) -> list[dict]:
    """
    Walk through each game in the atlas .pt file and find pounce-eligible
    decision points.

    Eligibility criteria (same as Wave 2.E but extended to current-trick count):
      1. Setter (player 1 or 3) is to move.
      2. Setter is following (not leading the current trick, i.e., n_trick >= 1).
      3. The LAST COMPLETED trick had at least one count domino (>=5 pts) from bidder team.
         --OR-- the current trick so far has count from bidder team (setter following).
      4. Setter has at least one legal winning move AND at least one legal non-winning move.

    Returns snapshot dicts with extra "_*" metadata fields.
    """
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    games = payload.get("results", [])
    seeds = payload.get("seeds", [])
    decl_ids = payload.get("decl_ids", [])

    snapshots = []

    for game_idx, game in enumerate(games):
        if len(snapshots) >= max_per_bid:
            break

        seed = int(seeds[game_idx]) if game_idx < len(seeds) else -1
        decl_id = int(getattr(game, "decl_id", decl_ids[game_idx] if game_idx < len(decl_ids) else -1))

        mock = MockGame(game)
        state = GameState.initial(mock)

        # Track current trick plays for trick-level analysis
        trick_plays_current: list[tuple[int, int]] = []

        for dec_idx, decision in enumerate(game.decisions):
            player = int(decision.player)
            legal_mask = decision.legal_mask.tolist()
            e_q = decision.e_q.tolist() if decision.e_q is not None else None

            trick_num = state.trick_number()
            n_in_trick = state.n_trick_plays()

            # Compute current trick plays from state
            # state.trick_plays reflects what's been put in so far
            trick_plays_so_far = [
                (
                    (state.leader + i) % 4,
                    state.trick_plays[i]
                )
                for i in range(n_in_trick)
                if state.trick_plays[i] >= 0
            ]

            # ---- Eligibility check ----
            is_setter = (player in SETTER_PLAYERS)
            is_following = (n_in_trick > 0)  # someone else has led

            if is_setter and is_following and e_q is not None:
                # Check count from bidder team in current trick so far
                count_in_current = count_pts_in_current_trick(trick_plays_so_far)

                # Also check last completed trick
                last_trick = _last_completed_trick(state)
                count_in_last = sum(
                    DOMINO_COUNT_POINTS[d] for p, d in last_trick if p in BIDDER_TEAM
                ) if last_trick else 0

                count_exposed = max(count_in_current, count_in_last)

                if count_exposed >= COUNT_THRESHOLD:
                    # Find legal slots
                    legal_slots = [s for s, ok in enumerate(legal_mask) if ok]
                    hand = state.hands[player]

                    # Determine winning vs non-winning legal moves
                    lead_domino = trick_plays_so_far[0][1]
                    led_suit = led_suit_for_lead_domino(lead_domino, decl_id)

                    winning_slots = []
                    declining_slots = []
                    for slot in legal_slots:
                        did = hand[slot]
                        if did < 0:
                            continue
                        wins = can_win_current_trick(did, trick_plays_so_far, decl_id)
                        if wins:
                            winning_slots.append((slot, did))
                        else:
                            declining_slots.append((slot, did))

                    # Must have both winning and non-winning options
                    if winning_slots and declining_slots:
                        snap = state.to_snapshot(bid_value=bid_value)
                        # Metadata
                        snap["_source_file"] = pt_path.name
                        snap["_game_idx"] = game_idx
                        snap["_decision_idx"] = dec_idx
                        snap["_seed"] = seed
                        snap["_decl_id"] = decl_id
                        snap["_bid_value"] = bid_value
                        snap["_setter_player"] = player
                        snap["_trick_num"] = trick_num
                        snap["_count_exposed"] = count_exposed
                        snap["_count_in_current"] = count_in_current
                        snap["_count_in_last"] = count_in_last
                        snap["_winning_slots"] = [(s, d) for s, d in winning_slots]
                        snap["_declining_slots"] = [(s, d) for s, d in declining_slots]
                        snap["_e_q_stored"] = e_q
                        snap["_legal_mask"] = legal_mask
                        snap["_led_suit"] = int(led_suit)
                        snap["_n_in_trick"] = n_in_trick
                        # Trick leader info
                        trick_leader = state.leader
                        snap["_bidder_team_led"] = int(trick_leader in BIDDER_TEAM)
                        snapshots.append(snap)

                        if len(snapshots) >= max_per_bid:
                            break

            # Apply the actual play to advance state
            actual_slot = int(decision.action_taken)
            actual_domino = state.hands[player][actual_slot] if 0 <= actual_slot < 7 else -1
            if actual_domino >= 0:
                state.apply_play(player, actual_domino)

    return snapshots


def mine_all_high_bid_snapshots(
    atlas_dir: Path,
    bids: list[int],
    max_per_bid: int = 500,
) -> dict[int, list[dict]]:
    """Mine pounce snapshots for each bid from the atlas files."""
    all_snapshots: dict[int, list[dict]] = {b: [] for b in bids}

    # Find all .pt files for each bid
    for bid in bids:
        bid_files = sorted(atlas_dir.glob(f"eq_pdf_seeds*_bid{bid}_v2.pt"))
        # Exclude validation seed 9430 from main corpus
        bid_files = [f for f in bid_files if "9430" not in f.name]
        print(f"[mine] bid={bid}: found {len(bid_files)} .pt files", flush=True)

        for pt_path in bid_files:
            if len(all_snapshots[bid]) >= max_per_bid:
                break
            t0 = time.perf_counter()
            snaps = mine_pounce_snapshots_from_pt(
                pt_path,
                bid_value=bid,
                max_per_bid=max_per_bid - len(all_snapshots[bid]),
            )
            all_snapshots[bid].extend(snaps)
            elapsed = time.perf_counter() - t0
            print(
                f"  {pt_path.name}: +{len(snaps)} snaps "
                f"(total={len(all_snapshots[bid])}) in {elapsed:.1f}s",
                flush=True,
            )

        print(f"  bid={bid}: total {len(all_snapshots[bid])} pounce-eligible snapshots", flush=True)

    return all_snapshots


# ---------------------------------------------------------------------------
# Snapshot selection: pounce vs decline actions
# ---------------------------------------------------------------------------

def select_pounce_decline(snap: dict) -> tuple[int | None, int | None, int | None, int | None]:
    """
    Return (pounce_slot, pounce_did, decline_slot, decline_did).

    Pounce = lowest-cost winning play (prefer non-trump, lower pip-sum first).
    Decline = lowest-cost non-winning play (prefer non-trump, lower pip-sum first).
    """
    winning_slots = snap.get("_winning_slots", [])
    declining_slots = snap.get("_declining_slots", [])

    if not winning_slots or not declining_slots:
        return None, None, None, None

    decl_id = snap["decl_id"]
    trump_set_7 = 7  # trump suit ID

    def cost(did: int) -> tuple[int, int, int]:
        """Lower cost = less valuable tile to play."""
        suit = led_suit_for_lead_domino(did, decl_id)
        is_trump = int(suit == trump_set_7)
        pip_sum = int(DOMINO_HIGH[did]) + int(DOMINO_LOW[did])
        return (is_trump, pip_sum)  # prefer non-trump, lower pip-sum

    # Best pounce = lowest cost winner
    pounce_slot, pounce_did = min(winning_slots, key=lambda x: cost(x[1]))
    # Best decline = lowest cost non-winner
    decline_slot, decline_did = min(declining_slots, key=lambda x: cost(x[1]))

    return pounce_slot, pounce_did, decline_slot, decline_did


# ---------------------------------------------------------------------------
# EQ evaluation via generate_eq_from_snapshots
# ---------------------------------------------------------------------------

def eval_action_eq(
    model,
    snap: dict,
    action_slot: int,
    n_samples: int,
    device: str,
) -> tuple[float, float, torch.Tensor | None]:
    """
    Run generate_eq_from_snapshots with this snapshot, return:
      (e_q_for_slot, cvar_10, pdf_tensor)

    Uses schema_v2 to get q_per_world.
    """
    from forge.eq.generate.pipeline import generate_eq_from_snapshots

    snap_clean = {k: v for k, v in snap.items() if not k.startswith("_")}
    records = generate_eq_from_snapshots(
        model=model,
        snapshots=[snap_clean],
        n_samples=n_samples,
        device=device,
        greedy=True,
        save_joint_worlds=True,
        schema_v2=True,
    )

    first_dec = records[0].decisions[0]

    if first_dec.e_q is not None and 0 <= action_slot < len(first_dec.e_q):
        ev = float(first_dec.e_q[action_slot].item())
    else:
        ev = float("nan")

    # PDF from e_q_pdf if available
    pdf = getattr(first_dec, "e_q_pdf", None)
    if pdf is not None and action_slot < pdf.shape[0]:
        action_pdf = pdf[action_slot].float()
        q_values = torch.arange(-42, 43, dtype=torch.float32)
        weights = action_pdf
        wsum = weights.sum() + 1e-12
        cum = torch.cumsum(weights, dim=0) / wsum
        mask_10 = cum <= 0.10
        if mask_10.any():
            cvar_10 = float((q_values * weights * mask_10.float()).sum() / (weights[mask_10].sum() + 1e-12))
        else:
            cvar_10 = float(q_values[0])
    else:
        action_pdf = None
        cvar_10 = float("nan")

    return ev, cvar_10, action_pdf


def p_set_from_pdf(pdf: torch.Tensor, bid_value: int) -> float:
    """P(set) = P(Q < bid_value) from Team 0 perspective — setter wins if bid fails."""
    q_values = torch.arange(-42, 43, dtype=torch.float32)
    mask = q_values < bid_value
    return float(pdf[mask].sum().item())


def mark_ev_from_worlds(
    q_per_world: torch.Tensor,  # (n_worlds,)
    bid_value: int,
    pre_t0: int,
    pre_t1: int,
) -> float:
    """Compute mark_ev from per-world Q values (scalar, Team 0 perspective)."""
    remaining_total = 42 - pre_t0 - pre_t1
    remaining_t0 = (q_per_world.float() + remaining_total) / 2.0
    final_t0 = (float(pre_t0) + remaining_t0).clamp(0.0, 42.0)
    mm = float(mark_multiplier(bid_value))

    contract_pts = 42 if bid_value == 84 else bid_value
    if bid_value < 42:
        made = (final_t0 >= float(contract_pts)).float()
    else:
        made = (final_t0 == 42.0).float()

    # mark_ev from Team 0 perspective: +mm if made, -mm if not
    mark_util = mm * made - mm * (1.0 - made)
    return float(mark_util.mean().item())


# ---------------------------------------------------------------------------
# Phase classification
# ---------------------------------------------------------------------------

def snap_phase(snap: dict) -> str:
    trick_num = snap.get("_trick_num", 0)
    if trick_num <= 2:
        return "early"
    if trick_num <= 5:
        return "mid"
    return "late"


# ---------------------------------------------------------------------------
# Main probe runner
# ---------------------------------------------------------------------------

def run_paired_contrasts(
    snapshots_by_bid: dict[int, list[dict]],
    model,
    n_samples: int,
    device: str,
    batch_size: int = 100,
) -> list[dict]:
    """Run paired contrast for each snapshot. Returns list of contrast rows."""
    rows = []
    total_count = sum(len(v) for v in snapshots_by_bid.values())
    done = 0

    for bid in HIGH_BIDS:
        snaps = snapshots_by_bid.get(bid, [])
        print(f"\n[probe] bid={bid}: {len(snaps)} snapshots to evaluate", flush=True)

        for i, snap in enumerate(snaps):
            pounce_slot, pounce_did, decline_slot, decline_did = select_pounce_decline(snap)
            if pounce_slot is None:
                continue

            bid_value = int(snap.get("_bid_value", bid))
            decl_id = int(snap["decl_id"])
            setter = int(snap.get("_setter_player", -1))
            trick_num = int(snap.get("_trick_num", 0))
            count_exposed = int(snap.get("_count_exposed", 0))
            bidder_team_led = int(snap.get("_bidder_team_led", 0))
            phase = snap_phase(snap)

            print(
                f"  [{done+1:4d}/{total_count}] bid={bid_value} decl={decl_id} "
                f"trick={trick_num} phase={phase} count={count_exposed}pts "
                f"pounce_slot={pounce_slot}(did={pounce_did}) "
                f"decline_slot={decline_slot}(did={decline_did})",
                end=" ",
                flush=True,
            )

            t0 = time.perf_counter()
            try:
                ev_pounce, cvar_pounce, pdf_pounce = eval_action_eq(
                    model, snap, pounce_slot, n_samples, device
                )
                ev_decline, cvar_decline, pdf_decline = eval_action_eq(
                    model, snap, decline_slot, n_samples, device
                )
            except Exception as exc:
                print(f"ERROR: {exc}", flush=True)
                done += 1
                continue

            elapsed = time.perf_counter() - t0

            # EV delta: pounce - decline (Team 0 perspective)
            # Setter (Team 1) wants LOWER Team 0 EV
            # Pounce is better for setter if pounce_ev < decline_ev (i.e., delta < 0)
            ev_delta_t0 = ev_pounce - ev_decline
            ev_delta_setter = -ev_delta_t0  # positive = pounce better for setter

            cvar_delta = cvar_pounce - cvar_decline

            # p_set from PDFs
            p_set_pounce = p_set_from_pdf(pdf_pounce, bid_value) if pdf_pounce is not None else float("nan")
            p_set_decline = p_set_from_pdf(pdf_decline, bid_value) if pdf_decline is not None else float("nan")
            p_set_delta = p_set_pounce - p_set_decline  # positive = pounce better for setter

            print(
                f"EV_delta(setter)={ev_delta_setter:+.2f} "
                f"p_set_delta={p_set_delta:+.4f} "
                f"t={elapsed:.1f}s",
                flush=True,
            )

            rows.append({
                "bid_bucket": bid_value,
                "decl_id": decl_id,
                "seed": int(snap.get("_seed", -1)),
                "game_idx": int(snap.get("_game_idx", -1)),
                "decision_idx": int(snap.get("_decision_idx", -1)),
                "setter_player": setter,
                "trick_num": trick_num,
                "phase": phase,
                "count_exposed_pts": count_exposed,
                "count_in_current_trick": int(snap.get("_count_in_current", 0)),
                "count_in_last_trick": int(snap.get("_count_in_last", 0)),
                "bidder_team_led": bidder_team_led,
                "pounce_slot": pounce_slot,
                "pounce_did": pounce_did,
                "decline_slot": decline_slot,
                "decline_did": decline_did,
                # EV (Team 0 perspective)
                "ev_pounce_t0": round(ev_pounce, 4) if math.isfinite(ev_pounce) else None,
                "ev_decline_t0": round(ev_decline, 4) if math.isfinite(ev_decline) else None,
                "ev_delta_t0": round(ev_delta_t0, 4) if math.isfinite(ev_delta_t0) else None,
                # Setter perspective (negated)
                "ev_delta_setter": round(ev_delta_setter, 4) if math.isfinite(ev_delta_setter) else None,
                # p_set
                "p_set_pounce": round(p_set_pounce, 5) if math.isfinite(p_set_pounce) else None,
                "p_set_decline": round(p_set_decline, 5) if math.isfinite(p_set_decline) else None,
                "p_set_delta": round(p_set_delta, 5) if math.isfinite(p_set_delta) else None,
                # CVaR
                "cvar_10_pounce": round(cvar_pounce, 4) if math.isfinite(cvar_pounce) else None,
                "cvar_10_decline": round(cvar_decline, 4) if math.isfinite(cvar_decline) else None,
                "cvar_delta": round(cvar_delta, 4) if math.isfinite(cvar_delta) else None,
            })
            done += 1

    return rows


# ---------------------------------------------------------------------------
# Analysis and summary
# ---------------------------------------------------------------------------

def ci95(arr: np.ndarray) -> tuple[float, float]:
    if len(arr) < 2:
        return float("nan"), float("nan")
    se = arr.std(ddof=1) / np.sqrt(len(arr))
    return float(arr.mean() - 1.96 * se), float(arr.mean() + 1.96 * se)


def slice_stats(rows: list[dict], label: str) -> dict:
    """Compute per-slice statistics."""
    valid = [r for r in rows if r["ev_delta_setter"] is not None]
    n = len(valid)
    if n == 0:
        return {"label": label, "N": 0}

    ev_arr = np.array([r["ev_delta_setter"] for r in valid])
    ps_arr = np.array([r["p_set_delta"] for r in valid if r["p_set_delta"] is not None])
    cvar_arr = np.array([r["cvar_delta"] for r in valid if r["cvar_delta"] is not None])

    ev_lo, ev_hi = ci95(ev_arr)
    ps_lo, ps_hi = ci95(ps_arr) if len(ps_arr) > 1 else (float("nan"), float("nan"))

    return {
        "label": label,
        "N": n,
        "ev_delta_setter_mean": round(float(ev_arr.mean()), 4),
        "ev_delta_t_stat": round(float(ev_arr.mean() / (ev_arr.std(ddof=1) / math.sqrt(n) + 1e-12)), 4) if n > 1 else None,
        "ev_delta_ci95_lo": round(ev_lo, 4),
        "ev_delta_ci95_hi": round(ev_hi, 4),
        "ev_direction": "pounce_better_for_setter" if ev_arr.mean() > 0 else "decline_better_for_setter",
        "pct_pounce_better": round(float(100 * (ev_arr > 0).mean()), 2),
        "p_set_delta_mean": round(float(ps_arr.mean()), 5) if len(ps_arr) else None,
        "p_set_delta_ci95_lo": round(ps_lo, 5) if math.isfinite(ps_lo) else None,
        "p_set_delta_ci95_hi": round(ps_hi, 5) if math.isfinite(ps_hi) else None,
        "cvar_delta_mean": round(float(cvar_arr.mean()), 4) if len(cvar_arr) else None,
    }


def propose_verdict(all_rows: list[dict], by_bid: dict[int, list[dict]]) -> str:
    """Propose a claim-ledger status based on cross-bid evidence."""
    bid_verdicts = {}
    for bid in HIGH_BIDS:
        rows = by_bid.get(bid, [])
        valid = [r for r in rows if r["ev_delta_setter"] is not None]
        n = len(valid)
        if n < 20:
            bid_verdicts[bid] = "underpowered"
            continue
        ev_arr = np.array([r["ev_delta_setter"] for r in valid])
        se = ev_arr.std(ddof=1) / math.sqrt(n)
        ci_lo = float(ev_arr.mean() - 1.96 * se)
        ci_hi = float(ev_arr.mean() + 1.96 * se)
        if ci_lo > 0:
            bid_verdicts[bid] = "supported"
        elif ci_hi < 0:
            bid_verdicts[bid] = "contradicted"
        else:
            bid_verdicts[bid] = "context-limited"

    supported_bids = [b for b, v in bid_verdicts.items() if v == "supported"]
    underpowered_bids = [b for b, v in bid_verdicts.items() if v == "underpowered"]

    if len(underpowered_bids) == 4:
        return "underpowered"
    if len(supported_bids) == 4:
        return "supported"
    if len(supported_bids) >= 2 and len(underpowered_bids) == 0:
        return "context-limited"
    if all(v in ("supported", "context-limited") for v in bid_verdicts.values() if v != "underpowered"):
        return "context-limited"
    return "context-limited"


def summarise(rows: list[dict], snapshots_by_bid: dict) -> dict:
    """Build full summary JSON."""
    by_bid: dict[int, list[dict]] = {b: [] for b in HIGH_BIDS}
    for r in rows:
        b = int(r["bid_bucket"])
        if b in by_bid:
            by_bid[b].append(r)

    summary: dict[str, Any] = {}

    # Overall
    overall = slice_stats(rows, "overall")
    summary["overall"] = overall

    # Per bid
    summary["by_bid"] = {}
    for bid in HIGH_BIDS:
        summary["by_bid"][str(bid)] = slice_stats(by_bid[bid], f"bid={bid}")

    # Phase slices (all bids combined)
    for phase in ["early", "mid", "late"]:
        sub = [r for r in rows if r["phase"] == phase]
        summary[f"phase_{phase}"] = slice_stats(sub, f"phase={phase}")

    # Count-at-stake slices
    for count_pts, label in [(5, "count_5pts"), (10, "count_10pts")]:
        sub = [r for r in rows if r["count_exposed_pts"] == count_pts]
        summary[label] = slice_stats(sub, label)

    # Trick leadership slices
    summary["bidder_team_led"] = slice_stats(
        [r for r in rows if r["bidder_team_led"] == 1], "bidder_team_led"
    )
    summary["setter_team_led"] = slice_stats(
        [r for r in rows if r["bidder_team_led"] == 0], "setter_team_led"
    )

    # Mining stats
    summary["mining"] = {
        str(bid): {
            "n_snapshots_mined": len(snapshots_by_bid.get(bid, [])),
            "n_contrasts_run": len(by_bid[bid]),
        }
        for bid in HIGH_BIDS
    }

    # Verdict
    summary["claim"] = "ch12-setter-pounce-high-bid-off"
    summary["bid_values_tested"] = HIGH_BIDS
    summary["claim_ledger_impact"] = propose_verdict(rows, by_bid)
    summary["bid_level_verdicts"] = {}
    for bid in HIGH_BIDS:
        bid_rows = by_bid[bid]
        valid = [r for r in bid_rows if r["ev_delta_setter"] is not None]
        n = len(valid)
        if n < 20:
            summary["bid_level_verdicts"][str(bid)] = "underpowered"
            continue
        ev_arr = np.array([r["ev_delta_setter"] for r in valid])
        se = ev_arr.std(ddof=1) / math.sqrt(n)
        ci_lo = float(ev_arr.mean() - 1.96 * se)
        ci_hi = float(ev_arr.mean() + 1.96 * se)
        if ci_lo > 0:
            summary["bid_level_verdicts"][str(bid)] = "supported"
        elif ci_hi < 0:
            summary["bid_level_verdicts"][str(bid)] = "contradicted"
        else:
            summary["bid_level_verdicts"][str(bid)] = "context-limited"

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--atlas-dir",
        type=Path,
        default=PROJECT_ROOT / "w42/book_validation_v1/wave2/bid_aware_atlas",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt",
    )
    parser.add_argument("--samples", type=int, default=100, help="World samples per decision")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--max-per-bid", type=int, default=500, help="Max snapshots to mine per bid bucket")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent,
    )
    parser.add_argument(
        "--mine-only",
        action="store_true",
        help="Mine snapshots only, do not run inference",
    )
    parser.add_argument(
        "--snapshots-file",
        type=Path,
        default=None,
        help="Load pre-mined snapshots JSONL instead of mining",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: Mine snapshots ---
    snaps_path = out_dir / "pounce_high_bid_snapshots.jsonl"

    if args.snapshots_file and args.snapshots_file.exists():
        print(f"Loading pre-mined snapshots from {args.snapshots_file}", flush=True)
        with open(args.snapshots_file) as f:
            all_snaps_flat = [json.loads(line) for line in f if line.strip()]
        snapshots_by_bid: dict[int, list[dict]] = {b: [] for b in HIGH_BIDS}
        for snap in all_snaps_flat:
            b = int(snap.get("_bid_value", 0))
            if b in snapshots_by_bid:
                snapshots_by_bid[b].append(snap)
    else:
        print("=== Phase 1: Mining pounce-eligible snapshots from bid-aware atlas ===", flush=True)
        t_mine = time.perf_counter()
        snapshots_by_bid = mine_all_high_bid_snapshots(
            args.atlas_dir,
            bids=HIGH_BIDS,
            max_per_bid=args.max_per_bid,
        )

        # Write to JSONL
        all_snaps_flat = []
        for bid in HIGH_BIDS:
            all_snaps_flat.extend(snapshots_by_bid[bid])
        with open(snaps_path, "w") as f:
            for snap in all_snaps_flat:
                f.write(json.dumps(snap) + "\n")
        print(f"Wrote {len(all_snaps_flat)} snapshots to {snaps_path}", flush=True)
        print(f"Mining done in {time.perf_counter() - t_mine:.1f}s", flush=True)

    # Print mining summary
    print("\nMining summary:", flush=True)
    for bid in HIGH_BIDS:
        n = len(snapshots_by_bid[bid])
        print(f"  bid={bid}: {n} pounce-eligible snapshots", flush=True)

    if args.mine_only:
        print("Mine-only mode. Exiting before inference.", flush=True)
        return 0

    # --- Phase 2: Load model ---
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"CUDA unavailable; falling back to {device}.", flush=True)
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
        print("MPS unavailable; falling back to CPU.", flush=True)

    ckpt = args.checkpoint
    if not ckpt.exists():
        print(f"Error: checkpoint not found: {ckpt}", flush=True)
        return 1

    from forge.eq.oracle import Stage1Oracle
    print(f"\nLoading model from {ckpt} on {device}...", flush=True)
    oracle = Stage1Oracle(str(ckpt), device=device, compile=False)
    print("Model loaded.", flush=True)

    # --- Phase 3: Run paired contrasts ---
    print(f"\n=== Phase 2: Paired-contrast probe (n_samples={args.samples}) ===", flush=True)
    t_probe = time.perf_counter()
    contrast_rows = run_paired_contrasts(
        snapshots_by_bid,
        oracle.model,
        n_samples=args.samples,
        device=device,
    )
    t_probe_elapsed = time.perf_counter() - t_probe
    print(f"\nProbe complete in {t_probe_elapsed:.1f}s ({len(contrast_rows)} valid pairs)", flush=True)

    if not contrast_rows:
        print("No valid pairs produced. Exiting.", flush=True)
        return 1

    # --- Phase 4: Save CSV ---
    csv_path = out_dir / "paired_contrasts.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(contrast_rows[0].keys()))
        writer.writeheader()
        writer.writerows(contrast_rows)
    print(f"Saved {csv_path}", flush=True)

    # --- Phase 5: Summary ---
    summary = summarise(contrast_rows, snapshots_by_bid)

    # Slice breakdown CSV
    slice_rows = []
    for key in ["overall", "phase_early", "phase_mid", "phase_late",
                "count_5pts", "count_10pts", "bidder_team_led", "setter_team_led"]:
        if key in summary:
            slice_rows.append(summary[key])
    for bid in HIGH_BIDS:
        if str(bid) in summary.get("by_bid", {}):
            slice_rows.append(summary["by_bid"][str(bid)])

    slice_csv = out_dir / "slice_breakdown.csv"
    if slice_rows:
        with slice_csv.open("w", newline="") as fh:
            all_keys = list(slice_rows[0].keys())
            for r in slice_rows[1:]:
                for k in r:
                    if k not in all_keys:
                        all_keys.append(k)
            writer = csv.DictWriter(fh, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(slice_rows)
        print(f"Saved {slice_csv}", flush=True)

    summary_path = out_dir / "summary.json"
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved {summary_path}", flush=True)

    # --- Phase 6: Manifest ---
    manifest = {
        "bead": "t42-8kbh",
        "claim": "ch12-setter-pounce-high-bid-off",
        "wave": "2.E.2",
        "probe": "pounce_high_bid_paired_contrast",
        "atlas_dir": str(args.atlas_dir),
        "checkpoint": str(ckpt),
        "device": device,
        "n_samples": args.samples,
        "max_per_bid": args.max_per_bid,
        "bid_values": HIGH_BIDS,
        "n_snapshots_mined": {str(b): len(snapshots_by_bid[b]) for b in HIGH_BIDS},
        "n_contrasts_run": len(contrast_rows),
        "n_contrasts_by_bid": {
            str(b): sum(1 for r in contrast_rows if r["bid_bucket"] == b)
            for b in HIGH_BIDS
        },
        "runtime_probe_seconds": round(t_probe_elapsed, 1),
        "artifacts": {
            "pounce_snapshots": str(snaps_path),
            "paired_contrasts": str(csv_path),
            "slice_breakdown": str(slice_csv),
            "summary": str(summary_path),
        },
        "sibling_bid30_probe": "w42/book_validation_v1/wave2/probes/t42-ntbe_pounce_window_bid30/",
        "command": (
            f"python {Path(__file__).name} "
            f"--atlas-dir {args.atlas_dir} "
            f"--checkpoint {ckpt} "
            f"--samples {args.samples} "
            f"--device {device} "
            f"--max-per-bid {args.max_per_bid}"
        ),
        "claim_ledger_impact": summary.get("claim_ledger_impact"),
        "bid_level_verdicts": summary.get("bid_level_verdicts"),
        "overall_ev_delta_setter": summary.get("overall", {}).get("ev_delta_setter_mean"),
        "overall_ev_ci95": [
            summary.get("overall", {}).get("ev_delta_ci95_lo"),
            summary.get("overall", {}).get("ev_delta_ci95_hi"),
        ],
    }

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Saved {manifest_path}", flush=True)

    # --- Print headline ---
    print("\n=== HEADLINE NUMBERS ===", flush=True)
    ov = summary.get("overall", {})
    print(f"Total paired contrasts: {ov.get('N', 0)}", flush=True)
    print(
        f"Overall EV delta (setter, pounce-decline): {ov.get('ev_delta_setter_mean', 'n/a'):+} "
        f"CI95 [{ov.get('ev_delta_ci95_lo', 'n/a')}, {ov.get('ev_delta_ci95_hi', 'n/a')}]",
        flush=True,
    )
    print(f"% pounce better by EV:  {ov.get('pct_pounce_better', 'n/a')}%", flush=True)
    print(f"p_set delta (mean):     {ov.get('p_set_delta_mean', 'n/a')}", flush=True)

    print("\n=== BY BID BUCKET ===", flush=True)
    for bid in HIGH_BIDS:
        bid_stat = summary.get("by_bid", {}).get(str(bid), {})
        print(
            f"  bid={bid}: N={bid_stat.get('N', 0)} "
            f"EV_delta={bid_stat.get('ev_delta_setter_mean', 'n/a'):+} "
            f"CI95=[{bid_stat.get('ev_delta_ci95_lo', 'n/a')}, {bid_stat.get('ev_delta_ci95_hi', 'n/a')}] "
            f"pounce_better={bid_stat.get('pct_pounce_better', 'n/a')}% "
            f"verdict={summary.get('bid_level_verdicts', {}).get(str(bid), 'n/a')}",
            flush=True,
        )

    print(f"\nOverall claim verdict: {summary.get('claim_ledger_impact')}", flush=True)
    print(f"Wave 2.E bid=30 reference: EV_delta(setter)=−3.09 (decline_better_for_setter)", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
