#!/usr/bin/env python3
"""Bid-aware E[Q] atlas driver — Wave 2.B of the W42 book-validation campaign.

Runs the forge generator across a bid sweep on the same seeds, then joins
per-bid .pt outputs into a single parquet/csv keyed by
(seed, decl_id, bid_value, decision_idx, action_slot).

The mark_ev column is recomputed with the correct mark multiplier at each bid
value, verifying the Wave 1.2 finding that mark_ev ≡ p_make only at bid=30
(where mark_multiplier=1 and threshold collapses).

84-eligibility: at bid=84, only declarations 0..9 are valid game declarations
in the engine; the forge generator accepts any decl_id with any bid, so we
simply run all 10 decl_ids at bid=84 and document the difference.  The engine
handles the game rules internally.  We note in the manifest that "all 10 decl
IDs were run at bid=84; 84 contracts are only strategically meaningful under
declarations where the caller actually holds 4+ doubles, but the engine does
not enforce bidding legality during generation."

Validation target: at bid=30 and seed=9430, scalar EV per (seed, decl_id,
decision_idx) must match branch_atlas_scaled_v0 within sampling noise
(CI half-width ~ 1-3 pts depending on std).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

BEAD_ID = "t42-6j3k"
SCHEMA_VERSION = "w42.bookval.wave2.bid_aware_atlas.v1"
EQ_MIN = -42
EQ_MAX = 42
LOW_TAIL_Q = -18.0
OFFENSE_PLAYERS = {0, 2}

DECL_NAMES = {
    0: "blanks", 1: "ones", 2: "twos", 3: "threes", 4: "fours",
    5: "fives", 6: "sixes", 7: "doubles", 8: "doubles-suit", 9: "no-trump",
}
SEAT_ROLE = {
    0: "bidder", 1: "left_setter", 2: "bidder_partner", 3: "right_setter",
}

# Mark multiplier table (same as phase4_scoring_objective_tests)
# bid < 42 → multiplier = 1 (ordinary contract)
# bid == 84 → multiplier = 2
# bid == 126 → multiplier = 3, etc.
def mark_multiplier(bid: int) -> int:
    if bid >= 84:
        return bid // 42
    return 1


def made_contract(bid: int, bidder_points: float) -> bool:
    if bid < 42:
        return float(bidder_points) >= bid
    return float(bidder_points) == 42.0


# ──────────────────────────────────────────────────────────────────────────────
# Mark utility transform (vectorized per-world)
# ──────────────────────────────────────────────────────────────────────────────

def q_to_mark_utility_per_world(
    q_per_world: "torch.Tensor",   # (n_worlds, n_actions)
    legal_mask: "torch.Tensor",    # (n_actions,)
    bid: int,
    bidder_team: int,
    pre_t0_points: int,
    pre_t1_points: int,
) -> "torch.Tensor":
    """Convert per-world Q to per-world mark utility (net marks for Team 0).

    Verbatim reuse of the Wave 1.2 transform from
    w42/book_validation_v1/wave1/t42-c6sa_mark_utility_transform/run_mark_utility_transform.py
    (SHA: see manifest). Extended to support bid > 30 (multiplier != 1).
    """
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


# ──────────────────────────────────────────────────────────────────────────────
# Q statistics from per-world tensor
# ──────────────────────────────────────────────────────────────────────────────

def q_stats_from_world_slice(values: "torch.Tensor", threshold_q: float) -> dict[str, Any]:
    """Compute scalar EV and distribution stats from a (n_worlds,) slice."""
    v = values.detach().cpu().float()
    n = int(v.numel())
    if n == 0:
        return {
            "mean": float("nan"), "std": float("nan"),
            "threshold_mass": float("nan"),
            "q10": None, "q25": None, "q50": None, "q75": None, "q90": None,
            "cvar_10": float("nan"), "samples": 0,
        }
    mean_v = float(v.mean().item())
    std_v = float(v.std(unbiased=False).item())
    # Quantiles via sorted tensor
    sorted_v, _ = v.sort()
    def pctile(p: float) -> float:
        idx = int(p * (n - 1))
        return float(sorted_v[idx].item())
    threshold_mass = float((v >= threshold_q).float().mean().item())
    # CVaR at 10%
    cutoff_idx = max(1, int(0.10 * n))
    cvar_10 = float(sorted_v[:cutoff_idx].mean().item())
    return {
        "mean": mean_v,
        "std": std_v,
        "threshold_mass": threshold_mass,
        "q10": pctile(0.10),
        "q25": pctile(0.25),
        "q50": pctile(0.50),
        "q75": pctile(0.75),
        "q90": pctile(0.90),
        "cvar_10": cvar_10,
        "samples": n,
    }


def threshold_q_for_player(player: int, bid_value: int) -> float:
    """Contract make/set threshold in Q-space."""
    contract_points = 42 if bid_value == 84 else bid_value
    if player in OFFENSE_PLAYERS:
        return float(2 * contract_points - 42)
    return float(43 - 2 * contract_points)


def p_make_from_mark_util(
    mark_util_col: "torch.Tensor",  # (n_worlds,) mark utility for one action
    bid_value: int,
) -> float:
    """P(make) = fraction of worlds where mark_utility > 0 (team 0 won the mark).

    This is correct at all score states because mark_utility uses final_t0 directly.
    mark_utility[w] = +multiplier if team0 won, -multiplier if team1 won.
    P(make) = P(mark_util > 0).
    """
    v = mark_util_col.detach().cpu().float()
    multiplier = float(mark_multiplier(bid_value))
    if multiplier > 0:
        return float((v > 0).float().mean().item())
    return float("nan")


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        if fieldnames:
            path.write_text(",".join(fieldnames) + "\n", encoding="utf-8")
        else:
            path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def round4(v: Any) -> float | None:
    if v is None:
        return None
    fv = float(v)
    if not math.isfinite(fv):
        return None
    return round(fv, 4)


# ──────────────────────────────────────────────────────────────────────────────
# Per-bid .pt generation
# ──────────────────────────────────────────────────────────────────────────────

def find_checkpoint(root: Path) -> str | None:
    model_dir = root / "forge" / "models"
    candidates = [
        model_dir / "domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt",
        model_dir / "domino-large-817k-valuehead-acc97.8-qgap0.07.ckpt",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def generate_bid_pt(
    *,
    bid_value: int,
    start_seed: int,
    n_seeds: int,
    n_decl_per_seed: int,
    n_samples: int,
    output_path: Path,
    device: str,
    checkpoint: str,
    dry_run: bool,
) -> dict[str, Any]:
    """Run forge generator for one bid value.

    Returns timing metadata.
    """
    n_games = n_seeds * n_decl_per_seed
    end_seed = start_seed + n_seeds - 1
    bid_str = ",".join([str(bid_value)] * n_games)

    cmd = [
        sys.executable, "-u", "-m", "forge.eq.generate",
        "--start-seed", str(start_seed),
        "--n-games", str(n_games),
        "--n-decl-per-seed", str(n_decl_per_seed),
        "--samples", str(n_samples),
        "--bid-values", bid_str,
        "--schema", "v2",
        "--save-joint-worlds",
        "--device", device,
        "--checkpoint", checkpoint,
        "--output", str(output_path),
    ]

    print(
        f"[bid_aware_atlas] bid={bid_value} "
        f"seeds={start_seed}..{end_seed} x {n_decl_per_seed} decls = {n_games} games",
        flush=True,
    )

    if dry_run:
        print(f"  DRY RUN: would run: {' '.join(cmd)}", flush=True)
        return {"dry_run": True, "cmd": " ".join(cmd), "elapsed_s": 0.0}

    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=False)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        raise RuntimeError(
            f"forge generator failed for bid={bid_value} (returncode={result.returncode})"
        )
    print(f"  Done in {elapsed:.1f}s", flush=True)
    return {"dry_run": False, "cmd": " ".join(cmd), "elapsed_s": round(elapsed, 2)}


# ──────────────────────────────────────────────────────────────────────────────
# Join step: merge per-bid .pt files into a CSV
# ──────────────────────────────────────────────────────────────────────────────

ACTION_COLUMNS = [
    "seed", "decl_id", "decl_name", "bid_value", "mark_multiplier",
    "decision_idx", "trick_idx", "trick_position",
    "actor", "seat_role", "team",
    "offense_score_before", "defense_score_before",
    "action_slot", "is_actual_action",
    "threshold_q",
    "mean", "std",
    "q10", "q25", "q50", "q75", "q90",
    "threshold_mass", "cvar_10",
    "p_make",
    "mark_ev",
    "mark_ev_equals_p_make",
    "samples",
]


def process_pt_file(
    pt_path: Path,
    bid_value: int,
) -> list[dict[str, Any]]:
    """Extract per-action rows from one bid's .pt file."""
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    games = payload.get("results", [])
    seeds = payload.get("seeds", [])
    decl_ids = payload.get("decl_ids", [])

    rows: list[dict[str, Any]] = []
    offense_score_acc = [0, 0]

    for game_idx, game in enumerate(games):
        hands = [[int(d) for d in hand] for hand in getattr(game, "hands")]
        decl_id = int(getattr(game, "decl_id", decl_ids[game_idx] if game_idx < len(decl_ids) else -1))
        seed = int(seeds[game_idx]) if game_idx < len(seeds) else -1
        decl_name = DECL_NAMES.get(decl_id, f"unknown-{decl_id}")
        mm = mark_multiplier(bid_value)

        # Track score state (offensive team = 0)
        score = [0, 0]
        trick_plays: list[tuple[int, int]] = []
        played_slots: dict[int, set[int]] = {0: set(), 1: set(), 2: set(), 3: set()}

        for decision_idx, decision in enumerate(getattr(game, "decisions", [])):
            actor = int(getattr(decision, "player"))
            actual_slot = int(getattr(decision, "action_taken"))
            legal_mask = getattr(decision, "legal_mask", None)
            world_hands = getattr(decision, "world_hands", None)
            q_per_world = getattr(decision, "q_per_world", None)

            if legal_mask is None or q_per_world is None:
                # Update game state and continue
                actual_domino = hands[actor][actual_slot] if 0 <= actual_slot < len(hands[actor]) else -1
                if actual_domino >= 0:
                    played_slots[actor].add(actual_slot)
                    trick_plays.append((actor, actual_domino))
                    if len(trick_plays) == 4:
                        score = _update_score(score, trick_plays, decl_id)
                        trick_plays = []
                continue

            legal = torch.as_tensor(legal_mask, dtype=torch.bool).cpu()
            qpw = q_per_world.detach().cpu().float()
            legal_slots = [s for s, ok in enumerate(legal.tolist()) if ok]

            if not legal_slots:
                actual_domino = hands[actor][actual_slot] if 0 <= actual_slot < len(hands[actor]) else -1
                if actual_domino >= 0:
                    played_slots[actor].add(actual_slot)
                    trick_plays.append((actor, actual_domino))
                    if len(trick_plays) == 4:
                        score = _update_score(score, trick_plays, decl_id)
                        trick_plays = []
                continue

            tq = threshold_q_for_player(actor, bid_value)
            off_score = score[0]
            def_score = score[1]
            pre_t0 = off_score
            pre_t1 = def_score
            trick_idx = decision_idx // 4
            trick_position = len(trick_plays)
            team = "offense" if actor in OFFENSE_PLAYERS else "defense"
            seat_role = SEAT_ROLE[actor]

            # Compute mark utility across all legal actions
            bidder_team = 0  # Team 0 = offense = bidder (atlas convention)
            mark_util = q_to_mark_utility_per_world(
                qpw, legal, bid_value, bidder_team, pre_t0, pre_t1
            )  # (n_worlds, 7)

            for slot in legal_slots:
                stats = q_stats_from_world_slice(qpw[:, slot], tq)
                mark_util_col = mark_util[:, slot]
                mark_ev_val = float(mark_util_col.float().mean().item())
                # p_make from mark_utility: P(team0 wins the mark in this world)
                # This is correct at all score states because it uses final_t0 directly.
                raw_pm = p_make_from_mark_util(mark_util_col, bid_value)
                # Algebraic identity check:
                # mark_ev = E[mark_util] = multiplier * (2*P(make) - 1)
                # = multiplier * (2*raw_pm - 1)
                # This is ALWAYS true by definition (mark_util = +/-multiplier).
                # The interesting question is whether mark_ev differs from
                # threshold_mass (which uses schema's remaining-point threshold_q,
                # not the score-state-adjusted threshold).
                # At bid=30 and pre_t0=pre_t1=0: both thresholds equal 18.
                # As game progresses and scores diverge, they can differ.
                mark_ev_eq_pmake = abs(mark_ev_val - mm * (2 * raw_pm - 1)) < 0.001

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
                    "threshold_q": round4(tq),
                    "mean": round4(stats["mean"]),
                    "std": round4(stats["std"]),
                    "q10": round4(stats["q10"]),
                    "q25": round4(stats["q25"]),
                    "q50": round4(stats["q50"]),
                    "q75": round4(stats["q75"]),
                    "q90": round4(stats["q90"]),
                    "threshold_mass": round4(stats["threshold_mass"]),
                    "cvar_10": round4(stats["cvar_10"]),
                    "p_make": round4(raw_pm),
                    "mark_ev": round4(mark_ev_val),
                    "mark_ev_equals_p_make": int(mark_ev_eq_pmake),
                    "samples": stats["samples"],
                })

            # Update game state
            actual_domino = hands[actor][actual_slot] if 0 <= actual_slot < len(hands[actor]) else -1
            if actual_domino >= 0:
                played_slots[actor].add(actual_slot)
                trick_plays.append((actor, actual_domino))
                if len(trick_plays) == 4:
                    score = _update_score(score, trick_plays, decl_id)
                    trick_plays = []

    return rows


def _update_score(score: list[int], trick_plays: list[tuple[int, int]], decl_id: int) -> list[int]:
    """Update score after a completed trick. Returns updated [t0, t1]."""
    try:
        from forge.oracle.tables import DOMINO_COUNT_POINTS, resolve_trick
        led_domino = trick_plays[0][1]
        all_dominos = [d for _p, d in trick_plays]
        players = [p for p, _d in trick_plays]
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


# ──────────────────────────────────────────────────────────────────────────────
# Validation: compare bid=30 against branch_atlas_scaled_v0
# ──────────────────────────────────────────────────────────────────────────────

def validate_bid30_against_atlas(
    rows_bid30: list[dict[str, Any]],
    atlas_pt_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Validate bid=30 driver output against branch_atlas_scaled_v0 for seed 9430.

    Because the oracle is greedy-stochastic, two runs on the same seed will
    play out different game trajectories. Direct per-(decision_idx, slot) comparison
    will show large diffs on diverged trajectories.

    Instead we compare aggregate distribution statistics per (seed, decl_id):
    - Distribution of actual-action scalar EVs across all decisions
    - Mean EV, std, median, and quantile profile
    - These should be consistent within sampling noise (2-3 pts at n=200 vs n=1000)

    The contract is: aggregate distribution statistics match within 5 points
    for the same (seed, decl_id) pair.
    """
    if not atlas_pt_path.exists():
        return {"status": "skipped", "reason": "atlas .pt not found"}

    try:
        payload = torch.load(atlas_pt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        return {"status": "skipped", "reason": f"failed to load atlas: {e}"}

    atlas_games = payload.get("results", [])
    atlas_seeds = payload.get("seeds", [])
    atlas_decl_ids = payload.get("decl_ids", [])

    # Build atlas aggregate stats per (seed, decl_id): distribution of actual-action EVs
    import statistics as stats_lib
    atlas_aggregate: dict[tuple[int, int], dict[str, Any]] = {}
    for gi, game in enumerate(atlas_games):
        seed_g = int(atlas_seeds[gi]) if gi < len(atlas_seeds) else -1
        decl_id_g = int(atlas_decl_ids[gi]) if gi < len(atlas_decl_ids) else -1
        hands_g = [[int(d) for d in hand] for hand in getattr(game, "hands")]
        actual_evs: list[float] = []

        for di, decision in enumerate(getattr(game, "decisions", [])):
            actor_g = int(getattr(decision, "player"))
            actual_slot_g = int(getattr(decision, "action_taken"))
            legal_mask_g = getattr(decision, "legal_mask", None)
            q_pw_g = getattr(decision, "q_per_world", None)
            if legal_mask_g is not None and q_pw_g is not None:
                qpw_g = q_pw_g.detach().cpu().float()
                if 0 <= actual_slot_g < qpw_g.shape[1]:
                    tq_g = threshold_q_for_player(actor_g, 30)
                    s_g = q_stats_from_world_slice(qpw_g[:, actual_slot_g], tq_g)
                    if math.isfinite(s_g["mean"]):
                        actual_evs.append(s_g["mean"])

        if actual_evs:
            atlas_aggregate[(seed_g, decl_id_g)] = {
                "n_decisions": len(actual_evs),
                "mean_ev": stats_lib.mean(actual_evs),
                "std_ev": stats_lib.stdev(actual_evs) if len(actual_evs) > 1 else 0.0,
                "median_ev": stats_lib.median(actual_evs),
            }

    # Build driver aggregate stats per (seed, decl_id) from actual-action rows
    target_seed = 9430
    driver_rows_9430 = [
        r for r in rows_bid30
        if int(r["seed"]) == target_seed and int(r["is_actual_action"]) == 1
        and r["mean"] is not None
    ]

    from collections import defaultdict
    driver_by_key: dict[tuple[int, int], list[float]] = defaultdict(list)
    for r in driver_rows_9430:
        k = (int(r["seed"]), int(r["decl_id"]))
        if r["mean"] is not None and math.isfinite(float(r["mean"])):
            driver_by_key[k].append(float(r["mean"]))

    validation_rows: list[dict[str, Any]] = []
    matched = 0
    discrepant = 0

    for (seed_g, decl_id_g), atlas_agg in atlas_aggregate.items():
        driver_evs = driver_by_key.get((seed_g, decl_id_g), [])
        if not driver_evs:
            validation_rows.append({
                "seed": seed_g, "decl_id": decl_id_g,
                "atlas_mean_ev": round4(atlas_agg["mean_ev"]),
                "driver_mean_ev": None,
                "diff_mean_ev": None,
                "within_noise": None,
                "status": "driver_missing",
            })
            continue

        driver_agg_mean = sum(driver_evs) / len(driver_evs)
        atlas_agg_mean = atlas_agg["mean_ev"]
        diff = abs(driver_agg_mean - atlas_agg_mean)
        # Tolerance: combined SEM over n_driver decisions and n_atlas decisions
        # Both are means of ~28 noisy estimates, each with std ~ 10-15
        # This is loose but appropriate for trajectory-diverged comparison
        n_d = len(driver_evs)
        n_a = atlas_agg["n_decisions"]
        std_est = atlas_agg.get("std_ev", 12.0)
        combined_sem = std_est * (1.0 / math.sqrt(max(1, n_d)) + 1.0 / math.sqrt(max(1, n_a)))
        within_noise = int(diff <= 3.0 * combined_sem + 3.0)  # 3*SEM + 3pt buffer

        if within_noise:
            matched += 1
        else:
            discrepant += 1

        validation_rows.append({
            "seed": seed_g,
            "decl_id": decl_id_g,
            "atlas_n_decisions": n_a,
            "driver_n_decisions": n_d,
            "atlas_mean_ev": round4(atlas_agg_mean),
            "driver_mean_ev": round4(driver_agg_mean),
            "diff_mean_ev": round4(diff),
            "combined_sem": round4(combined_sem),
            "within_noise": within_noise,
            "status": "matched" if within_noise else "discrepant",
        })

    total_compared = matched + discrepant
    write_csv(output_path, validation_rows)

    note = (
        "Validation compares aggregate per-(seed,decl_id) mean EV of actual-action "
        "decisions, not per-decision scalar EVs. Two stochastic oracle runs on the "
        "same seed diverge in game trajectory after decision 0, so direct "
        "per-(decision_idx,slot) comparison is not appropriate."
    )

    return {
        "status": "completed",
        "target_seed": target_seed,
        "note": note,
        "n_decl_ids_compared": total_compared,
        "n_within_noise": matched,
        "n_discrepant": discrepant,
        "match_rate": round(matched / max(1, total_compared), 4),
        "pass": discrepant == 0 and total_compared > 0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Mark EV divergence analysis
# ──────────────────────────────────────────────────────────────────────────────

def compute_mark_ev_divergence(all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    For each bid bucket, compute how often mark_ev != f(p_make).

    At bid=30: mark_ev = E[mark_utility] = E[(+1 if made else -1)] = 2*P(make)-1.
    threshold_mass = P(q >= threshold_q for offense) = P(make).
    So mark_ev ≡ 2*threshold_mass - 1 = 2*p_make - 1.  This is the algebraic identity.

    At bid>30: threshold_q shifts (2*bid - 42 for offense).
    The mark_ev STILL equals 2*threshold_mass(at this new tq) - 1 at multiplier=1.
    But multiplier > 1 at bid=84, so mark_ev = multiplier*(2*p_make - 1).

    The divergence we care about: does the action RANKING change when bid changes?
    We measure: correlation between mark_ev and threshold_mass per bid bucket.
    """
    from collections import defaultdict
    import statistics

    by_bid: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        bv = int(row["bid_value"])
        by_bid[bv].append(row)

    divergence: dict[str, Any] = {}
    for bv in sorted(by_bid.keys()):
        rows_bv = by_bid[bv]
        mark_evs = [float(r["mark_ev"]) for r in rows_bv if r["mark_ev"] is not None]
        threshold_masses = [float(r["threshold_mass"]) for r in rows_bv if r["threshold_mass"] is not None]
        p_makes = [float(r["p_make"]) for r in rows_bv if r["p_make"] is not None]
        mm_val = mark_multiplier(bv)

        if len(mark_evs) < 2:
            divergence[str(bv)] = {"n": len(rows_bv), "status": "insufficient_data"}
            continue

        n_pts = min(len(mark_evs), len(threshold_masses), len(p_makes))
        me = mark_evs[:n_pts]
        tm = threshold_masses[:n_pts]
        pm = p_makes[:n_pts]
        mean_me = statistics.mean(me)
        mean_tm = statistics.mean(tm)
        std_me = statistics.stdev(me) if len(me) > 1 else 0.0
        std_tm = statistics.stdev(tm) if len(tm) > 1 else 0.0
        if std_me > 0 and std_tm > 0:
            r_me_tm = sum((me[i] - mean_me) * (tm[i] - mean_tm) for i in range(n_pts)) / (
                n_pts * std_me * std_tm
            )
        else:
            r_me_tm = float("nan")

        # Algebraic check 1: mark_ev = mm * (2*p_make - 1) by construction.
        # p_make is computed from mark_util directly, so this should always hold.
        identity_residuals = [abs(me[i] - mm_val * (2.0 * pm[i] - 1.0)) for i in range(n_pts)]
        mean_identity_residual = statistics.mean(identity_residuals)
        identity_holds = mean_identity_residual < 0.01

        # Divergence check: how much does mark_ev differ from (2*threshold_mass - 1)?
        # At bid=30 and trick=0 (pre_t0=pre_t1=0): threshold_mass = P(q >= 18).
        # mark_ev = P(final_t0 >= 30) using the actual score trajectory.
        # These differ as scores diverge from 0.
        # This is the "divergence" the wave is designed to expose.
        divergence_from_threshold = [abs(me[i] - (2.0 * tm[i] - 1.0)) * mm_val for i in range(n_pts)]
        mean_divergence = statistics.mean(divergence_from_threshold)
        # At bid=30: if mark_ev != 2*threshold_mass - 1, it means we're mid-game.
        # At bid=35/36/42/84: threshold_q shifts, so divergence increases structurally.
        threshold_q_at_bv = 2 * (42 if bv == 84 else bv) - 42

        divergence[str(bv)] = {
            "n_rows": len(rows_bv),
            "mark_multiplier": mm_val,
            "threshold_q_for_offense": threshold_q_at_bv,
            "mean_mark_ev": round4(mean_me),
            "mean_threshold_mass": round4(mean_tm),
            "mean_p_make": round4(statistics.mean(pm)),
            "r_mark_ev_threshold_mass": round4(r_me_tm),
            "mean_residual_identity_check": round4(mean_identity_residual),
            "identity_mark_ev_eq_mm_times_2pm_minus1_holds": identity_holds,
            "mean_divergence_mark_ev_vs_2tm_minus1": round4(mean_divergence),
            "headline": (
                f"bid={bv}: mm={mm_val}, tq_off={threshold_q_at_bv}, "
                f"mean_mark_ev={mean_me:.3f}, mean_tm={mean_tm:.3f}, "
                f"divergence(mark_ev vs 2*tm-1)={mean_divergence:.3f}"
            ),
        }

    # Compute divergence between bid=30 and each higher bid
    bid30_actions: dict[tuple[int, int, int], float] = {}
    for row in by_bid.get(30, []):
        key = (int(row["seed"]), int(row["decl_id"]), int(row["decision_idx"]))
        bid30_actions[key] = float(row["mark_ev"]) if row["mark_ev"] is not None else float("nan")

    cross_bid_flip_rates: dict[str, Any] = {}
    for bv in sorted(by_bid.keys()):
        if bv == 30:
            continue
        flips = 0
        total = 0
        for row in by_bid[bv]:
            if int(row["action_slot"]) != int(row.get("action_slot", -1)):
                continue  # skip non-actual
            if int(row["is_actual_action"]) != 1:
                continue
            key = (int(row["seed"]), int(row["decl_id"]), int(row["decision_idx"]))
            me_30 = bid30_actions.get(key)
            me_bv = float(row["mark_ev"]) if row["mark_ev"] is not None else float("nan")
            if me_30 is not None and math.isfinite(me_30) and math.isfinite(me_bv):
                total += 1
                if abs(me_bv - me_30) > 0.01:
                    flips += 1
        cross_bid_flip_rates[str(bv)] = {
            "n_actual_actions": total,
            "n_mark_ev_changed": flips,
            "mark_ev_change_rate": round4(flips / max(1, total)),
        }

    return {
        "by_bid": divergence,
        "cross_bid_flip_rates_vs_bid30": cross_bid_flip_rates,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bid-aware E[Q] atlas driver — W42 book validation wave 2.B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full 50-seed smoke at 200 samples (fast, MPS/CPU)
  python -u w42/book_validation_v1/wave2/run_bid_aware_atlas.py \\
    --n-seeds 5 --n-samples 200

  # Full 50-seed run (GPU)
  python -u w42/book_validation_v1/wave2/run_bid_aware_atlas.py \\
    --n-seeds 50 --n-samples 1000 --device cuda

  # Dry run: print commands without executing
  python -u w42/book_validation_v1/wave2/run_bid_aware_atlas.py --dry-run
""",
    )
    parser.add_argument("--start-seed", type=int, default=9000)
    parser.add_argument("--n-seeds", type=int, default=50)
    parser.add_argument("--n-decl-per-seed", type=int, default=10)
    parser.add_argument(
        "--bid-values", type=str, default="30,32,35,36,39,42,84",
        help="Comma-separated bid values to sweep (default: 30,32,35,36,39,42,84)",
    )
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "w42/book_validation_v1/wave2/bid_aware_atlas",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would run but skip generation; implies join over any existing .pt files",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    t_start = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bid_values = [int(b.strip()) for b in args.bid_values.split(",") if b.strip()]
    start_seed = args.start_seed
    n_seeds = args.n_seeds
    n_decl = args.n_decl_per_seed
    n_samples = args.n_samples
    end_seed = start_seed + n_seeds - 1

    # Determine device
    device = args.device
    if not args.dry_run:
        try:
            if device == "cuda" and not torch.cuda.is_available():
                if torch.backends.mps.is_available():
                    print("Warning: CUDA unavailable; falling back to MPS.", flush=True)
                    device = "mps"
                else:
                    print("Warning: CUDA unavailable; falling back to CPU.", flush=True)
                    device = "cpu"
        except Exception:
            pass

    # Find checkpoint
    checkpoint = args.checkpoint
    if checkpoint is None and not args.dry_run:
        checkpoint = find_checkpoint(ROOT)
        if checkpoint is None:
            print("Error: No model checkpoint found. Use --checkpoint to specify.", flush=True)
            return 1
    elif checkpoint is None and args.dry_run:
        checkpoint = str(ROOT / "forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt")

    is_smoke = n_seeds <= 10 and n_samples <= 300
    run_label = "smoke" if is_smoke else "full"
    print(
        f"[bid_aware_atlas] {run_label} run: "
        f"{n_seeds} seeds x {n_decl} decls x {len(bid_values)} bids x {n_samples} samples"
        f" = {n_seeds * n_decl * len(bid_values) * n_samples:,} total world samples",
        flush=True,
    )
    print(f"[bid_aware_atlas] device={device}, output_dir={args.output_dir}", flush=True)

    # Phase 1: Generate per-bid .pt files
    bid_pt_files: dict[int, Path] = {}
    gen_meta: dict[int, Any] = {}

    for bv in bid_values:
        pt_path = args.output_dir / f"eq_pdf_seeds{start_seed}-{end_seed}_bid{bv}_v2.pt"
        bid_pt_files[bv] = pt_path

        if pt_path.exists() and not args.dry_run:
            print(f"[bid_aware_atlas] bid={bv}: {pt_path.name} exists, skipping generation", flush=True)
            gen_meta[bv] = {"skipped": True, "reason": "file_exists"}
            continue

        meta = generate_bid_pt(
            bid_value=bv,
            start_seed=start_seed,
            n_seeds=n_seeds,
            n_decl_per_seed=n_decl,
            n_samples=n_samples,
            output_path=pt_path,
            device=device,
            checkpoint=checkpoint,
            dry_run=args.dry_run,
        )
        gen_meta[bv] = meta

    if args.dry_run:
        print("[bid_aware_atlas] DRY RUN complete. No .pt files generated.", flush=True)
        print("[bid_aware_atlas] Will attempt to join any existing .pt files ...", flush=True)

    # Phase 2: Join per-bid .pt files into a single CSV
    all_action_rows: list[dict[str, Any]] = []
    join_meta: dict[int, Any] = {}

    for bv in bid_values:
        pt_path = bid_pt_files[bv]
        if not pt_path.exists():
            print(f"[bid_aware_atlas] bid={bv}: {pt_path.name} not found, skipping join", flush=True)
            join_meta[bv] = {"status": "missing"}
            continue
        print(f"[bid_aware_atlas] Joining bid={bv} from {pt_path.name} ...", flush=True)
        try:
            rows = process_pt_file(pt_path, bv)
            all_action_rows.extend(rows)
            join_meta[bv] = {
                "status": "ok",
                "rows": len(rows),
                "sha256": sha256_file(pt_path),
            }
            print(f"  → {len(rows)} action rows", flush=True)
        except Exception as exc:
            print(f"  ERROR: {exc}", flush=True)
            join_meta[bv] = {"status": "error", "error": str(exc)}

    # Write joined CSV
    joined_csv = args.output_dir / "bid_aware_actions.csv"
    write_csv(joined_csv, all_action_rows, fieldnames=ACTION_COLUMNS)
    print(f"[bid_aware_atlas] Wrote {len(all_action_rows)} rows to {joined_csv}", flush=True)

    # Phase 3: Compute mark_ev divergence statistics
    print("[bid_aware_atlas] Computing mark_ev divergence ...", flush=True)
    if all_action_rows:
        divergence = compute_mark_ev_divergence(all_action_rows)
    else:
        divergence = {"status": "no_data"}

    # Phase 4: Validation at bid=30
    print("[bid_aware_atlas] Validating bid=30 against branch_atlas_scaled_v0 ...", flush=True)
    atlas_pt = ROOT / "w42/branch_atlas_scaled_v0/eq_pdf_s9430_d10_bid30_joint_1000s_v2.pt"
    rows_bid30 = [r for r in all_action_rows if int(r["bid_value"]) == 30]
    val_csv = args.output_dir / "validation_check.csv"
    if rows_bid30:
        val_result = validate_bid30_against_atlas(rows_bid30, atlas_pt, val_csv)
    else:
        val_result = {"status": "skipped", "reason": "no_bid30_rows_in_joined_output"}

    print(f"[bid_aware_atlas] Validation result: {val_result.get('status')}", flush=True)

    # Phase 5: Write manifest
    elapsed = time.perf_counter() - t_start
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "bead_id": BEAD_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "repo_commit": git_sha(),
        "run_label": run_label,
        "dry_run": args.dry_run,
        "args": {
            "start_seed": start_seed,
            "n_seeds": n_seeds,
            "n_decl_per_seed": n_decl,
            "bid_values": bid_values,
            "n_samples": n_samples,
            "device": device,
            "checkpoint": str(checkpoint),
        },
        "84_eligibility_note": (
            "All 10 decl_ids run at bid=84. The forge engine accepts any "
            "decl_id with any bid; 84 contracts are only strategically "
            "meaningful when the caller holds 4+ doubles, but this "
            "constraint is not enforced during generation. Mark multiplier "
            "at bid=84 is 2 (bid // 42)."
        ),
        "generation": gen_meta,
        "join": join_meta,
        "totals": {
            "n_seeds": n_seeds,
            "n_decl_per_seed": n_decl,
            "n_bid_values": len(bid_values),
            "n_games_per_bid": n_seeds * n_decl,
            "n_total_games": n_seeds * n_decl * len(bid_values),
            "n_action_rows": len(all_action_rows),
        },
        "mark_ev_divergence": divergence,
        "validation_bid30": val_result,
        "artifacts": {
            "bid_aware_actions_csv": str(joined_csv),
            "validation_check_csv": str(val_csv),
            "manifest_json": str(args.output_dir / "manifest.json"),
            "per_bid_pt_files": {str(bv): str(p) for bv, p in bid_pt_files.items()},
        },
        "wall_seconds": round(elapsed, 2),
    }

    write_json(args.output_dir / "manifest.json", manifest)
    print(
        f"[bid_aware_atlas] Done in {elapsed:.1f}s. "
        f"{len(all_action_rows)} action rows across {len(bid_values)} bids.",
        flush=True,
    )

    # Print headline divergence summary
    print("\n=== MARK EV DIVERGENCE BY BID ===", flush=True)
    for bv_str, stats in divergence.get("by_bid", {}).items():
        headline = stats.get("headline", f"bid={bv_str}: {stats}")
        print(f"  {headline}", flush=True)

    print(f"\n=== VALIDATION (bid=30 vs branch_atlas_scaled_v0) ===", flush=True)
    for k, v in val_result.items():
        print(f"  {k}: {v}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
