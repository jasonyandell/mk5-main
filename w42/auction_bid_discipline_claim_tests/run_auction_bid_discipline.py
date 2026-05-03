#!/usr/bin/env python3
"""Generated auction-pressure corpus for W42 bidding-discipline claims.

Phase 2 proved the easy part of "bid only enough": for a fixed hand and
declaration, raising the contract threshold cannot improve the same play-outcome
distribution. This phase-3 probe adds auction context. It generates full deals,
estimates candidate contract value for every seat's strongest declarations, and
then builds P0 bid/pass/overcall rows against synthetic partner/opponent high
bids.

This is still a generated proxy, not a logged human auction. It intentionally
keeps partner/opponent contract labels as offline comparison targets.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forge.bidding.estimator import BID_THRESHOLDS, evaluate_bids
from forge.bidding.inference import PolicyModel
from forge.bidding.simulator import simulate_games
from forge.oracle.declarations import DECL_ID_TO_NAME
from forge.oracle.rng import deal_from_seed
from w42.bidding_risk_budget_claim_validation.validate_bidding_risk_budget import (
    domino_name,
    hand_eval,
)


OUT_DIR = ROOT / "w42" / "auction_bid_discipline_claim_tests"
BEAD_ID = "t42-qtwb.1"
PIP_DECLS = tuple(range(7))
HIGH_BID_CONTEXTS = (30, 31, 32, 34, 35, 36)
NATURAL_BUCKET_BIDS = {30, 31, 32, 33, 35, 36}


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def hand_label(hand: list[int]) -> str:
    return ",".join(domino_name(d) for d in hand)


def risk_bucket(unique_exposed_points: int) -> str:
    if unique_exposed_points <= 12:
        return "risk_le_12"
    if unique_exposed_points <= 20:
        return "risk_13_20"
    return "risk_gt_20"


def natural_bid_bucket(bid: int) -> str:
    if bid <= 0:
        return "no_profitable_bid"
    if bid in (30, 31):
        return "natural_30_31"
    if bid in (32, 33):
        return "natural_32_33"
    if bid == 34:
        return "bid_34"
    if bid in (35, 36):
        return "natural_35_36"
    if bid < 42:
        return "high_37_41"
    return "bid_42"


def static_key(ev: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        -int(ev["unique_exposed_points"]),
        int(ev["trump_count"]),
        int(ev["held_count_points"]),
        -int(ev["duplicate_overcount_points"]),
    )


def pick_declarations(hand: list[int], n: int) -> list[tuple[int, dict[str, Any]]]:
    scored: list[tuple[tuple[int, int, int, int], int, dict[str, Any]]] = []
    hand_tuple = tuple(sorted(hand))
    for decl_id in PIP_DECLS:
        ev = hand_eval(hand_tuple, decl_id)
        scored.append((static_key(ev), decl_id, ev))
    scored.sort(reverse=True)
    return [(decl_id, ev) for _key, decl_id, ev in scored[:n]]


def bid_result_map(points: list[int], decl_id: int) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    result = evaluate_bids(points, decl_id)
    for bid in result.bid_results:
        if bid.bid in BID_THRESHOLDS:
            out[bid.bid] = {
                "p_make": float(bid.p_make),
                "mark_swing": float(bid.mark_swing),
                "ci_low": float(bid.ci_low),
                "ci_high": float(bid.ci_high),
            }
    return out


def contract_row(
    *,
    seed: int,
    seat: int,
    hand: list[int],
    decl_id: int,
    ev: dict[str, Any],
    points: list[int],
    by_bid: dict[int, dict[str, float]],
    sim_seed: int,
) -> dict[str, Any]:
    mark_swings = {bid: by_bid[bid]["mark_swing"] for bid in BID_THRESHOLDS}
    profitable = [bid for bid in BID_THRESHOLDS if mark_swings[bid] >= 0]
    wilson_safe = [bid for bid in BID_THRESHOLDS if by_bid[bid]["ci_low"] >= 0.5]
    best_bid = max(BID_THRESHOLDS, key=lambda bid: (mark_swings[bid], -bid))
    max_profitable_bid = max(profitable) if profitable else 0
    max_wilson_safe_bid = max(wilson_safe) if wilson_safe else 0

    row = {
        "seed": seed,
        "seat": seat,
        "team": seat % 2,
        "hand": hand_label(hand),
        "decl_id": decl_id,
        "decl_name": DECL_ID_TO_NAME.get(decl_id, f"decl_{decl_id}"),
        "n_samples": len(points),
        "sim_seed": sim_seed,
        "mean_points": round(mean(points), 6),
        "min_points": min(points),
        "max_points": max(points),
        "best_bid_by_mark_swing": best_bid,
        "best_mark_swing": round(mark_swings[best_bid], 6),
        "max_profitable_bid": max_profitable_bid,
        "max_profitable_bid_bucket": natural_bid_bucket(max_profitable_bid),
        "max_wilson_safe_bid": max_wilson_safe_bid,
        "max_wilson_safe_bid_bucket": natural_bid_bucket(max_wilson_safe_bid),
        "natural_bucket_candidate": int(max_profitable_bid in NATURAL_BUCKET_BIDS),
        "static_unique_exposed_points": int(ev["unique_exposed_points"]),
        "static_naive_exposed_points": int(ev["naive_exposed_points"]),
        "static_duplicate_overcount_points": int(ev["duplicate_overcount_points"]),
        "static_bid_ceiling_proxy": int(ev["bid_ceiling_proxy"]),
        "static_trump_count": int(ev["trump_count"]),
        "static_held_count_points": int(ev["held_count_points"]),
        "static_off_count": int(ev["off_count"]),
        "static_risk_bucket": risk_bucket(int(ev["unique_exposed_points"])),
        "static_strong_trump_bad_risk_trap": int(ev["strong_trump_bad_risk_trap"]),
        "static_minimum_shape_low_risk": int(ev["minimum_shape_low_risk"]),
        "static_four_five_off": int(ev["four_five_off"]),
    }
    for bid in BID_THRESHOLDS:
        row[f"p_make_{bid}"] = round(by_bid[bid]["p_make"], 6)
        row[f"mark_swing_{bid}"] = round(by_bid[bid]["mark_swing"], 6)
        row[f"ci_low_{bid}"] = round(by_bid[bid]["ci_low"], 6)
        row[f"ci_high_{bid}"] = round(by_bid[bid]["ci_high"], 6)
    return row


def owner_role(owner_seat: int | None) -> str:
    if owner_seat is None:
        return "none"
    if owner_seat == 2:
        return "partner"
    if owner_seat == 1:
        return "left_opponent"
    if owner_seat == 3:
        return "right_opponent"
    return "self"


def contract_value_for_team0(owner_role_name: str, owner_contract: dict[str, Any] | None, high_bid: int) -> float:
    if owner_contract is None:
        return 0.0
    value = float(owner_contract[f"mark_swing_{high_bid}"])
    if owner_role_name in {"left_opponent", "right_opponent"}:
        return -value
    return value


def make_auction_contexts(best_contract_by_seat: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    contexts = [
        {
            "auction_context_id": "opening",
            "current_high_bid": 29,
            "minimum_winning_bid": 30,
            "high_bid_owner_seat": "",
            "high_bid_owner_role": "none",
            "high_bid_decl_id": "",
            "high_bid_decl_name": "",
            "pass_value_mark_swing_team0": 0.0,
        }
    ]
    for owner in (1, 2, 3):
        role = owner_role(owner)
        contract = best_contract_by_seat[owner]
        for high_bid in HIGH_BID_CONTEXTS:
            contexts.append(
                {
                    "auction_context_id": f"{role}_high_{high_bid}",
                    "current_high_bid": high_bid,
                    "minimum_winning_bid": high_bid + 1,
                    "high_bid_owner_seat": owner,
                    "high_bid_owner_role": role,
                    "high_bid_decl_id": contract["decl_id"],
                    "high_bid_decl_name": contract["decl_name"],
                    "pass_value_mark_swing_team0": round(contract_value_for_team0(role, contract, high_bid), 6),
                }
            )
    return contexts


def bid_action_rows_for_context(
    seed: int,
    context: dict[str, Any],
    p0_contracts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pass_value = float(context["pass_value_mark_swing_team0"])
    min_bid = int(context["minimum_winning_bid"])
    if min_bid > 42:
        return rows, {
            "seed": seed,
            **context,
            "best_action": "pass",
            "best_bid": "",
            "best_decl_id": "",
            "best_delta_vs_pass": 0.0,
            "best_bid_value_team0": pass_value,
            "pass_value_mark_swing_team0": pass_value,
            "candidate_bid_rows": 0,
        }

    for contract in p0_contracts:
        min_result_value = float(contract[f"mark_swing_{min_bid}"])
        for actual_bid in range(min_bid, 43):
            value = float(contract[f"mark_swing_{actual_bid}"])
            rows.append(
                {
                    "seed": seed,
                    **context,
                    "candidate_action": "bid",
                    "candidate_bidder_seat": 0,
                    "candidate_decl_id": contract["decl_id"],
                    "candidate_decl_name": contract["decl_name"],
                    "actual_bid": actual_bid,
                    "unnecessary_bid_margin": actual_bid - min_bid,
                    "candidate_value_mark_swing_team0": round(value, 6),
                    "candidate_delta_vs_pass": round(value - pass_value, 6),
                    "candidate_delta_vs_min_bid": round(value - min_result_value, 6),
                    "p_make": contract[f"p_make_{actual_bid}"],
                    "ci_low": contract[f"ci_low_{actual_bid}"],
                    "ci_high": contract[f"ci_high_{actual_bid}"],
                    "static_unique_exposed_points": contract["static_unique_exposed_points"],
                    "static_trump_count": contract["static_trump_count"],
                    "static_risk_bucket": contract["static_risk_bucket"],
                    "static_bid_ceiling_proxy": contract["static_bid_ceiling_proxy"],
                    "static_strong_trump_bad_risk_trap": contract["static_strong_trump_bad_risk_trap"],
                }
            )

    best_row = max(rows, key=lambda row: (float(row["candidate_value_mark_swing_team0"]), -int(row["actual_bid"])))
    best_value = float(best_row["candidate_value_mark_swing_team0"])
    if pass_value >= best_value:
        decision = {
            "seed": seed,
            **context,
            "best_action": "pass",
            "best_bid": "",
            "best_decl_id": "",
            "best_decl_name": "",
            "best_delta_vs_pass": 0.0,
            "best_bid_value_team0": pass_value,
            "pass_value_mark_swing_team0": pass_value,
            "candidate_bid_rows": len(rows),
        }
    else:
        decision = {
            "seed": seed,
            **context,
            "best_action": "bid",
            "best_bid": best_row["actual_bid"],
            "best_decl_id": best_row["candidate_decl_id"],
            "best_decl_name": best_row["candidate_decl_name"],
            "best_delta_vs_pass": round(best_value - pass_value, 6),
            "best_bid_value_team0": best_value,
            "pass_value_mark_swing_team0": pass_value,
            "candidate_bid_rows": len(rows),
        }
    return rows, decision


def summarize_group(rows: list[dict[str, Any]], keys: list[str], value_fields: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    out: list[dict[str, Any]] = []
    for key_values, group in sorted(grouped.items(), key=lambda item: item[0]):
        record = {key: value for key, value in zip(keys, key_values)}
        record["n"] = len(group)
        for field in value_fields:
            values = [float(row[field]) for row in group if row.get(field) not in {"", None}]
            if values:
                record[f"mean_{field}"] = round(mean(values), 6)
                record[f"min_{field}"] = round(min(values), 6)
                record[f"max_{field}"] = round(max(values), 6)
        out.append(record)
    return out


def summarize_counts(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    counts: Counter[tuple[Any, ...]] = Counter(tuple(row[key] for key in keys) for row in rows)
    out: list[dict[str, Any]] = []
    total = sum(counts.values())
    for key_values, n in sorted(counts.items()):
        record = {key: value for key, value in zip(keys, key_values)}
        record["n"] = n
        record["pct"] = round(n / total, 6) if total else 0.0
        out.append(record)
    return out


def maybe_log_wandb(args: argparse.Namespace, summary: dict[str, Any], margin_summary: list[dict[str, Any]], partner_summary: list[dict[str, Any]]) -> dict[str, Any]:
    if args.wandb_mode == "disabled":
        return {"mode": "disabled"}
    try:
        import wandb
    except Exception as exc:  # pragma: no cover - optional integration
        return {"mode": args.wandb_mode, "error": f"wandb import failed: {exc}"}

    run = wandb.init(
        project=args.wandb_project,
        group=args.wandb_group,
        name=args.wandb_name,
        mode=args.wandb_mode,
        config={
            "bead": BEAD_ID,
            "seeds": args.seeds,
            "samples": args.samples,
            "decls_per_hand": args.decls_per_hand,
            "device": args.device,
        },
    )
    wandb.log({"summary/contract_rows": summary["counts"]["contract_rows"], "summary/bid_action_rows": summary["counts"]["bid_action_rows"]})
    for row in margin_summary:
        payload = {f"margin/{k}": v for k, v in row.items() if isinstance(v, (int, float))}
        payload["unnecessary_bid_margin"] = int(row["unnecessary_bid_margin"])
        wandb.log(payload)
    for idx, row in enumerate(partner_summary):
        payload = {f"partner/{k}": v for k, v in row.items() if isinstance(v, (int, float))}
        payload["partner_row"] = idx
        wandb.log(payload)
    run_url = run.url
    run_id = run.id
    wandb.finish()
    return {"mode": args.wandb_mode, "run_id": run_id, "run_url": run_url}


def run(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.perf_counter()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = PolicyModel(args.checkpoint, device=args.device, compile_model=False)

    contract_rows: list[dict[str, Any]] = []
    point_sample_rows: list[dict[str, Any]] = []
    auction_context_rows: list[dict[str, Any]] = []
    bid_action_rows: list[dict[str, Any]] = []
    context_decision_rows: list[dict[str, Any]] = []

    for seed in range(args.start_seed, args.start_seed + args.seeds):
        deal = [list(hand) for hand in deal_from_seed(seed)]
        seed_contracts_by_seat: dict[int, list[dict[str, Any]]] = {}

        for seat, hand in enumerate(deal):
            rows_for_seat: list[dict[str, Any]] = []
            for decl_id, ev in pick_declarations(hand, args.decls_per_hand):
                sim_seed = args.sim_seed + seed * 10000 + seat * 100 + decl_id
                points_tensor = simulate_games(
                    model=model,
                    bidder_hand=hand,
                    decl_id=decl_id,
                    n_games=args.samples,
                    seed=sim_seed,
                    greedy=True,
                )
                points = [int(x) for x in points_tensor.cpu().tolist()]
                by_bid = bid_result_map(points, decl_id)
                row = contract_row(
                    seed=seed,
                    seat=seat,
                    hand=hand,
                    decl_id=decl_id,
                    ev=ev,
                    points=points,
                    by_bid=by_bid,
                    sim_seed=sim_seed,
                )
                rows_for_seat.append(row)
                contract_rows.append(row)
                point_sample_rows.append(
                    {
                        "seed": seed,
                        "seat": seat,
                        "hand": row["hand"],
                        "decl_id": decl_id,
                        "decl_name": row["decl_name"],
                        "sim_seed": sim_seed,
                        "points": "|".join(str(p) for p in points),
                    }
                )
            seed_contracts_by_seat[seat] = rows_for_seat

        best_contract_by_seat = {
            seat: max(rows, key=lambda row: (float(row["best_mark_swing"]), -int(row["best_bid_by_mark_swing"])))
            for seat, rows in seed_contracts_by_seat.items()
        }
        contexts = make_auction_contexts(best_contract_by_seat)
        auction_context_rows.extend({"seed": seed, **context} for context in contexts)

        p0_contracts = seed_contracts_by_seat[0]
        for context in contexts:
            action_rows, decision = bid_action_rows_for_context(seed, context, p0_contracts)
            bid_action_rows.extend(action_rows)
            context_decision_rows.append(decision)

    margin_rows = [row for row in bid_action_rows if int(row["unnecessary_bid_margin"]) > 0]
    minimum_rows = [row for row in bid_action_rows if int(row["unnecessary_bid_margin"]) == 0]
    partner_decisions = [row for row in context_decision_rows if row["high_bid_owner_role"] == "partner"]

    margin_summary = summarize_group(
        margin_rows,
        ["unnecessary_bid_margin"],
        ["candidate_delta_vs_min_bid", "candidate_delta_vs_pass", "candidate_value_mark_swing_team0"],
    )
    margin_by_context_summary = summarize_group(
        margin_rows,
        ["high_bid_owner_role", "current_high_bid", "unnecessary_bid_margin"],
        ["candidate_delta_vs_min_bid", "candidate_delta_vs_pass"],
    )
    natural_bucket_summary = summarize_counts(contract_rows, ["seat", "static_risk_bucket", "max_profitable_bid_bucket"])
    risk_budget_summary = summarize_group(
        contract_rows,
        ["static_risk_bucket"],
        ["p_make_30", "mark_swing_30", "max_profitable_bid", "static_bid_ceiling_proxy"],
    )
    partner_signal_summary = summarize_group(
        partner_decisions,
        ["current_high_bid", "best_action"],
        ["best_delta_vs_pass", "best_bid_value_team0", "pass_value_mark_swing_team0"],
    )
    opponent_pressure_summary = summarize_group(
        [row for row in context_decision_rows if row["high_bid_owner_role"] in {"left_opponent", "right_opponent"}],
        ["current_high_bid", "best_action"],
        ["best_delta_vs_pass", "best_bid_value_team0", "pass_value_mark_swing_team0"],
    )

    write_csv(args.out_dir / "contract_rows.csv", contract_rows)
    write_csv(args.out_dir / "point_samples.csv", point_sample_rows)
    write_csv(args.out_dir / "auction_context_rows.csv", auction_context_rows)
    write_csv(args.out_dir / "bid_action_rows.csv", bid_action_rows)
    write_csv(args.out_dir / "context_decision_rows.csv", context_decision_rows)
    write_csv(args.out_dir / "bid_margin_summary.csv", margin_summary)
    write_csv(args.out_dir / "bid_margin_by_context_summary.csv", margin_by_context_summary)
    write_csv(args.out_dir / "natural_bucket_summary.csv", natural_bucket_summary)
    write_csv(args.out_dir / "risk_budget_summary.csv", risk_budget_summary)
    write_csv(args.out_dir / "partner_signal_summary.csv", partner_signal_summary)
    write_csv(args.out_dir / "opponent_pressure_summary.csv", opponent_pressure_summary)

    negative_margin = sum(1 for row in margin_rows if float(row["candidate_delta_vs_min_bid"]) < 0)
    zero_margin = sum(1 for row in margin_rows if float(row["candidate_delta_vs_min_bid"]) == 0)
    positive_margin = sum(1 for row in margin_rows if float(row["candidate_delta_vs_min_bid"]) > 0)
    min_positive_vs_pass = sum(1 for row in minimum_rows if float(row["candidate_delta_vs_pass"]) > 0)
    min_negative_vs_pass = sum(1 for row in minimum_rows if float(row["candidate_delta_vs_pass"]) < 0)
    partner_overcalls = sum(1 for row in partner_decisions if row["best_action"] == "bid")

    summary: dict[str, Any] = {
        "schema_version": "w42.auction_bid_discipline_claim_tests.v1",
        "bead": BEAD_ID,
        "git_sha": git_sha(),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - t0, 3),
        "config": {
            "start_seed": args.start_seed,
            "seeds": args.seeds,
            "samples": args.samples,
            "decls_per_hand": args.decls_per_hand,
            "device": args.device,
            "sim_seed": args.sim_seed,
            "checkpoint": str(args.checkpoint or "default forge.bidding checkpoint"),
            "high_bid_contexts": list(HIGH_BID_CONTEXTS),
            "natural_bucket_bids": sorted(NATURAL_BUCKET_BIDS),
        },
        "counts": {
            "deals": args.seeds,
            "contract_rows": len(contract_rows),
            "point_sample_rows": len(point_sample_rows),
            "auction_context_rows": len(auction_context_rows),
            "context_decision_rows": len(context_decision_rows),
            "bid_action_rows": len(bid_action_rows),
            "minimum_bid_action_rows": len(minimum_rows),
            "positive_margin_rows": len(margin_rows),
        },
        "headline": {
            "positive_margin_delta_vs_min_positive_rows": positive_margin,
            "positive_margin_delta_vs_min_zero_rows": zero_margin,
            "positive_margin_delta_vs_min_negative_rows": negative_margin,
            "minimum_bid_positive_vs_pass_rows": min_positive_vs_pass,
            "minimum_bid_negative_vs_pass_rows": min_negative_vs_pass,
            "partner_contexts": len(partner_decisions),
            "partner_overcall_recommended_contexts": partner_overcalls,
            "partner_overcall_rate": round(partner_overcalls / len(partner_decisions), 6) if partner_decisions else 0.0,
            "contract_natural_bucket_rate": round(sum(int(row["natural_bucket_candidate"]) for row in contract_rows) / len(contract_rows), 6) if contract_rows else 0.0,
        },
        "scientific_status": {
            "what_this_tests": "Generated auction-pressure rows compare pass, minimum overcall, and higher bid amounts using bidder-hand Monte Carlo make labels for P0, partner, and opponents.",
            "supported_slices": "Bid-only-enough is now tested under opening, partner-high, and opponent-high auction contexts. Static risk buckets are joined to empirical make labels. Natural bid buckets are measured as max-profitable threshold buckets.",
            "boundary": "This is not a fully conditioned four-hand auction engine. Partner/opponent values use their hand as a bidder-hand Monte Carlo label, and pass values remain offline comparison labels.",
        },
        "artifacts": {
            "contract_rows": str(args.out_dir / "contract_rows.csv"),
            "point_samples": str(args.out_dir / "point_samples.csv"),
            "auction_context_rows": str(args.out_dir / "auction_context_rows.csv"),
            "bid_action_rows": str(args.out_dir / "bid_action_rows.csv"),
            "context_decision_rows": str(args.out_dir / "context_decision_rows.csv"),
            "bid_margin_summary": str(args.out_dir / "bid_margin_summary.csv"),
            "bid_margin_by_context_summary": str(args.out_dir / "bid_margin_by_context_summary.csv"),
            "natural_bucket_summary": str(args.out_dir / "natural_bucket_summary.csv"),
            "risk_budget_summary": str(args.out_dir / "risk_budget_summary.csv"),
            "partner_signal_summary": str(args.out_dir / "partner_signal_summary.csv"),
            "opponent_pressure_summary": str(args.out_dir / "opponent_pressure_summary.csv"),
        },
    }
    wandb_info = maybe_log_wandb(args, summary, margin_summary, partner_signal_summary)
    summary["wandb"] = wandb_info
    write_json(args.out_dir / "summary.json", summary)

    if args.smoke:
        smoke_assertions(summary, bid_action_rows, contract_rows)

    return summary


def smoke_assertions(summary: dict[str, Any], bid_action_rows: list[dict[str, Any]], contract_rows: list[dict[str, Any]]) -> None:
    if summary["counts"]["contract_rows"] <= 0:
        raise AssertionError("No contract rows generated")
    if summary["counts"]["bid_action_rows"] <= 0:
        raise AssertionError("No bid action rows generated")
    if not any(int(row["unnecessary_bid_margin"]) > 0 for row in bid_action_rows):
        raise AssertionError("No positive bid-margin rows generated")
    if summary["headline"]["positive_margin_delta_vs_min_positive_rows"] != 0:
        raise AssertionError("A positive-margin bid improved over the minimum bid")
    if not any(int(row["max_profitable_bid"]) > 0 for row in contract_rows):
        raise AssertionError("No profitable contracts found")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--decls-per-hand", type=int, default=3)
    parser.add_argument("--sim-seed", type=int, default=430100)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--wandb-mode", choices=["disabled", "offline", "online"], default="disabled")
    parser.add_argument("--wandb-project", default="w42")
    parser.add_argument("--wandb-group", default="w42-auction-bid-discipline")
    parser.add_argument("--wandb-name", default="t42-qtwb.1-auction-bid-discipline-v0")
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary["counts"], indent=2, sort_keys=True))
    print(json.dumps(summary["headline"], indent=2, sort_keys=True))
    print(json.dumps(summary.get("wandb", {}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
