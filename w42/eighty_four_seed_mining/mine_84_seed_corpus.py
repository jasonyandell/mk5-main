#!/usr/bin/env python3
"""Mine natural seed deals for W42 phase-3 84 endgame tests.

The dynamic 84 lab proved that hand-built fixtures can reach useful
preservation labels. This miner looks for naturally generated deals that match
the book's 84 surfaces: laydowns, protected one-offs, straight offs, two-offs in
one suit, defender live doubles, same-suit pairs, pair protectors, and dead
asset release controls.
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
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forge.oracle.rng import deal_from_seed
from forge.oracle.tables import DOMINOES, DOMINO_IS_DOUBLE, domino_contains_pip


OUT_DIR = ROOT / "w42" / "eighty_four_seed_mining"
BEAD_ID = "t42-qtwb.2"
PIP_DECLS = tuple(range(7))


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def domino_name(domino_id: int) -> str:
    high, low = DOMINOES[domino_id]
    return f"{high}-{low}"


def hand_label(hand: list[int]) -> str:
    return ",".join(domino_name(d) for d in sorted(hand))


def double_id(pip: int) -> int:
    return DOMINOES.index((pip, pip))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def is_trump(domino_id: int, decl_id: int) -> bool:
    return domino_contains_pip(domino_id, decl_id)


def off_tiles(hand: list[int], decl_id: int) -> list[int]:
    return [
        d
        for d in hand
        if not is_trump(d, decl_id) and not DOMINO_IS_DOUBLE[d]
    ]


def matching_doubles(tile: int) -> set[int]:
    high, low = DOMINOES[tile]
    return {double_id(high), double_id(low)}


def shared_pips(tiles: list[int]) -> set[int]:
    if not tiles:
        return set()
    sets = [set(DOMINOES[tile]) for tile in tiles]
    out = sets[0]
    for s in sets[1:]:
        out &= s
    return out


def same_suit_count(hand: list[int], pip: int, decl_id: int) -> int:
    return sum(1 for d in hand if not is_trump(d, decl_id) and domino_contains_pip(d, pip))


def defender_asset_counts(deal: list[list[int]], bidder_seat: int, decl_id: int, final_off: int | None) -> dict[str, Any]:
    partner_seat = (bidder_seat + 2) % 4
    opponent_seats = [seat for seat in range(4) if seat not in {bidder_seat, partner_seat}]
    final_pips = set(DOMINOES[final_off]) if final_off is not None else set()

    opponent_matching_double_seats: list[str] = []
    partner_matching_double = 0
    opponent_same_suit_pair_seats: list[str] = []
    opponent_pair_protector_seats: list[str] = []
    opponent_dead_double_assets = 0

    for pip in final_pips:
        did = double_id(pip)
        if did in deal[partner_seat]:
            partner_matching_double += 1
        for seat in opponent_seats:
            if did in deal[seat]:
                opponent_matching_double_seats.append(str(seat))
            suit_count = same_suit_count(deal[seat], pip, decl_id)
            if suit_count >= 2:
                opponent_same_suit_pair_seats.append(str(seat))
            if suit_count >= 3:
                opponent_pair_protector_seats.append(str(seat))

    for seat in opponent_seats:
        for domino_id in deal[seat]:
            if DOMINO_IS_DOUBLE[domino_id]:
                pip = DOMINOES[domino_id][0]
                if pip != decl_id and pip not in final_pips:
                    opponent_dead_double_assets += 1

    return {
        "opponent_matching_double_count": len(set(opponent_matching_double_seats)),
        "opponent_matching_double_seats": "|".join(sorted(set(opponent_matching_double_seats))),
        "partner_matching_double_count": partner_matching_double,
        "opponent_same_suit_pair_count": len(set(opponent_same_suit_pair_seats)),
        "opponent_same_suit_pair_seats": "|".join(sorted(set(opponent_same_suit_pair_seats))),
        "opponent_pair_protector_count": len(set(opponent_pair_protector_seats)),
        "opponent_pair_protector_seats": "|".join(sorted(set(opponent_pair_protector_seats))),
        "opponent_dead_double_assets": opponent_dead_double_assets,
    }


def classify_candidate(seed: int, deal: list[list[int]], seat: int, decl_id: int) -> list[dict[str, Any]]:
    hand = sorted(deal[seat])
    trump_tiles = [d for d in hand if is_trump(d, decl_id)]
    offs = off_tiles(hand, decl_id)
    nontrump_doubles = [d for d in hand if DOMINO_IS_DOUBLE[d] and not is_trump(d, decl_id)]
    double_count = sum(1 for d in hand if DOMINO_IS_DOUBLE[d])
    surfaces: set[str] = set()

    if len(trump_tiles) == 7:
        surfaces.add("laydown_all_trumps")
    if len(offs) == 1:
        final_off = offs[0]
        if matching_doubles(final_off) & set(hand):
            surfaces.add("protected_one_off")
        else:
            surfaces.add("straight_one_off")
        if len(trump_tiles) >= 3 and double_count >= 3:
            surfaces.add("three_trump_three_double_one_off")
        final_offs = [final_off]
    elif len(offs) == 2 and shared_pips(offs):
        surfaces.add("two_off_same_suit")
        final_offs = offs
    else:
        final_offs = offs[:1] if offs else [None]

    rows: list[dict[str, Any]] = []
    for final_off in final_offs:
        assets = defender_asset_counts(deal, seat, decl_id, final_off)
        row_surfaces = set(surfaces)
        if assets["opponent_matching_double_count"] > 0:
            row_surfaces.add("defender_live_double_weapon")
        if assets["opponent_same_suit_pair_count"] > 0:
            row_surfaces.add("defender_live_same_suit_pair")
        if assets["opponent_pair_protector_count"] > 0:
            row_surfaces.add("pair_protector_pressure")
        if assets["opponent_dead_double_assets"] > 0 and final_off is not None:
            row_surfaces.add("dead_asset_release_control")

        if not (row_surfaces & {
            "laydown_all_trumps",
            "protected_one_off",
            "straight_one_off",
            "two_off_same_suit",
            "three_trump_three_double_one_off",
        }):
            continue

        score = (
            4 * int("protected_one_off" in row_surfaces)
            + 4 * int("straight_one_off" in row_surfaces)
            + 3 * int("two_off_same_suit" in row_surfaces)
            + 3 * int("defender_live_same_suit_pair" in row_surfaces)
            + 2 * int("defender_live_double_weapon" in row_surfaces)
            + 2 * int("pair_protector_pressure" in row_surfaces)
            + int("dead_asset_release_control" in row_surfaces)
            + int("laydown_all_trumps" in row_surfaces)
        )
        rows.append(
            {
                "seed": seed,
                "bidder_seat": seat,
                "partner_seat": (seat + 2) % 4,
                "decl_id": decl_id,
                "decl_name": ["blanks", "ones", "twos", "threes", "fours", "fives", "sixes"][decl_id],
                "bidder_hand": hand_label(hand),
                "final_off": "" if final_off is None else domino_name(final_off),
                "trump_count": len(trump_tiles),
                "off_count": len(offs),
                "nontrump_double_count": len(nontrump_doubles),
                "total_double_count": double_count,
                "surfaces": "|".join(sorted(row_surfaces)),
                "surface_count": len(row_surfaces),
                "recommendation_score": score,
                **assets,
                "full_deal": json.dumps([hand_label(h) for h in deal]),
            }
        )
    return rows


def summarize_by_surface(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    example_seeds: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        for surface in row["surfaces"].split("|"):
            counts[surface] += 1
            if len(example_seeds[surface]) < 10:
                example_seeds[surface].append(str(row["seed"]))
    out = []
    for surface, n in sorted(counts.items()):
        out.append(
            {
                "surface": surface,
                "candidate_rows": n,
                "example_seeds": "|".join(example_seeds[surface]),
            }
        )
    return out


def summarize_by_family(rows: list[dict[str, Any]], scanned: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for surface in row["surfaces"].split("|"):
            grouped[(surface, row["decl_name"])].append(row)
    out = []
    for (surface, decl_name), group in sorted(grouped.items()):
        seeds = {int(row["seed"]) for row in group}
        out.append(
            {
                "surface": surface,
                "decl_name": decl_name,
                "candidate_rows": len(group),
                "unique_seeds": len(seeds),
                "seed_hit_rate": round(len(seeds) / scanned, 8),
                "mean_recommendation_score": round(sum(int(row["recommendation_score"]) for row in group) / len(group), 6),
            }
        )
    return out


def maybe_log_wandb(args: argparse.Namespace, summary: dict[str, Any], surface_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if args.wandb_mode == "disabled":
        return {"mode": "disabled"}
    try:
        import wandb
    except Exception as exc:  # pragma: no cover
        return {"mode": args.wandb_mode, "error": f"wandb import failed: {exc}"}

    run = wandb.init(
        project=args.wandb_project,
        group=args.wandb_group,
        name=args.wandb_name,
        mode=args.wandb_mode,
        config=summary["config"],
    )
    wandb.log({
        "summary/scanned_seeds": summary["counts"]["scanned_seeds"],
        "summary/candidate_rows": summary["counts"]["candidate_rows"],
        "summary/recommended_rows": summary["counts"]["recommended_rows"],
    })
    for row in surface_rows:
        wandb.log({
            "surface/candidate_rows": int(row["candidate_rows"]),
            "surface/name_index": len(row["surface"]),
        })
    run_url = run.url
    run_id = run.id
    wandb.finish()
    return {"mode": args.wandb_mode, "run_id": run_id, "run_url": run_url}


def run(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.perf_counter()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidate_rows: list[dict[str, Any]] = []

    for seed in range(args.start_seed, args.start_seed + args.seeds):
        deal = [list(hand) for hand in deal_from_seed(seed)]
        for seat in range(4):
            for decl_id in PIP_DECLS:
                candidate_rows.extend(classify_candidate(seed, deal, seat, decl_id))

    candidate_rows.sort(
        key=lambda row: (
            -int(row["recommendation_score"]),
            int(row["seed"]),
            int(row["bidder_seat"]),
            int(row["decl_id"]),
            row["final_off"],
        )
    )
    recommended_rows = candidate_rows[: args.recommendations]
    surface_summary = summarize_by_surface(candidate_rows)
    family_summary = summarize_by_family(candidate_rows, args.seeds)
    seed_summary_rows = []
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        by_seed[int(row["seed"])].append(row)
    for seed, rows in sorted(by_seed.items(), key=lambda item: (-max(int(r["recommendation_score"]) for r in item[1]), item[0]))[: args.recommendations]:
        surfaces = sorted({surface for row in rows for surface in row["surfaces"].split("|")})
        seed_summary_rows.append(
            {
                "seed": seed,
                "candidate_rows": len(rows),
                "best_score": max(int(row["recommendation_score"]) for row in rows),
                "surfaces": "|".join(surfaces),
                "best_rows": json.dumps(rows[:5]),
            }
        )

    write_csv(args.out_dir / "candidate_84_seed_rows.csv", candidate_rows)
    write_csv(args.out_dir / "recommended_84_seed_rows.csv", recommended_rows)
    write_csv(args.out_dir / "surface_summary.csv", surface_summary)
    write_csv(args.out_dir / "family_decl_summary.csv", family_summary)
    write_csv(args.out_dir / "seed_recommendations.csv", seed_summary_rows)
    examples = {
        row["surface"]: [
            candidate
            for candidate in candidate_rows
            if row["surface"] in candidate["surfaces"].split("|")
        ][:5]
        for row in surface_summary
    }
    write_json(args.out_dir / "examples.json", examples)

    summary: dict[str, Any] = {
        "schema_version": "w42.eighty_four_seed_mining.v1",
        "bead": BEAD_ID,
        "git_sha": git_sha(),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - t0, 3),
        "config": {
            "start_seed": args.start_seed,
            "seeds": args.seeds,
            "recommendations": args.recommendations,
            "pip_declarations": list(PIP_DECLS),
        },
        "counts": {
            "scanned_seeds": args.seeds,
            "seat_declaration_checks": args.seeds * 4 * len(PIP_DECLS),
            "candidate_rows": len(candidate_rows),
            "recommended_rows": len(recommended_rows),
            "unique_candidate_seeds": len(by_seed),
        },
        "surface_counts": {row["surface"]: int(row["candidate_rows"]) for row in surface_summary},
        "scientific_status": {
            "what_this_tests": "A large generated seed range is mined for exact 84 candidate structures and full-deal defender asset patterns.",
            "what_this_enables": "Natural seed lists for follow-up branch-atlas or state-injection runs, replacing hand-built-only fixtures.",
            "what_this_does_not_test": "No legal late-state action value is measured here; dynamic preservation regret still needs branch-atlas generation or a true state injector.",
        },
        "artifacts": {
            "candidate_rows": str(args.out_dir / "candidate_84_seed_rows.csv"),
            "recommended_rows": str(args.out_dir / "recommended_84_seed_rows.csv"),
            "surface_summary": str(args.out_dir / "surface_summary.csv"),
            "family_decl_summary": str(args.out_dir / "family_decl_summary.csv"),
            "seed_recommendations": str(args.out_dir / "seed_recommendations.csv"),
            "examples": str(args.out_dir / "examples.json"),
        },
    }
    summary["wandb"] = maybe_log_wandb(args, summary, surface_summary)
    write_json(args.out_dir / "summary.json", summary)

    if args.smoke:
        smoke_assertions(summary)
    return summary


def smoke_assertions(summary: dict[str, Any]) -> None:
    if summary["counts"]["candidate_rows"] <= 0:
        raise AssertionError("No candidate 84 seed rows found")
    required = {"protected_one_off", "straight_one_off", "defender_live_double_weapon", "defender_live_same_suit_pair"}
    missing = required - set(summary["surface_counts"])
    if missing:
        raise AssertionError(f"Missing expected surfaces: {sorted(missing)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=200000)
    parser.add_argument("--recommendations", type=int, default=256)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--wandb-mode", choices=["disabled", "offline", "online"], default="disabled")
    parser.add_argument("--wandb-project", default="w42")
    parser.add_argument("--wandb-group", default="w42-84-seed-mining")
    parser.add_argument("--wandb-name", default="t42-qtwb.2-84-seed-mining-v0")
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary["counts"], indent=2, sort_keys=True))
    print(json.dumps(summary["surface_counts"], indent=2, sort_keys=True))
    print(json.dumps(summary.get("wandb", {}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
