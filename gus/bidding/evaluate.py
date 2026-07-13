"""Gus bidding evaluator — P(make) per (trump × bid) for a given hand.

Iterates over pip trumps 0..6 and doubles (decl 7). Skips doubles-as-suit (8)
and follow-me variants (9) — not worth evaluating.

Uses forge/bidding/estimator.py for Wilson CI aggregation.

Usage:
    .venv/bin/python -m gus.bidding.evaluate \\
        --hand "6-3,5-4,4-3,5-1,3-1,2-1,1-1" \\
        --samples 100
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from forge.bidding.estimator import evaluate_bids, find_best_bid, TrumpResult
from gus.bidding.simulate import simulate_all_gus_batch, _pick_device
from gus.model.load import load_student

ADAPTER_DEFAULT = Path(__file__).resolve().parents[2] / "gus/adapters/v2_voids_3000g_big.pt"

# Pip declarations 0..6 + doubles-trump (7). Exclude doubles-as-suit (8) per user ask.
TRUMP_IDS = [0, 1, 2, 3, 4, 5, 6, 7]

DECL_NAMES = {
    0: "blanks", 1: "ones", 2: "twos", 3: "threes",
    4: "fours", 5: "fives", 6: "sixes", 7: "doubles",
}

DOMINO_NAMES = [f"{a}-{b}" for a in range(7) for b in range(a + 1)]


def _name_to_id(name: str) -> int:
    a, b = name.split("-")
    hi, lo = max(int(a), int(b)), min(int(a), int(b))
    return DOMINO_NAMES.index(f"{hi}-{lo}")


def parse_hand(spec: str) -> list[int]:
    return [_name_to_id(t.strip()) for t in spec.split(",") if t.strip()]


def load_gus(adapter_path: Path, device: str):
    return load_student(adapter_path, device)


def evaluate_hand(
    model, is_voids: bool, hand: list[int], n_samples: int, device: str, seed: int = 42
) -> list[TrumpResult]:
    t0 = time.time()
    pts_per_trump = simulate_all_gus_batch(
        model, is_voids, hand, TRUMP_IDS, n_samples, device, seed
    )
    dt = time.time() - t0
    print(f"  batched {len(TRUMP_IDS)} trumps × {n_samples} games "
          f"({len(TRUMP_IDS) * n_samples} total) in {dt:.1f}s", flush=True)

    results: list[TrumpResult] = []
    for decl_id in TRUMP_IDS:
        pts = pts_per_trump[decl_id]
        mean_pts = float(pts.float().mean().item())
        p_win = float((pts >= 30).float().mean().item())
        print(f"  decl {DECL_NAMES[decl_id]:<8} "
              f"mean={mean_pts:5.1f} pts  P(make30)={p_win:.2f}", flush=True)
        results.append(evaluate_bids(pts.cpu().tolist(), decl_id))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", required=True)
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--adapter", type=str, default=str(ADAPTER_DEFAULT))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    hand = parse_hand(args.hand)
    if len(hand) != 7:
        raise SystemExit(f"need 7 dominoes, got {len(hand)}")

    device = _pick_device()
    print(f"device: {device}")
    model, is_voids = load_gus(Path(args.adapter), device)
    print(f"adapter: {args.adapter}")
    print(f"hand: {[DOMINO_NAMES[d] for d in hand]}")
    print(f"samples per trump: {args.samples}\n")

    results = evaluate_hand(model, is_voids, hand, args.samples, device, args.seed)

    trump, bid, swing = find_best_bid(results)
    print(f"\nbest bid: **{bid} on {trump}**  (expected mark swing {swing:+.3f})")

    if args.json:
        payload = {
            "hand": args.hand,
            "samples_per_trump": args.samples,
            "best_trump": trump,
            "best_bid": bid,
            "best_swing": swing,
            "trumps": [
                {
                    "decl_id": r.decl_id,
                    "name": r.trump_name,
                    "points": r.points,
                    "p_make_by_bid": {
                        str(b.bid): b.p_make for b in r.bid_results
                    },
                    "mark_swing_by_bid": {
                        str(b.bid): b.mark_swing for b in r.bid_results
                    },
                }
                for r in results
            ],
        }
        print("\n" + json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
