"""Mid-hand paired eval: rob vs E[Q] picking up dropped-30 hands in progress.

rob's solver is only exact once few enough tricks remain; before that it
leans on a rollout model. This eval removes the rollout regime entirely:
a fast heuristic (JudPlay, deterministic argmax, no oracle) plays ALL four
seats through the first `start_trick - 1` tricks of a dropped-30 hand, then
the position is frozen and played out twice from the identical state —
once with rob's team holding the contract (E[Q] defending), once with
rob's team defending (E[Q] holding the contract). The prefix is
deterministic from the deal, so the two continuations start from the
byte-identical position; the paired offense-vs-offense comparison is pure
play skill from the exact window onward.

    python -u -m arena.midhand_eval --n-positions 16 --start-trick 4

Writes midhand_results.json under --out-dir and prints a per-position table.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from arena.auction import run_auction
from arena.bidders import Bid30Bidder
from arena.engine import hand_seed
from forge.oracle.rng import deal_from_seed
from forge.zeb.game import apply_action, current_player, is_terminal
from forge.zeb.types import BidState, GamePhase, ZebGameState


def build_position(base_seed: int, idx: int, prefix_play, prefix_plays: int):
    """Deal, force the 30 bid, and heuristic-play through the prefix."""
    seed = hand_seed(base_seed, idx, 0, 0)
    hands = tuple(tuple(h) for h in deal_from_seed(seed))
    dealer = idx % 4
    rng = random.Random(hash((seed, 0xA0C7)))
    result = run_auction(
        hands, dealer, (Bid30Bidder(),) * 4, rng,
        force_shaker=False, marks=(0, 0), marks_to_win=7,
    )
    assert result is not None and result.high_bid == 30
    state = ZebGameState(
        hands=hands,
        dealer=dealer,
        phase=GamePhase.PLAYING,
        bid_state=BidState(
            bids=result.bids, high_bidder=result.winner, high_bid=result.high_bid,
        ),
        decl_id=result.decl_id,
        bidder=result.winner,
        played=frozenset(),
        play_history=(),
        current_trick=(),
        trick_leader=result.winner,
        team_points=(0, 0),
    )
    for _ in range(prefix_plays):
        action = prefix_play.choose([state], [30])[0]
        state = apply_action(state, action)
    return seed, state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-positions", type=int, default=16)
    parser.add_argument("--start-trick", type=int, default=4,
                        help="rob/E[Q] take over at the start of this trick "
                             "(1-indexed); the heuristic plays everything before")
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--out-dir", type=str,
                        default=str(Path(__file__).parent / "results" / "midhand_t4"))
    args = parser.parse_args()
    prefix_plays = (args.start_trick - 1) * 4

    import torch
    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    torch.manual_seed(args.base_seed)

    from arena.jud_play import JudPlay
    from arena.lens_play import LensPlay
    from arena.rob_play import RobPlay
    from champion.jud_net import load_jud_net
    from forge.zeb.eval.loading import DEFAULT_ORACLE, load_oracle

    prefix_play = JudPlay(load_jud_net("champion/jud_net.pt", device="cpu"))
    print(f"Loading oracle: {DEFAULT_ORACLE} on {device}", flush=True)
    model = load_oracle(str(PROJECT_ROOT / DEFAULT_ORACLE), device)
    eq_play = LensPlay(model, utility="ev", n_samples=args.n_samples, device=device)
    rob_play = RobPlay()
    print(f"rob: {rob_play}", flush=True)

    print(f"Building {args.n_positions} positions "
          f"(heuristic prefix = {prefix_plays} plays, "
          f"takeover at trick {args.start_trick})", flush=True)
    positions = [
        build_position(args.base_seed, i, prefix_play, prefix_plays)
        for i in range(args.n_positions)
    ]

    # Two continuations per position from the identical frozen state:
    # rob_team = bidder's team (rob offense) and the complement (rob defense).
    live = []
    for idx, (seed, state) in enumerate(positions):
        for rob_team in (state.bidder % 2, 1 - state.bidder % 2):
            live.append({"idx": idx, "seed": seed, "rob_team": rob_team,
                         "state": state, "snap_points": state.team_points})

    t0 = time.time()
    tick = 0
    while True:
        pending = [g for g in live if not is_terminal(g["state"])]
        if not pending:
            break
        tick += 1
        rob_side = [g for g in pending
                    if current_player(g["state"]) % 2 == g["rob_team"]]
        eq_side = [g for g in pending
                   if current_player(g["state"]) % 2 != g["rob_team"]]
        for side, policy in ((rob_side, rob_play), (eq_side, eq_play)):
            if not side:
                continue
            actions = policy.choose(
                [g["state"] for g in side], [30] * len(side),
                [(0, 0)] * len(side), 7,
            )
            for g, a in zip(side, actions):
                g["state"] = apply_action(g["state"], a)
        print(f"  tick {tick}: {len(pending)} live  ({time.time() - t0:.0f}s)",
              flush=True)

    # Pair up: same position, offense held by rob vs by E[Q].
    rows = []
    for idx, (seed, snap) in enumerate(positions):
        bidder_team = snap.bidder % 2
        by_team = {g["rob_team"]: g for g in live if g["idx"] == idx}
        rob_off = by_team[bidder_team]      # rob's team holds the contract
        eq_off = by_team[1 - bidder_team]   # E[Q]'s team holds it
        rows.append({
            "idx": idx,
            "seed": seed,
            "dealer": snap.dealer,
            "bidder": snap.bidder,
            "decl_id": snap.decl_id,
            "snap_points": list(snap.team_points),
            "rob_off_points": list(rob_off["state"].team_points),
            "eq_off_points": list(eq_off["state"].team_points),
            "rob_off_made": rob_off["state"].team_points[bidder_team] >= 30,
            "eq_off_made": eq_off["state"].team_points[bidder_team] >= 30,
        })

    rob_made = sum(r["rob_off_made"] for r in rows)
    eq_made = sum(r["eq_off_made"] for r in rows)
    rob_only = sum(r["rob_off_made"] and not r["eq_off_made"] for r in rows)
    eq_only = sum(r["eq_off_made"] and not r["rob_off_made"] for r in rows)
    bt = lambda r: r["bidder"] % 2  # noqa: E731
    rob_off_pts = sum(r["rob_off_points"][bt(r)] for r in rows)
    eq_off_pts = sum(r["eq_off_points"][bt(r)] for r in rows)

    print("\nidx decl snap(off) | rob-off pts made | eq-off pts made | verdict")
    for r in rows:
        t = bt(r)
        tag = ("=" if r["rob_off_made"] == r["eq_off_made"]
               else ("ROB" if r["rob_off_made"] else "EQ"))
        print(f" {r['idx']:2d}   {r['decl_id']:2d}   {r['snap_points'][t]:2d}     "
              f"|   {r['rob_off_points'][t]:2d}   {int(r['rob_off_made'])}    "
              f"|   {r['eq_off_points'][t]:2d}   {int(r['eq_off_made'])}    | {tag}")
    n = len(rows)
    print(f"\npositions {n}  (takeover trick {args.start_trick}, "
          f"prefix by JudPlay all seats)")
    print(f"offense made: rob {rob_made}/{n}  eq {eq_made}/{n}   "
          f"discordant rob-only {rob_only} eq-only {eq_only}")
    print(f"offense points/hand: rob {rob_off_pts / n:.1f}  eq {eq_off_pts / n:.1f}"
          f"  (defense = 42 - opponent offense on same positions)")
    print(f"({time.time() - t0:.1f}s)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "midhand_results.json").write_text(json.dumps({
        "protocol": {
            "n_positions": n, "start_trick": args.start_trick,
            "prefix": "JudPlay all seats", "n_samples": args.n_samples,
            "base_seed": args.base_seed,
        },
        "totals": {
            "rob_off_made": rob_made, "eq_off_made": eq_made,
            "rob_only": rob_only, "eq_only": eq_only,
            "rob_off_pts_per_hand": rob_off_pts / n,
            "eq_off_pts_per_hand": eq_off_pts / n,
        },
        "rows": rows,
    }, indent=2) + "\n")
    print(f"Wrote {out_dir}/midhand_results.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
