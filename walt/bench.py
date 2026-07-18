"""walt/bench.py — endgame-solve performance bench (builder B4).

Generate R random endgame roots (new_game(seed), then advance with JudPlay
until the mover first holds <= H tiles; skip terminal hands), solve each with
beliefs 'u' and 'sigma', and report p50/p95/max solve wall ms, the world-count
distribution, field-query counts, and total wall. Hard budget: any paired
perf bench <= 10 min wall (DESIGN.md); prints a projection warning and stops
early if the running wall would blow the cap.

    python -u walt/bench.py --horizon 4 --n 200
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from walt.contracts import EndgameRoot

BUDGET_S = 600.0  # 10 min hard cap


def _pct(xs: list[float], q: float) -> float:
    return float(np.percentile(xs, q)) if xs else float("nan")


def build_root_at_horizon(seed: int, horizon: int, jud_play):
    """Play a fresh random game with JudPlay until the mover first holds
    <= horizon tiles; return the EndgameRoot, or None if the hand ended first.
    """
    from forge.zeb.game import apply_action, current_player, is_terminal
    from forge.zeb.game import new_game

    state = new_game(seed)  # PLAYING state with a random winning auction
    while not is_terminal(state):
        mover = current_player(state)
        remaining = [d for d in state.hands[mover] if d not in state.played]
        if len(remaining) <= horizon:
            return EndgameRoot(
                decl_id=state.decl_id,
                bidder=state.bidder,
                bid_value=int(state.bid_state.high_bid),
                bids=tuple(state.bid_state.bids),
                dealer=state.dealer,
                me=mover,
                my_hand=tuple(sorted(remaining)),
                play_history=tuple(state.play_history),
                trick_leader=state.trick_leader,
                current_trick=tuple(state.current_trick),
                team_points=tuple(state.team_points),
            )
        action = jud_play.choose([state], [state.bid_state.high_bid])[0]
        state = apply_action(state, action)
    return None  # terminal before reaching the horizon


def solve_one(root: EndgameRoot, beliefs: str, oracle):
    """Enumerate (+optional sigma filter) and solve; returns (SolveResult, ms)."""
    from walt.worlds import enumerate_worlds
    from walt.solver import solve

    worlds = enumerate_worlds(root)
    if beliefs == "sigma" and len(worlds) > 0:
        from walt.field import sigma_consistent
        from walt.grade import _make_moves_filter

        keep = sigma_consistent(root, worlds, oracle, _make_moves_filter(root))
        filtered = worlds[keep]
        if len(filtered) > 0:
            worlds = filtered
    n = max(len(worlds), 1)
    weights = np.full(len(worlds), 1.0 / n, dtype=np.float64)
    t0 = time.perf_counter()
    res = solve(root, worlds, weights, oracle, payoff="points")
    return res, (time.perf_counter() - t0) * 1e3


def main() -> int:
    parser = argparse.ArgumentParser(description="walt endgame-solve bench")
    parser.add_argument("--horizon", type=int, choices=(3, 4), default=4)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--field-net", type=str, default="champion/jud_net.pt")
    parser.add_argument("--seed0", type=int, default=100_000)
    args = parser.parse_args()

    import torch

    torch.set_num_threads(1)
    from arena.jud_play import JudPlay
    from champion.jud_net import load_jud_net
    from walt.field import FieldOracle

    jud = JudPlay(load_jud_net(args.field_net, device="cpu"))
    oracle = FieldOracle(net_path=args.field_net, device="cpu")

    print(f"bench: horizon={args.horizon} n={args.n} net={args.field_net}", flush=True)

    per = {"u": [], "sigma": []}
    worlds_pre = []   # world count before sigma filter (== u world count)
    worlds_post = []  # world count after sigma filter
    fq = {"u": [], "sigma": []}
    n_roots = 0
    skipped = 0
    t_start = time.time()
    last_log = t_start
    seed = args.seed0

    while n_roots < args.n:
        root = build_root_at_horizon(seed, args.horizon, jud)
        seed += 1
        if root is None:
            skipped += 1
            continue
        for beliefs in ("u", "sigma"):
            res, ms = solve_one(root, beliefs, oracle)
            per[beliefs].append(ms)
            fq[beliefs].append(res.n_field_queries)
            if beliefs == "u":
                worlds_pre.append(res.n_worlds)
            else:
                worlds_post.append(res.n_worlds)
        oracle.evict_if_huge()
        n_roots += 1

        now = time.time()
        wall = now - t_start
        if now - last_log >= 30.0:
            last_log = now
            proj = wall / n_roots * args.n
            print(
                f"  {n_roots}/{args.n} roots  wall {wall:.0f}s  "
                f"proj {proj:.0f}s  p50(u) {_pct(per['u'], 50):.1f}ms",
                flush=True,
            )
        # Early projection at ~10 roots.
        if n_roots == 10:
            proj = wall / n_roots * args.n
            flag = "  !! OVER 10min BUDGET" if proj > BUDGET_S else ""
            print(f"  [projection @10] total ~{proj:.0f}s for n={args.n}{flag}",
                  flush=True)
        if wall > BUDGET_S:
            print(f"  !! wall {wall:.0f}s exceeded {BUDGET_S:.0f}s cap; "
                  f"stopping at {n_roots} roots", flush=True)
            break

    total_wall = time.time() - t_start
    print("\n=== bench results ===", flush=True)
    print(f"roots solved: {n_roots}  (skipped terminal: {skipped})", flush=True)
    for beliefs in ("u", "sigma"):
        xs = per[beliefs]
        print(
            f"{beliefs:>5} solve ms:  p50 {_pct(xs, 50):8.2f}  "
            f"p95 {_pct(xs, 95):8.2f}  max {max(xs) if xs else float('nan'):8.2f}  "
            f"field_q p50 {_pct(fq[beliefs], 50):.0f}",
            flush=True,
        )
    print(
        f"worlds (u):     min {min(worlds_pre) if worlds_pre else 0}  "
        f"p50 {_pct(worlds_pre, 50):.0f}  p95 {_pct(worlds_pre, 95):.0f}  "
        f"max {max(worlds_pre) if worlds_pre else 0}",
        flush=True,
    )
    print(
        f"worlds (sigma): min {min(worlds_post) if worlds_post else 0}  "
        f"p50 {_pct(worlds_post, 50):.0f}  p95 {_pct(worlds_post, 95):.0f}  "
        f"max {max(worlds_post) if worlds_post else 0}",
        flush=True,
    )
    proj_full = total_wall / n_roots * args.n if n_roots else float("nan")
    print(f"total wall: {total_wall:.1f}s  "
          f"(projected full-n: {proj_full:.0f}s, budget {BUDGET_S:.0f}s)",
          flush=True)
    if proj_full > BUDGET_S:
        print("  !! projected over the 10-min bench cap — shrink --n", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
