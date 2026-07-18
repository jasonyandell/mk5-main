"""walt/solver.py — the exact endgame information-set solver.

`solve()` computes the exact best response of ``root.me`` in the information-set
game against the deterministic field ``oracle`` (jud argmax), per DESIGN.md and
contracts.py. The recursion partitions alive worlds by the observed σ-action
sequence between my decisions; strategy fusion is killed by construction because
every world that reaches a node shares that node's public history, hence shares
one information set and one chosen move.

Value unit: E[final declaring-team hand points, 0..42] (banked + remaining), the
same orientation as arena.jud_play.JudPlay. The walt seat maximizes ``sign*value``
with ``sign = +1 if me%2 == bidder%2 else -1`` FIXED at the root (equivalently:
declarer maximizes declaring points, defender minimizes them). ``value`` in the
result is always reported in declaring orientation (never sign-flipped).

Design choices:
- ``sign`` fixed at the root; always argmax ``sign*value`` at my nodes (never a
  per-node min/max switch). Reported value is un-flipped.
- Recursion runs to the true terminal (len(hist) == 28); banked points are
  accumulated with `walt.field.resolve_lut` (LUTs tabulated from the same
  forge rule authority the engine uses — tables.py parity gates), so leaf
  payoffs match zeb scoring exactly. The opponent-only tail after my hand
  empties is the ordinary opponent branch — no special case.
- The public context (`walt.field.NodeCtx`) is carried incrementally down the
  recursion: POV play blocks, banked feature points, per-seat played masks —
  O(1) per play instead of an O(history) replay per field query.
- Per opponent node, every alive world is submitted in ONE
  ``oracle.decisions_at`` call, which dedupes worlds sharing the acting seat's
  hand before touching the memo/net; the returned per-world moves partition
  the worlds into observation branches.
- No solver-level memo: node identity is the full public history, unique per
  recursion-tree node, so such a memo provably never hits (measured 0/19,566).

``n_field_queries`` counts total per-world oracle decision requests issued
across the solve (sum of query-batch sizes), including the walker
instrumentation's trick rollouts; ``n_nodes`` counts recursion nodes
(including the root).
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

from walt.contracts import SolveResult
from walt.field import NodeCtx, resolve_lut
from walt.tables import (
    get_luts,
    hand_to_mask,
    legal_moves_mask,
)


def _bits(mask: int) -> Iterator[int]:
    """Yield the domino ids set in a bitmask, ascending."""
    m = int(mask)
    while m:
        b = m & -m
        yield b.bit_length() - 1
        m ^= b


def solve(root, worlds, weights, oracle, payoff: str = "points") -> SolveResult:
    """Exact best response of ``root.me`` vs the deterministic field ``oracle``.

    Args:
        root: EndgameRoot (must be one of ``root.me``'s decisions).
        worlds: (N, 3) uint32 CURRENT-remaining hand masks for the three non-me
            seats in ascending absolute seat order (walt/worlds.py contract).
        weights: (N,) belief weights (need not be normalized).
        oracle: walt.field.FieldOracle (jud argmax, batched + memoized).
        payoff: 'points' -> E[declaring-team final points]; 'make' -> P(declaring
            final points >= root.bid_value). In both cases the value is reported
            in declaring orientation and the walt seat maximizes sign*value.
    """
    if payoff not in ("points", "make"):
        raise ValueError(f"payoff must be 'points' or 'make', got {payoff!r}")

    worlds = np.asarray(worlds, dtype=np.uint32).reshape(-1, 3)
    N = int(worlds.shape[0])
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if N == 0:
        raise ValueError("solve() needs at least one world")
    if weights.shape[0] != N:
        raise ValueError(f"weights ({weights.shape[0]}) must match worlds ({N})")

    decl = int(root.decl_id)
    bidder = int(root.bidder)
    me = int(root.me)
    bid_team = bidder % 2
    sign = 1.0 if me % 2 == bid_team else -1.0
    bids = tuple(int(b) for b in root.bids)
    dealer = int(root.dealer)
    bid_value = int(root.bid_value)
    auction = (decl, bidder, bids, dealer)

    seats = [s for s in range(4) if s != me]          # ascending non-me seats
    col_of = {s: i for i, s in enumerate(seats)}
    luts = get_luts(decl)
    _lsuit = [int(x) for x in luts.led_suit]          # python ints for hot path
    _cfb = [int(x) for x in luts.can_follow_bits]

    def legal_int(mask: int, led) -> int:
        """Follow-suit-legal mask, pure-int (== tables.legal_moves_mask)."""
        if led is None:
            return mask
        f = mask & _cfb[_lsuit[led]]
        return f if f else mask

    total_w = float(weights.sum())
    stats = {"nodes": 0, "queries": 0}
    # NOTE: no solver-level memo. Node identity is the full public history,
    # which is unique per node of the recursion tree (paths are append-only),
    # so a (hand, history)-keyed memo provably never hits — measured 0 hits
    # in 19,566 lookups on a representative H4 solve. The memo that pays is
    # the FieldOracle's (seat, hand, pubkey) decision memo.

    def leaf_value(banked: tuple) -> float:
        decl_pts = banked[bid_team]
        if payoff == "points":
            return float(decl_pts)
        return 1.0 if decl_pts >= bid_value else 0.0

    def rec(ctx, ct, leader, banked, my_mask, rem, w) -> float:
        """Weighted SUM (over the node's worlds) of the declaring-oriented payoff."""
        stats["nodes"] += 1
        if len(ctx.hist) == 28:                  # terminal: hands fully pinned
            return leaf_value(banked) * float(w.sum())

        cp = (leader + len(ct)) % 4
        if cp == me:
            led = ct[0] if ct else None
            lm = legal_int(my_mask, led)
            best = None
            for a in _bits(lm):
                nctx = ctx.advance(me, a, bidder, luts)
                nct = ct + (a,)
                nm = my_mask & ~(1 << a)
                if len(nct) == 4:
                    off, p = resolve_lut(nct, luts)
                    wn = (leader + off) % 4
                    nb = list(banked)
                    nb[wn % 2] += p
                    v = rec(nctx, (), wn, (nb[0], nb[1]), nm, rem, w)
                else:
                    v = rec(nctx, nct, leader, banked, nm, rem, w)
                if best is None or sign * v > sign * best:
                    best = v
            return best

        # opponent (or partner) — all non-me seats roll via σ
        col = col_of[cp]
        stats["queries"] += int(rem.shape[0])
        mv = oracle.decisions_at(cp, ctx, rem[:, col], ct, auction, luts)
        total = 0.0
        for m in np.unique(mv):
            g = mv == m
            mi = int(m)
            nctx = ctx.advance(cp, mi, bidder, luts)
            nct = ct + (mi,)
            nrem = rem[g]                 # boolean indexing already copies
            nrem[:, col] &= np.uint32((~(1 << mi)) & 0xFFFFFFFF)
            nw = w[g]
            if len(nct) == 4:
                off, p = resolve_lut(nct, luts)
                wn = (leader + off) % 4
                nb = list(banked)
                nb[wn % 2] += p
                total += rec(nctx, (), wn, (nb[0], nb[1]), my_mask, nrem, nw)
            else:
                total += rec(nctx, nct, leader, banked, my_mask, nrem, nw)
        return total

    # ---- root decision (a me-node), captured explicitly to record best_move ----
    hist0 = tuple((int(s), int(d)) for s, d in root.play_history)
    ct0 = tuple(int(d) for d in root.current_trick)
    my0 = int(hand_to_mask(root.my_hand))
    banked0 = (int(root.team_points[0]), int(root.team_points[1]))
    leader0 = int(root.trick_leader)
    if (leader0 + len(ct0)) % 4 != me:
        raise ValueError("root must be a decision of root.me")

    ctx0 = NodeCtx.from_history(hist0, bidder, luts)
    stats["nodes"] += 1
    led = ct0[0] if ct0 else None
    lm = legal_int(my0, led)
    best_val = None
    best_move = None
    for a in _bits(lm):                          # ascending → ties keep lowest id
        nctx = ctx0.advance(me, a, bidder, luts)
        nct = ct0 + (a,)
        nm = my0 & ~(1 << a)
        if len(nct) == 4:
            off, p = resolve_lut(nct, luts)
            wn = (leader0 + off) % 4
            nb = list(banked0)
            nb[wn % 2] += p
            v = rec(nctx, (), wn, (nb[0], nb[1]), nm, worlds, weights)
        else:
            v = rec(nctx, nct, leader0, banked0, nm, worlds, weights)
        if best_val is None or sign * v > sign * best_val:
            best_val = v
            best_move = a

    value = best_val / total_w

    # ---- walker instrumentation (only meaningful when I am leading) ----
    walker_flags: dict = {}
    if not ct0:
        walker_flags = _walker_flags(
            ctx0, my0, worlds, weights, total_w, me, bidder, auction,
            col_of, luts, oracle, stats,
        )

    return SolveResult(
        value=value,
        best_move=int(best_move),
        n_worlds=N,
        n_nodes=stats["nodes"],
        n_field_queries=stats["queries"],
        walker_flags=walker_flags,
    )


def _walker_flags(ctx0, my0, worlds, weights, total_w, me, bidder, auction,
                  col_of, luts, oracle, stats) -> dict:
    """Flag my leads that win the current trick in EVERY alive world while their
    global beat_count ranks bottom-half among all 28 tiles (the walker signature:
    trash by beat-count, unbeatable when led, harvesting the riding count)."""
    bc = luts.beat_count
    # bottom-half by beat_count: strictly-less rank < 14 (of 28).
    rank_of = {t: int(np.sum(bc < bc[t])) for t in range(28)}
    decl = auction[0]

    def finish_trick(ctx, ct, leader, rem, w):
        """List of (winner_seat, trick_points, weight) groups for a completed
        trick, rolling the remaining responders via σ (partitioned by world)."""
        if len(ct) == 4:
            off, p = resolve_lut(ct, luts)
            wn = (leader + off) % 4
            return [(wn, int(p), float(w.sum()))]
        cp = (leader + len(ct)) % 4   # never me here (me already led at position 0)
        col = col_of[cp]
        stats["queries"] += int(rem.shape[0])
        mv = oracle.decisions_at(cp, ctx, rem[:, col], ct, auction, luts)
        out = []
        for m in np.unique(mv):
            g = mv == m
            mi = int(m)
            nrem = rem[g]
            nrem[:, col] &= np.uint32((~(1 << mi)) & 0xFFFFFFFF)
            out += finish_trick(
                ctx.advance(cp, mi, bidder, luts), ct + (mi,), leader, nrem, w[g]
            )
        return out

    flags: dict = {}
    lm = int(legal_moves_mask(np.uint32(my0), None, decl))
    for a in _bits(lm):
        groups = finish_trick(
            ctx0.advance(me, a, bidder, luts), (a,), me, worlds, weights
        )
        win_w = sum(wt for (winner, _pts, wt) in groups if winner == me)
        wins_all = abs(win_w - total_w) < 1e-9
        rk = rank_of[a]
        if wins_all and rk < 14:
            pts = sorted({p for (_wn, p, _wt) in groups})
            flags[int(a)] = {
                "beat_count": int(bc[a]),
                "beat_rank": rk,
                "wins_all": True,
                "trick_points": pts[0] if len(pts) == 1 else pts,
                "is_walker": True,
            }
    return flags
