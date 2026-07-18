"""walt/solver.py — the exact endgame information-set solver (wavefront engine).

`solve()` computes the exact best response of ``root.me`` in the information-set
game against the deterministic field ``oracle`` (jud argmax), per DESIGN.md and
contracts.py. Strategy fusion is killed by construction: every world that
reaches a node shares that node's public history, hence shares one information
set and one chosen move.

Value unit: E[final declaring-team hand points, 0..42] (banked + remaining), the
same orientation as arena.jud_play.JudPlay. The walt seat maximizes ``sign*value``
with ``sign = +1 if me%2 == bidder%2 else -1`` FIXED at the root (equivalently:
declarer maximizes declaring points, defender minimizes them). ``value`` in the
result is always reported in declaring orientation (never sign-flipped).

Engine shape (the issue-#74 "frontier batching" rewrite): instead of a per-node
depth-first recursion, the tree is walked in level-synchronous waves — every
node of a wave sits at the same play index p, so per-play featurization
constants (trick position, trick index, fill one-hot) are wave-wide scalars.
All state is struct-of-arrays numpy over the wave's nodes and world-slots:

- world-slots: (hand-mask triple, weight, owning node), grouped by node;
  σ moves partition a node's slots among its children, my moves replicate them.
- per wave, ALL σ decisions are resolved together: slots are deduped to unique
  (node, acting hand) decisions, one-legal decisions short-circuit, and every
  remaining decision has its legal-child feature rows built in a handful of
  vectorized scatters and scored by ONE chunked net forward for the whole wave
  (FieldOracle.ev_rows) — thousands of rows per forward instead of the old 4.6.
- play blocks for miss nodes are rebuilt from the root's NodeCtx blocks plus
  the node's play path (walked by vectorized parent-pointer gathers), which is
  bit-identical to the incremental NodeCtx maintenance it replaces.
- values propagate backward wave by wave: σ nodes sum their children in
  ascending-move order (np.bincount preserves it), my nodes take the sign-max;
  the root replays the baseline tie rule verbatim (moves ascending, first
  strict ``sign*v > sign*best`` improvement wins).

σ decision semantics are unchanged: value = mean_points of the child logits,
sign = +1 iff the mover is on the bidding team, argmax of sign*value with
first-max (lowest domino id) tie-breaking, exactly as FieldOracle/JudPlay.

There is no cross-solve σ-decision memo in this path: within a solve every
public history is unique (paths are append-only), so the old (seat, hand,
pubkey) memo could only hit across solves, and with batched forwards a hit now
saves ~2µs of row work while costing dict maintenance per node. The per-wave
(node, hand) dedup preserves the memo's entire within-solve effect.

``n_field_queries`` counts total per-world oracle decision requests issued
across the solve (sum over σ nodes of alive worlds, plus the walker
instrumentation's trick rollouts); ``n_nodes`` counts tree nodes including the
root and terminal leaves — both identical to the recursive implementation.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

from walt.contracts import SolveResult
from walt.field import (
    AUCTION_DIM,
    HAND_DIM,
    N_TRICKS,
    PLAY_DIM,
    _GLOBAL_OFF,
    NodeCtx,
    resolve_lut,
)
from walt.tables import get_luts, hand_to_mask, legal_moves_mask

_F91 = HAND_DIM + AUCTION_DIM        # 91: play-block offset within a feature row
_G = _GLOBAL_OFF                     # 252: global-tail offset within the block
_FEAT = _F91 + PLAY_DIM              # 350
_AR28 = np.arange(28)
_ALL28 = np.int64((1 << 28) - 1)
_I64_1 = np.int64(1)


def _bits(mask: int) -> Iterator[int]:
    """Yield the domino ids set in a bitmask, ascending."""
    m = int(mask)
    while m:
        b = m & -m
        yield b.bit_length() - 1
        m ^= b


def _empty_wave() -> dict:
    z = np.empty(0, dtype=np.int64)
    return {"parent": z, "tile": z, "pseat": z, "me": np.empty(0, dtype=bool)}


def solve(root, worlds, weights, oracle, payoff: str = "points") -> SolveResult:
    """Exact best response of ``root.me`` vs the deterministic field ``oracle``.

    Args:
        root: EndgameRoot (must be one of ``root.me``'s decisions).
        worlds: (N, 3) uint32 CURRENT-remaining hand masks for the three non-me
            seats in ascending absolute seat order (walt/worlds.py contract).
        weights: (N,) belief weights (need not be normalized).
        oracle: walt.field.FieldOracle (jud argmax, batched).
        payoff: 'points' -> E[declaring-team final points]; 'make' -> P(declaring
            final points >= root.bid_value). In both cases the value is reported
            in declaring orientation and the walt seat maximizes sign*value.
    """
    if payoff not in ("points", "make"):
        raise ValueError(f"payoff must be 'points' or 'make', got {payoff!r}")

    worlds_u32 = np.asarray(worlds, dtype=np.uint32).reshape(-1, 3)
    N = int(worlds_u32.shape[0])
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
    luts = get_luts(decl)

    # int64 LUT copies: uint32/int8 mixing costs casts in every hot expression
    LED = luts.led_suit.astype(np.int64)
    RANK = luts.rank.astype(np.int64)
    CNT = luts.count.astype(np.int64)
    CFB = luts.can_follow_bits.astype(np.int64)
    col_arr = np.full(4, -1, dtype=np.int64)     # seat -> world column
    for i, s in enumerate(s for s in range(4) if s != me):
        col_arr[s] = i

    total_w = float(weights.sum())

    # ---- root public state --------------------------------------------------
    hist0 = tuple((int(s), int(d)) for s, d in root.play_history)
    ct0 = tuple(int(d) for d in root.current_trick)
    p0 = len(hist0)
    my0 = int(hand_to_mask(root.my_hand))
    leader0 = int(root.trick_leader)
    if (leader0 + len(ct0)) % 4 != me:
        raise ValueError("root must be a decision of root.me")
    tp = tuple(int(x) for x in root.team_points)

    ctx0 = NodeCtx.from_history(hist0, bidder, luts)
    root_blocks = ctx0.blocks                     # (4, 259) float32, per POV seat
    # featurization pts come from the history replay (NodeCtx parity); banked
    # value pts come from root.team_points, exactly as the recursion did.
    ptsf0 = ctx0.pts                              # (declaring, defending)
    ptsv0 = tp[bid_team]                          # declaring, for leaf payoffs

    led0, brank0, bseat0, cnt0 = -1, -1, -1, 0    # current-trick carry state
    if ct0:
        led0 = int(LED[ct0[0]])
        for i, t in enumerate(ct0):
            r = int(RANK[led0, t])
            if r > brank0:
                brank0, bseat0 = r, (leader0 + i) % 4
            cnt0 += int(CNT[t])

    # ---- wave-0 node arrays (single node: the root) --------------------------
    leader = np.array([leader0], dtype=np.int64)
    led = np.array([led0], dtype=np.int64)
    brank = np.array([brank0], dtype=np.int64)
    bseat = np.array([bseat0], dtype=np.int64)
    tcnt = np.array([cnt0], dtype=np.int64)
    ptsfd = np.array([ptsf0[0]], dtype=np.int64)  # featurization: declaring
    ptsff = np.array([ptsf0[1]], dtype=np.int64)  # featurization: defending
    ptsvd = np.array([ptsv0], dtype=np.int64)     # value: declaring banked
    mymask = np.array([my0], dtype=np.int64)
    played = np.array([ctx0.played], dtype=np.int64).reshape(1, 4)

    sw = worlds_u32.astype(np.int64)              # (S, 3) hand masks
    swt = weights.copy()                          # (S,)
    snode = np.zeros(N, dtype=np.int64)           # slot -> owning node
    counts = np.array([N], dtype=np.int64)        # slots per node

    waves: list[dict] = [_empty_wave()]           # waves[0] = root
    n_nodes = 1
    n_queries = 0
    h91_local: dict[tuple, np.ndarray] = {}       # (seat, orig_mask) -> [91] f32

    # ---- forward expansion: one wave per play index --------------------------
    for p in range(p0, 28):
        pos = p & 3
        cw = len(waves) - 1                       # index of the wave being expanded
        actor = (leader + pos) % 4
        me_mask = actor == me
        waves[cw]["me"] = me_mask
        starts_node = np.concatenate(([0], np.cumsum(counts)))

        # ---- σ nodes: one deduped decision pass for the whole wave ----------
        slot_actor_me = me_mask[snode]
        sel = np.flatnonzero(~slot_actor_me)
        n_queries += len(sel)
        if len(sel):
            nsl = snode[sel]
            cols = col_arr[actor[nsl]]
            h = sw[sel, cols]
            uq, inv = np.unique((nsl << 28) | h, return_inverse=True)
            u_n = uq >> 28
            u_h = uq & _ALL28
            ls = led[u_n]
            fb = np.where(ls >= 0, CFB[ls], _ALL28)
            lm = u_h & fb
            lm = np.where(lm != 0, lm, u_h)
            bm = ((lm[:, None] >> _AR28) & 1).astype(bool)   # (D, 28) legal sets
            ncand = bm.sum(axis=1)
            mv_dec = np.empty(len(uq), dtype=np.int64)
            one = ncand == 1
            if one.any():
                mv_dec[one] = np.argmax(bm[one], axis=1)
            multi = np.flatnonzero(~one)

            if len(multi):
                m_n = u_n[multi]
                m_h = u_h[multi]
                m_act = actor[m_n]
                # play blocks for the unique miss nodes, from the acting seat's
                # POV: root block + the node's play path (parent-pointer walk)
                bn, binv = np.unique(m_n, return_inverse=True)
                U = len(bn)
                povs = actor[bn]
                blocks = root_blocks[povs].copy()            # (U, 259)
                arU = np.arange(U)
                anc = bn
                for j in range(cw, 0, -1):
                    wj = waves[j]
                    tj = wj["tile"][anc]
                    sj = wj["pseat"][anc]
                    k = p0 + j - 1
                    base = 9 * tj
                    blocks[arU, base + (sj - povs) % 4] = 1.0
                    blocks[arU, base + 4 + (k & 3)] = 1.0
                    blocks[arU, base + 8] = np.float32((k // 4) / (N_TRICKS - 1))
                    anc = wj["parent"][anc]
                blocks[:, _G + 3:_G + 7] = 0.0
                blocks[:, _G] = (ptsfd[bn] / 42.0).astype(np.float32)
                blocks[:, _G + 1] = (ptsff[bn] / 42.0).astype(np.float32)

                # child-invariant 91-dim head per decision (cached)
                D2 = len(multi)
                H91 = np.empty((D2, 91), dtype=np.float32)
                orig_l = (m_h | played[m_n, m_act]).tolist()
                act_l = m_act.tolist()
                for i in range(D2):
                    hk = (act_l[i], orig_l[i])
                    row = h91_local.get(hk)
                    if row is None:
                        row = oracle._h91(hk[0], hk[1], bids, bidder, dealer, decl)
                        h91_local[hk] = row
                    H91[i] = row

                # one feature row per legal child move, wave-wide
                bm2 = bm[multi]
                di, mv2 = np.nonzero(bm2)          # ascending (decision, move)
                R = len(di)
                starts = np.zeros(D2, dtype=np.int64)
                np.cumsum(ncand[multi][:-1], out=starts[1:])
                X = np.empty((R, _FEAT), dtype=np.float32)
                X[:, :_F91] = H91[di]
                X[:, _F91:] = blocks[binv[di]]
                arR = np.arange(R)
                mb = _F91 + 9 * mv2
                X[arR, mb] = 1.0                   # mover is POV
                X[arR, mb + 4 + pos] = 1.0
                X[arR, mb + 8] = np.float32((p // 4) / (N_TRICKS - 1))
                X[:, _F91 + _G + 2] = np.float32((p + 1) / 28.0)
                X[:, _F91 + _G + 3 + ((p + 1) & 3)] = 1.0
                if pos == 3:                       # move completes a trick
                    nrow = m_n[di]
                    r_m = RANK[led[nrow], mv2]
                    win = np.where(r_m > brank[nrow], m_act[di], bseat[nrow])
                    padd = tcnt[nrow] + CNT[mv2] + 1
                    dwin = (win % 2) == bid_team
                    nd = ptsfd[nrow] + np.where(dwin, padd, 0)
                    nf = ptsff[nrow] + np.where(dwin, 0, padd)
                    X[:, _F91 + _G] = (nd / 42.0).astype(np.float32)
                    X[:, _F91 + _G + 1] = (nf / 42.0).astype(np.float32)

                ev = oracle.ev_rows(X)
                sig = np.where(m_act % 2 == bid_team,
                               np.float32(1.0), np.float32(-1.0))
                sev = ev * sig[di]
                smax = np.maximum.reduceat(sev, starts)
                posr = np.where(sev == smax[di], arR, R)     # first-max tie rule
                first = np.minimum.reduceat(posr, starts)
                mv_dec[multi] = mv2[first]

            # partition σ slots among observation-branch children
            mv_slot = mv_dec[inv]
            swc = sw[sel]
            swc[np.arange(len(sel)), cols] &= ~(_I64_1 << mv_slot)
            swtc = swt[sel]
            cuq, cinv = np.unique((nsl << 5) | mv_slot, return_inverse=True)
            order = np.argsort(cinv, kind="stable")
            sw_sig = swc[order]
            swt_sig = swtc[order]
            child_sig = cinv[order]
            par_sig = cuq >> 5
            tile_sig = cuq & 31
            counts_sig = np.bincount(cinv, minlength=len(cuq)).astype(np.int64)
        else:
            sw_sig = np.empty((0, 3), dtype=np.int64)
            swt_sig = np.empty(0, dtype=np.float64)
            child_sig = par_sig = tile_sig = counts_sig = \
                np.empty(0, dtype=np.int64)

        # ---- my nodes: branch over my legal moves, replicating slots --------
        men = np.flatnonzero(me_mask)
        if len(men):
            lsm = led[men]
            fb = np.where(lsm >= 0, CFB[lsm], _ALL28)
            hm = mymask[men]
            lm_me = hm & fb
            lm_me = np.where(lm_me != 0, lm_me, hm)
            bmm = ((lm_me[:, None] >> _AR28) & 1).astype(bool)
            pi, tile_me = np.nonzero(bmm)          # ascending (node, move)
            tile_me = tile_me.astype(np.int64)
            par_me = men[pi]
            sizes = counts[par_me]
            total = int(sizes.sum())
            cum0 = np.zeros(len(sizes), dtype=np.int64)
            np.cumsum(sizes[:-1], out=cum0[1:])
            gidx = np.repeat(starts_node[par_me] - cum0, sizes) + np.arange(total)
            sw_me = sw[gidx]
            swt_me = swt[gidx]
            child_me = np.repeat(np.arange(len(par_me)), sizes)
            mymask_me = mymask[par_me] & ~(_I64_1 << tile_me)
        else:
            sw_me = np.empty((0, 3), dtype=np.int64)
            swt_me = np.empty(0, dtype=np.float64)
            child_me = par_me = tile_me = np.empty(0, dtype=np.int64)
            sizes = np.empty(0, dtype=np.int64)
            mymask_me = np.empty(0, dtype=np.int64)

        # ---- build the child wave (σ children first, then mine) -------------
        C1 = len(par_sig)
        parent_c = np.concatenate((par_sig, par_me))
        tile_c = np.concatenate((tile_sig, tile_me))
        pseat_c = actor[parent_c]
        C = len(parent_c)
        n_nodes += C

        played_c = played[parent_c]
        played_c[np.arange(C), pseat_c] |= _I64_1 << tile_c
        mymask_c = np.concatenate((mymask[par_sig], mymask_me))
        leader_p = leader[parent_c]
        ptsfd_c = ptsfd[parent_c]
        ptsff_c = ptsff[parent_c]
        ptsvd_c = ptsvd[parent_c]
        if pos == 0:                               # this play leads a trick
            led_c = LED[tile_c]
            brank_c = RANK[led_c, tile_c]
            bseat_c = pseat_c.copy()
            tcnt_c = CNT[tile_c]
            leader_c = leader_p
        elif pos < 3:                              # mid-trick response
            led_c = led[parent_c]
            r = RANK[led_c, tile_c]
            bet = r > brank[parent_c]
            brank_c = np.where(bet, r, brank[parent_c])
            bseat_c = np.where(bet, pseat_c, bseat[parent_c])
            tcnt_c = tcnt[parent_c] + CNT[tile_c]
            leader_c = leader_p
        else:                                      # completes the trick
            r = RANK[led[parent_c], tile_c]
            winner = np.where(r > brank[parent_c], pseat_c, bseat[parent_c])
            padd = tcnt[parent_c] + CNT[tile_c] + 1
            dwin = (winner % 2) == bid_team
            gain = np.where(dwin, padd, 0)
            ptsfd_c = ptsfd_c + gain
            ptsff_c = ptsff_c + np.where(dwin, 0, padd)
            ptsvd_c = ptsvd_c + gain
            leader_c = winner
            led_c = np.full(C, -1, dtype=np.int64)
            brank_c = np.full(C, -1, dtype=np.int64)
            bseat_c = np.full(C, -1, dtype=np.int64)
            tcnt_c = np.zeros(C, dtype=np.int64)

        waves.append({"parent": parent_c, "tile": tile_c, "pseat": pseat_c,
                      "me": None})
        leader, led, brank, bseat, tcnt = leader_c, led_c, brank_c, bseat_c, tcnt_c
        ptsfd, ptsff, ptsvd = ptsfd_c, ptsff_c, ptsvd_c
        mymask, played = mymask_c, played_c
        sw = np.concatenate((sw_sig, sw_me))
        swt = np.concatenate((swt_sig, swt_me))
        snode = np.concatenate((child_sig, C1 + child_me))
        counts = np.concatenate((counts_sig, sizes))

    # ---- leaf values, then backward propagation ------------------------------
    L = len(waves) - 1
    wsum = np.bincount(snode, weights=swt, minlength=len(ptsvd))
    if payoff == "points":
        leafv = ptsvd.astype(np.float64)
    else:
        leafv = (ptsvd >= bid_value).astype(np.float64)
    vals = leafv * wsum

    for j in range(L - 1, 0, -1):
        par = waves[j + 1]["parent"]
        memask_j = waves[j]["me"]
        M = len(memask_j)
        par_is_me = memask_j[par]
        # astype guards the all-me wave: bincount of an empty selection
        # returns int64 zeros, and int64 acc would truncate the sign-max
        # assignment below
        acc = np.bincount(
            par[~par_is_me], weights=vals[~par_is_me], minlength=M,
        ).astype(np.float64, copy=False)
        seli = np.flatnonzero(par_is_me)
        if len(seli):
            # my children sit in a contiguous tail block sorted by parent,
            # each parent's children ascending by move
            pm = par[seli]
            sv = sign * vals[seli]
            seg = np.concatenate(([0], np.flatnonzero(np.diff(pm)) + 1))
            acc[pm[seg]] = sign * np.maximum.reduceat(sv, seg)
        vals = acc

    # ---- root decision: baseline tie rule verbatim ---------------------------
    t1 = waves[1]["tile"]                          # root children, moves ascending
    best_val = None
    best_move = None
    root_values: dict[int, float] = {}
    for i in range(len(t1)):
        v = float(vals[i])
        root_values[int(t1[i])] = v / total_w
        if best_val is None or sign * v > sign * best_val:
            best_val = v
            best_move = int(t1[i])
    value = best_val / total_w

    # ---- walker instrumentation (only meaningful when I am leading) ----------
    stats = {"queries": n_queries}
    walker_flags: dict = {}
    if not ct0:
        col_of = {s: i for i, s in enumerate(s for s in range(4) if s != me)}
        walker_flags = _walker_flags(
            ctx0, my0, worlds_u32, weights, total_w, me, bidder, auction,
            col_of, luts, oracle, stats,
        )

    return SolveResult(
        value=value,
        best_move=int(best_move),
        n_worlds=N,
        n_nodes=n_nodes,
        n_field_queries=stats["queries"],
        walker_flags=walker_flags,
        root_values=root_values,
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
