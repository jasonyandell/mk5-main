"""hoyt/subgame.py — the net-free wave engine over the H≤5 subgame.

`build_subgame(root, worlds, weights)` is the CONTRACTS.md entry point. The
Subgame it returns is deliberately LAZY: it holds the root's parsed public
state, the world distribution, and the rule LUTs — no tree. The reachable
tree depends on who is best-responding and what profile the other seats
play (a deterministic profile PARTITIONS worlds where full width REPLICATES
them), so eager enumeration would always build the wrong (or a hostile
union) tree. Instead `run_engine` walks the tree wave-at-a-time in the
wavefront solver's SoA shape (walt/solver.py, issue #74), with the σ
decision step abstracted behind a provider:

- one move per slot (SigmaTable replay / rule σ / jud-net capture) —
  worlds partition among observation branches, walt-identical;
- (moves, probs) per slot (StochasticProfile / uniform) — slots replicate
  with weight *= prob, the expectimax stress shape;
- all legal moves per slot (`expand_full_width`) — the CFR lane's
  structural walk: chance = the world distribution, a world survives a
  public action iff the acting hidden seat holds the tile.

Node identity is positional per wave (children sorted ascending by
(parent, move), walt parity); content identity for cross-walk consumers is
the 128-bit rolling path hash (`ph1`/`ph2` wave arrays, enabled by
`need_path_ids`), which profiles.StochasticProfile uses as its node key.

Values: leaves score `payoff43[declaring_points] * weight`; the backward
pass is walt's bincount / segmented sign-max, so hero's value is exact and
strategy fusion stays dead by construction. Everything here is numpy over
`walt.tables` LUTs — no torch anywhere in this module.
"""
from __future__ import annotations

import hashlib

import numpy as np

from walt.tables import get_luts, hand_to_mask

_AR28 = np.arange(28)
_ALL28 = np.int64((1 << 28) - 1)
_I64_1 = np.int64(1)
# per-move clear masks, int32: bits 28-31 stay set after truncation, so
# 28-bit payloads are unaffected; gathering these avoids the int64
# promotion an `x &= ~(_I64_1 << mv)` would force on int32 slot arrays
_NB28_I32 = (~(_I64_1 << np.arange(28))).astype(np.int32)

# 128-bit rolling path hash (two independent 64-bit lanes, wraparound mult).
_H1_0 = np.uint64(0xCBF29CE484222325)
_H2_0 = np.uint64(0x9E3779B97F4A7C15)
_H1_P = np.uint64(0x100000001B3)
_H2_P = np.uint64(0xDA942042E4DD58B5)
_M64 = (1 << 64) - 1

# ~1.4GB of slot payload; stochastic full-support trees can hit this.
DEFAULT_SLOT_BUDGET = 32_000_000


class KernelMemoryError(MemoryError):
    """A wave outgrew the slot budget; caller should chunk (root move /
    smaller world cap) rather than swap the machine to death."""


def path_hash(path) -> tuple[int, int]:
    """The (h1, h2) rolling hash of a public tile path from the subgame
    root — python mirror of the vectorized per-wave update; the node key
    used by profiles.StochasticProfile."""
    h1, h2 = int(_H1_0), int(_H2_0)
    for t in path:
        v = int(t) + 1
        h1 = (h1 * int(_H1_P) + v) & _M64
        h2 = ((h2 ^ v) * int(_H2_P)) & _M64
    return h1, h2


class Subgame:
    """Parsed root public state + world distribution + LUTs. Opaque per
    CONTRACTS.md; tree enumeration is lazy (see module docstring)."""

    def __init__(self, root, worlds, weights):
        worlds_u32 = np.asarray(worlds, dtype=np.uint32).reshape(-1, 3)
        N = int(worlds_u32.shape[0])
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if N == 0:
            raise ValueError("build_subgame needs at least one world")
        if weights.shape[0] != N:
            raise ValueError(
                f"weights ({weights.shape[0]}) must match worlds ({N})")

        self.root = root
        self.worlds = worlds_u32
        # int32, not uint32: the walk's slot arrays are int32 and
        # uint32/int32 mixing would promote every expression to int64
        self.worlds_i32 = worlds_u32.astype(np.int32)
        self.weights = weights
        self.total_w = float(weights.sum())

        self.decl = int(root.decl_id)
        self.bidder = int(root.bidder)
        self.me = int(root.me)
        self.bid_team = self.bidder % 2
        self.bids = tuple(int(b) for b in root.bids)
        self.dealer = int(root.dealer)
        self.bid_value = int(root.bid_value)
        self.luts = get_luts(self.decl)

        # int64 LUT copies (uint32/int8 mixing costs casts per hot expression)
        self.LED = self.luts.led_suit.astype(np.int64)
        self.RANK = self.luts.rank.astype(np.int64)
        self.CNT = self.luts.count.astype(np.int64)
        self.CFB = self.luts.can_follow_bits.astype(np.int64)
        self.col_arr = np.full(4, -1, dtype=np.int64)   # seat -> world column
        self.hidden_seats = tuple(s for s in range(4) if s != self.me)
        for i, s in enumerate(self.hidden_seats):
            self.col_arr[s] = i

        self.hist0 = tuple((int(s), int(d)) for s, d in root.play_history)
        self.ct0 = tuple(int(d) for d in root.current_trick)
        self.p0 = len(self.hist0)
        self.my0 = int(hand_to_mask(root.my_hand))
        self.leader0 = int(root.trick_leader)
        if (self.leader0 + len(self.ct0)) % 4 != self.me:
            raise ValueError("root must be a decision of root.me")
        tp = tuple(int(x) for x in root.team_points)
        self.ptsv0 = tp[self.bid_team]                  # banked declaring pts

        # current-trick carry state (walt parity)
        led0, brank0, bseat0, cnt0 = -1, -1, -1, 0
        if self.ct0:
            led0 = int(self.LED[self.ct0[0]])
            for i, t in enumerate(self.ct0):
                r = int(self.RANK[led0, t])
                if r > brank0:
                    brank0, bseat0 = r, (self.leader0 + i) % 4
                cnt0 += int(self.CNT[t])
        self.led0, self.brank0, self.bseat0, self.cnt0 = \
            led0, brank0, bseat0, cnt0

        # identity: binds a SigmaTable to (root information set, world set).
        # Weights are deliberately NOT bound — swapping beliefs re-solves free.
        self.root_key = (self.decl, self.bidder, self.bids, self.dealer,
                         self.me, tuple(int(t) for t in root.my_hand),
                         self.hist0)
        self.worlds_digest = hashlib.blake2b(
            worlds_u32.tobytes(), digest_size=16).hexdigest()


def build_subgame(root, worlds, weights) -> Subgame:
    """CONTRACTS.md entry point: the (lazy) net-free subgame."""
    return Subgame(root, worlds, weights)


def _empty_wave() -> dict:
    z = np.empty(0, dtype=np.int64)
    return {"parent": z, "tile": z, "pseat": z, "hero": None}


def _cat2(a, b):
    """concatenate((a, b)) without the full-array copy when either side is
    empty — the hero=-1 full-width walk concatenates an empty hero side
    every wave (perf-log 18p). Both inputs are freshly built per wave, so
    returning one unaliased is safe."""
    if not len(b):
        return a
    if not len(a):
        return b
    return np.concatenate((a, b))


def run_engine(sub: Subgame, provider, *, hero=None, worlds=None,
               weights=None, heromask0=None, capture=False,
               keep_slots=False, need_path_ids=False, net_state=None,
               root_moves=None, slot_budget=DEFAULT_SLOT_BUDGET) -> dict:
    """Forward wave expansion, walt/solver.py shape, σ step via ``provider``.

    provider.expand(eng, p, pos, sel, nsl, seats, hands) returns
    (rep, mv, wmul, dec): output profile slots are ``rep`` (absolute slot
    indices), playing ``mv``, with weight multiplied by ``wmul`` (None = 1,
    deterministic — then rep must be exactly ``sel``). ``dec`` optionally
    carries (node, hand, seat, move) unique-decision arrays for capture.

    hero: the full-width best-responding seat (default root.me); hero=-1
    means NO seat is full-width (profile_value / full-width walks).
    heromask0: hero's root hand mask when hero is a hidden seat (its hand
    is then constant across the given ``worlds`` — the caller groups).
    net_state: dict with ptsf0/played0 when a provider needs jud
    featurization state (compile_sigma only).

    Returns {"waves", "leaf", "n_nodes", "n_leaf_slots", "cap"} where
    waves[j] has parent/tile/pseat (int64) + hero flags of wave j's nodes,
    and leaf holds the final wave's (snode, swt, sworld, ptsvd, M).
    """
    me = sub.me
    hero = me if hero is None else int(hero)
    hero_is_me = hero == me
    LED, RANK, CNT, CFB = sub.LED, sub.RANK, sub.CNT, sub.CFB
    col_arr = sub.col_arr
    bid_team = sub.bid_team
    p0 = sub.p0

    sw = sub.worlds_i32 if worlds is None else \
        np.asarray(worlds, dtype=np.int32).reshape(-1, 3)
    N = int(sw.shape[0])
    swt = (sub.weights if weights is None else
           np.asarray(weights, dtype=np.float64)).copy()
    snode = np.zeros(N, dtype=np.int64)
    sworld = np.arange(N, dtype=np.int32)
    counts = np.array([N], dtype=np.int64)

    leader = np.array([sub.leader0], dtype=np.int64)
    led = np.array([sub.led0], dtype=np.int64)
    brank = np.array([sub.brank0], dtype=np.int64)
    bseat = np.array([sub.bseat0], dtype=np.int64)
    tcnt = np.array([sub.cnt0], dtype=np.int64)
    ptsvd = np.array([sub.ptsv0], dtype=np.int64)
    mymask = np.array([sub.my0], dtype=np.int64)
    if hero_is_me:
        heromask = mymask
    else:
        if hero >= 0 and heromask0 is None:
            raise ValueError("hidden hero needs heromask0 (group hand)")
        heromask = np.array([int(heromask0 or 0)], dtype=np.int64)

    track_net = net_state is not None
    if track_net:
        ptsfd = np.array([net_state["ptsf0"][0]], dtype=np.int64)
        ptsff = np.array([net_state["ptsf0"][1]], dtype=np.int64)
        played = np.array([net_state["played0"]],
                          dtype=np.int64).reshape(1, 4)
    if need_path_ids:
        ph1 = np.array([_H1_0], dtype=np.uint64)
        ph2 = np.array([_H2_0], dtype=np.uint64)

    waves: list[dict] = [_empty_wave()]
    n_nodes = 1
    cap = {"mv": [], "dec": []} if capture else None

    eng = _EngineView()  # provider-visible state, refreshed per wave

    for p in range(p0, 28):
        pos = p & 3
        cw = len(waves) - 1
        actor = (leader + pos) % 4
        hmask_nodes = actor == hero
        waves[cw]["hero"] = hmask_nodes
        if keep_slots:
            # node ids < slot budget, worlds < N, hands are 28-bit — int32
            # halves the resident slot payload. sw/sworld working arrays are
            # int32 too, so copy=False stores an alias (the walk rebinds
            # them per wave, never writes in place — sw_sig is a gather
            # copy); snode stays int64 (index array, and snode<<5 keys
            # can pass 2^31 near the slot budget), so its store copies
            waves[cw]["snode"] = snode.astype(np.int32)
            waves[cw]["sworld"] = sworld.astype(np.int32, copy=False)
            waves[cw]["counts"] = counts
            waves[cw]["sw"] = sw.astype(np.int32, copy=False)
            waves[cw]["actor"] = actor.astype(np.int8)
        if need_path_ids:
            waves[cw]["ph1"], waves[cw]["ph2"] = ph1, ph2

        eng.sub, eng.p0, eng.cw, eng.waves = sub, p0, cw, waves
        eng.actor, eng.led, eng.brank, eng.bseat, eng.tcnt = \
            actor, led, brank, bseat, tcnt
        eng.mymask, eng.heromask = mymask, heromask
        if track_net:
            eng.ptsfd, eng.ptsff, eng.played = ptsfd, ptsff, played
        if need_path_ids:
            eng.ph1, eng.ph2 = ph1, ph2

        # ---- profile (non-hero) slots -----------------------------------
        sel = np.flatnonzero(~hmask_nodes[snode])
        if len(sel):
            nsl = snode[sel]
            seats = actor[nsl]
            if hero_is_me:
                hands = sw[sel, col_arr[seats]]
            else:
                cols = col_arr[seats]           # -1 where the actor is me
                hands = sw[sel, np.where(cols < 0, 0, cols)]
                m_me = seats == me              # minority write beats a
                if m_me.any():                  # full-width where (18p)
                    hands[m_me] = mymask[nsl[m_me]]
            rep, mv, wmul, dec = provider.expand(
                eng, p, pos, sel, nsl, seats, hands)
            if capture:
                if wmul is not None or rep is not sel:
                    raise ValueError(
                        "capture requires a deterministic provider")
                cap["mv"].append(mv.astype(np.int8))
                cap["dec"].append(dec)

            key = (snode[rep] << 5) | mv
            order = np.argsort(key, kind="stable")
            rows = rep[order]
            mv_o = mv[order]
            skey = key[order]
            bnd = np.flatnonzero(skey[1:] != skey[:-1]) + 1
            seg = np.concatenate(([0], bnd))
            cuq = skey[seg]
            counts_sig = np.diff(np.concatenate((seg, [len(skey)])))
            par_sig = cuq >> 5
            tile_sig = cuq & 31

            sw_sig = sw[rows]
            seats_o = actor[snode[rows]]
            cols_o = col_arr[seats_o]
            hid = cols_o >= 0                   # me-as-profile: node-level
            sw_sig[hid, cols_o[hid]] &= _NB28_I32[mv_o[hid]]
            if wmul is None:
                swt_sig = swt[rows]
            else:
                swt_sig = (swt[rep] * wmul)[order]
            sworld_sig = sworld[rows]
        else:
            sw_sig = np.empty((0, 3), dtype=np.int32)
            swt_sig = np.empty(0, dtype=np.float64)
            sworld_sig = np.empty(0, dtype=np.int32)
            par_sig = tile_sig = counts_sig = np.empty(0, dtype=np.int64)
            rows = np.empty(0, dtype=np.int64)
            if capture:
                cap["mv"].append(np.empty(0, dtype=np.int8))
                cap["dec"].append(None)

        # ---- hero nodes: branch full-width, replicating slots ------------
        men = np.flatnonzero(hmask_nodes)
        if len(men):
            starts_node = np.concatenate(([0], np.cumsum(counts)))
            lsm = led[men]
            fb = np.where(lsm >= 0, CFB[lsm], _ALL28)
            hm = heromask[men]
            lm_me = hm & fb
            lm_me = np.where(lm_me != 0, lm_me, hm)
            if root_moves is not None and p == p0:
                lm_me = lm_me & np.int64(root_moves)
            bmm = ((lm_me[:, None] >> _AR28) & 1).astype(bool)
            pi, tile_me = np.nonzero(bmm)       # ascending (node, move)
            tile_me = tile_me.astype(np.int64)
            par_me = men[pi]
            sizes = counts[par_me]
            total = int(sizes.sum())
            cum0 = np.zeros(len(sizes), dtype=np.int64)
            np.cumsum(sizes[:-1], out=cum0[1:])
            gidx = np.repeat(starts_node[par_me] - cum0, sizes) \
                + np.arange(total)
            sw_me = sw[gidx]
            swt_me = swt[gidx]
            sworld_me = sworld[gidx]
            child_me_sizes = sizes
        else:
            sw_me = np.empty((0, 3), dtype=np.int32)
            swt_me = np.empty(0, dtype=np.float64)
            sworld_me = np.empty(0, dtype=np.int32)
            par_me = tile_me = np.empty(0, dtype=np.int64)
            child_me_sizes = np.empty(0, dtype=np.int64)
            gidx = np.empty(0, dtype=np.int64)

        # ---- build the child wave (σ children first, then hero's) --------
        parent_c = _cat2(par_sig, par_me)
        tile_c = _cat2(tile_sig, tile_me)
        pseat_c = actor[parent_c]
        C = len(parent_c)
        n_nodes += C
        bit_c = _I64_1 << tile_c

        mymask_c = mymask[parent_c]
        mymask_c = np.where(pseat_c == me, mymask_c & ~bit_c, mymask_c)
        if hero_is_me:
            heromask_c = mymask_c
        else:
            heromask_c = heromask[parent_c]
            if hero >= 0:
                heromask_c = np.where(
                    pseat_c == hero, heromask_c & ~bit_c, heromask_c)

        if track_net:
            played_c = played[parent_c]
            played_c[np.arange(C), pseat_c] |= bit_c
            ptsfd_c = ptsfd[parent_c]
            ptsff_c = ptsff[parent_c]
        ptsvd_c = ptsvd[parent_c]
        leader_p = leader[parent_c]
        if pos == 0:                            # this play leads a trick
            led_c = LED[tile_c]
            brank_c = RANK[led_c, tile_c]
            bseat_c = pseat_c.copy()
            tcnt_c = CNT[tile_c]
            leader_c = leader_p
        elif pos < 3:                           # mid-trick response
            led_c = led[parent_c]
            r = RANK[led_c, tile_c]
            bet = r > brank[parent_c]
            brank_c = np.where(bet, r, brank[parent_c])
            bseat_c = np.where(bet, pseat_c, bseat[parent_c])
            tcnt_c = tcnt[parent_c] + CNT[tile_c]
            leader_c = leader_p
        else:                                   # completes the trick
            r = RANK[led[parent_c], tile_c]
            winner = np.where(r > brank[parent_c], pseat_c, bseat[parent_c])
            padd = tcnt[parent_c] + CNT[tile_c] + 1
            dwin = (winner % 2) == bid_team
            gain = np.where(dwin, padd, 0)
            ptsvd_c = ptsvd_c + gain
            if track_net:
                ptsfd_c = ptsfd_c + gain
                ptsff_c = ptsff_c + np.where(dwin, 0, padd)
            leader_c = winner
            led_c = np.full(C, -1, dtype=np.int64)
            brank_c = np.full(C, -1, dtype=np.int64)
            bseat_c = np.full(C, -1, dtype=np.int64)
            tcnt_c = np.zeros(C, dtype=np.int64)

        if need_path_ids:
            t64 = tile_c.astype(np.uint64) + np.uint64(1)
            ph1 = ph1[parent_c] * _H1_P + t64
            ph2 = (ph2[parent_c] ^ t64) * _H2_P

        # compact per-node storage (SigmaTable dtypes); working arrays for
        # the transition above stay int64. Arithmetic consumers cast on read
        # (int8 tile in expressions like 9*tile would overflow silently).
        waves.append({"parent": parent_c.astype(np.int32),
                      "tile": tile_c.astype(np.int8),
                      "pseat": pseat_c.astype(np.int8),
                      "hero": None})
        if keep_slots:
            # parent-slot index per child slot: resident in the walk (rows
            # for σ children, gidx for hero's), so consumers never re-derive
            # it by (node, world) key search (perf-log 18n)
            waves[-1]["pslot"] = _cat2(rows, gidx).astype(np.int32)
        leader, led, brank, bseat, tcnt = \
            leader_c, led_c, brank_c, bseat_c, tcnt_c
        ptsvd, mymask = ptsvd_c, mymask_c
        heromask = mymask if hero_is_me else heromask_c
        if track_net:
            ptsfd, ptsff, played = ptsfd_c, ptsff_c, played_c
        sw = _cat2(sw_sig, sw_me)
        swt = _cat2(swt_sig, swt_me)
        sworld = _cat2(sworld_sig, sworld_me)
        snode = _cat2(
            np.repeat(np.arange(len(par_sig)), counts_sig),
            len(par_sig) + np.repeat(np.arange(len(par_me)),
                                     child_me_sizes))
        counts = _cat2(counts_sig, child_me_sizes)
        if sw.shape[0] > slot_budget:
            raise KernelMemoryError(
                f"wave p={p + 1} holds {sw.shape[0]} slots > budget "
                f"{slot_budget}; chunk by root move or cap worlds")

    L = len(waves) - 1
    if keep_slots:
        waves[L]["snode"] = snode.astype(np.int32)
        waves[L]["sworld"] = sworld.astype(np.int32, copy=False)
        waves[L]["counts"], waves[L]["sw"] = \
            counts, sw.astype(np.int32, copy=False)
    if need_path_ids:
        waves[L]["ph1"], waves[L]["ph2"] = ph1, ph2
    return {
        "waves": waves,
        "leaf": {"snode": snode, "swt": swt, "sworld": sworld,
                 "ptsvd": ptsvd, "M": len(ptsvd)},
        "n_nodes": n_nodes,
        "n_leaf_slots": len(snode),
        "cap": cap,
    }


class _EngineView:
    """Provider-visible slice of the walk state (plain attribute bag)."""
    __slots__ = ("sub", "p0", "cw", "waves", "actor", "led", "brank",
                 "bseat", "tcnt", "mymask", "heromask", "ptsfd", "ptsff",
                 "played", "ph1", "ph2")


def backward(waves, vals, sign, collect_choice=False):
    """walt-parity backward pass: σ nodes sum children (bincount, child
    order = ascending (node, move)); hero nodes take the sign-max, hero
    children being a contiguous tail block sorted by parent.

    Returns (wave-1 node values, choice) where choice[j] = (hero node ids
    at wave j, chosen child ids at wave j+1) under first-max tie-breaking
    (moves ascending — walt parity).
    """
    L = len(waves) - 1
    choice: dict[int, tuple] = {}
    for j in range(L - 1, 0, -1):
        par = waves[j + 1]["parent"]
        hj = waves[j]["hero"]
        M = len(hj)
        par_is_h = hj[par]
        # astype guards the all-hero wave: bincount of an empty selection
        # returns int64 zeros, and int64 acc would truncate the sign-max
        acc = np.bincount(
            par[~par_is_h], weights=vals[~par_is_h], minlength=M,
        ).astype(np.float64, copy=False)
        seli = np.flatnonzero(par_is_h)
        if len(seli):
            pm = par[seli]
            sv = sign * vals[seli]
            seg = np.concatenate(([0], np.flatnonzero(np.diff(pm)) + 1))
            mx = np.maximum.reduceat(sv, seg)
            acc[pm[seg]] = sign * mx
            if collect_choice:
                R = len(seli)
                seglen = np.diff(np.concatenate((seg, [R])))
                segid = np.repeat(np.arange(len(seg)), seglen)
                posr = np.where(sv == mx[segid], np.arange(R), R)
                first = np.minimum.reduceat(posr, seg)
                choice[j] = (pm[seg], seli[first])
        vals = acc
    return vals, choice


def paths_of(waves, j, nodes) -> np.ndarray:
    """(len(nodes), j) tile paths from the root to the given wave-j nodes,
    by parent-pointer gathers (walt's block-rebuild walk shape)."""
    nodes = np.asarray(nodes, dtype=np.int64)
    out = np.empty((len(nodes), j), dtype=np.int64)
    anc = nodes
    for k in range(j, 0, -1):
        out[:, k - 1] = waves[k]["tile"][anc]
        anc = waves[k]["parent"][anc]
    return out


def extract_strategy(waves, choice, best_move, hero_is_me) -> dict:
    """Hero's BR strategy: one move per hero info set reachable WHILE HERO
    FOLLOWS IT (profile seats free to play anything the walk contains).
    Keys are public tile paths from the subgame root; hero=me's root entry
    is keyed ()."""
    L = len(waves) - 1
    strategy: dict[tuple, int] = {}
    t1 = waves[1]["tile"]
    if hero_is_me:
        strategy[()] = int(best_move)
        reach = t1 == best_move
    else:
        reach = np.ones(len(t1), dtype=bool)
    for j in range(1, L):
        hj = waves[j]["hero"]
        hnodes = np.flatnonzero(hj & reach)
        if len(hnodes) and j in choice:
            pm, ch = choice[j]
            tiles = waves[j + 1]["tile"]
            # hnodes ⊆ pm (every hero node has a choice); map via searchsorted
            mv_at = tiles[ch[np.searchsorted(pm, hnodes)]].astype(np.int64)
            paths = paths_of(waves, j, hnodes).tolist()
            for row, mv in zip(paths, mv_at.tolist()):
                strategy[tuple(row)] = mv
        # propagate reach to wave j+1
        par = waves[j + 1]["parent"]
        keep = reach[par]
        if hj.any() and j in choice:
            pm, ch = choice[j]
            chosen_mask = np.zeros(len(par), dtype=bool)
            chosen_mask[ch] = True
            keep &= ~hj[par] | chosen_mask
        reach = keep
    return strategy


def expand_full_width(sub: Subgame, *, keep_slots=True, need_path_ids=True,
                      slot_budget=DEFAULT_SLOT_BUDGET,
                      kernels=False) -> dict:
    """The CFR lane's structural walk: ALL FOUR seats full-width, worlds
    surviving a public action iff the acting hidden seat holds the tile
    (hero=-1: no seat is best-responding; weights flow un-scaled). Returns
    the raw run_engine result with per-wave slot partitions and 128-bit
    node path ids kept. Memory is the caller's affair — budget-guarded.
    kernels=True selects the numba move-emission fill (buildkernel.py,
    output-identical; the numpy provider stays the pinned mirror)."""
    from hoyt.profiles import _FullWidthProvider

    return run_engine(sub, _FullWidthProvider(kernels=kernels), hero=-1,
                      keep_slots=keep_slots, need_path_ids=need_path_ids,
                      slot_budget=slot_budget)
