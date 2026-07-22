"""hoyt/cfr.py — CFR+ over a walt endgame subgame, priced by exact BR.

WHAT THIS CONVERGES TO, AND WHAT IT DOES NOT CLAIM
--------------------------------------------------
Texas 42 endgames are TWO-TEAM zero-sum games with PRIVATE hands. cfr_solve
runs self-play regret minimization (CFR+: regret-matching+, alternating
updates, linear averaging) over the AGENT-FORM information sets of all four
seats: each seat is its own regret minimizer whose payoff is its team's sign
on the shared declaring-points payoff table. Partners do not share
information, cannot correlate beyond what public play reveals, and are
updated as independent agents.

The Nash-convergence theorems behind CFR are 2-PLAYER zero-sum theorems.
They do not apply here: with hidden information inside a team, the
partnership seam makes even the right equilibrium concept murky, and
self-play regret minimization in team games can cycle or settle anywhere
low-regret. We therefore claim NOTHING about equilibrium. The deliverable is
a **low-exploitability reference profile priced by exact best response**:
the returned trace is (iteration, gap) where gap = max over seats of the
exact single-seat BR gain against the current average profile — v1
exploitability, single-seat deviation with the partner staying ON profile
(team-pair deviation is explicitly out of scope). The measured gap IS the
claim, full stop. On 2-player-izable toys (two seats pinned to a known
profile) the 2p theorems do apply, and the verification battery checks
convergence to enumerated/LP game values there.

MECHANICS
---------
The subgame's chance node is the given world distribution; a world reaches a
public node iff every hidden seat's played tiles were legal from its
remaining hand there (a strategy that respects legality gives such worlds
probability zero, so pruning them is exact). Terminal payoff is public —
payoff43[final declaring points] — so ALL world-dependence lives in reach
probabilities. Regret updates use counterfactual reach (chance x all other
seats, per world); an info set's own-reach is constant across its worlds
(its seat's past probabilities are pinned by the public path), which the
implementation exploits.

Four interchangeable iteration engines (the three fp64 lanes are
parity-gated in hoyt/tests/test_cfr_parity.py; Metal is accuracy-calibrated):

- engine="metal": custom Metal kernels over the fused segment layout. CFR+
  iterate state stays on the GPU between explicit gap-audit boundaries and
  uses float32 arithmetic. The CPU fp64 lane is its calibration oracle, not a
  bit-replication target.

- engine="fused" (default): numba edge kernels (hoyt/iterkernel.py) over
  the wave engine's structure, with native int32 indices, forced-edge
  skips, and a FORCED-SLOT-COMPRESSED strategy space (#82). Forced info
  sets (single legal move, ~83% at H4 scale) are provably inert in RM+:
  their sigma is exactly 1.0 forever and their regret update is exactly
  0.0 (cf - cfv == 0 on a single-slot iset), so reg/avg/cf/xI live only on
  non-forced slots, grouped per seat so each seat's update is a contiguous
  slice. fp64 results are BITWISE identical to engine="wave".
- engine="wave": the numpy vectorized engine over the static public tree +
  info-set index reconstructed from `hoyt.expand_full_width`'s SoA waves
  (parent slots matched by sorted (node, world) keys; per-edge strategy
  slots by masked-popcount move ranks against per-info-set legality).
  Iterations are pure per-wave gathers + bincounts — zero python per-node
  work. Kept as the pinned pure-numpy mirror for the fused lane.
- engine="loop": the original recursive python traversal, kept verbatim as
  the verified correctness mirror for parity tests. Toy sizes only.

The fused lane is fp64 only: an fp32 dtype knob was built and measured
dead on the anchor (P7 refuted — gap drift 2.6e-3 over the 1e-3 license
bar AND zero throughput win; the fused loops are gather-latency-bound,
not float-bandwidth-bound). Receipts: wiki perf-log 18l.

Exported profiles skip FORCED decisions (single legal move): both BR
implementations play forced moves without consulting the profile, so
entries there are dead weight (~85% of info sets at H4 scale).

INJECTABILITY
-------------
cfr_solve touches the subgame ONLY through `.root/.worlds/.weights`, and
touches the injected `impl` module ONLY through:

    impl.br_solve(subgame, profile, payoff43, hero=seat,
                  want_strategy=False)     # gap pricing, engine="loop" only
    impl.StochasticProfile()                              # export format
    profile.set(seat, hand_mask, node, moves, probs)      # one call per iset

The wave/fused engines price gaps IN-STRUCT (`_wave_br`: exact single-seat
BR as a forward-reach + backward-argmax pass over the resident wave
structure — no export, no re-walk; perf-log 18m) and export the profile
once at the stop. impl.br_solve remains the standing pricing oracle via
the loop engine and the P4 cross-gate (1e-9).

where node = tuple of domino ids played since the root (the profile-domain
convention in hoyt/reference.py; the kernel lane hashes it
internally). The orchestrator runs the SAME cfr_solve on the fast kernel
via:

    import hoyt as K
    sub = K.build_subgame(root, worlds, weights)
    res = cfr_solve(sub, payoff43, iters=500, target_gap=0.05, impl=K)

`impl` defaults to hoyt (the fast lane); tests inject
hoyt.reference. The wave engine's STRUCTURE always comes from the
kernel's wave walk (net-free, walt.tables rules only); `impl` only prices
gaps and hosts the export format. Zero torch anywhere in this module.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from hoyt.reference import (  # shared public-state algebra + protocol
    _pub_step,
    build_subgame as _view_subgame,
    legal_tiles,
    sign_of,
)
from walt.tables import get_luts

__all__ = ["CFRResult", "cfr_solve"]

_AR28 = np.arange(28)
_ALL28 = np.int64((1 << 28) - 1)


@dataclass
class CFRResult:
    profile: object                    # impl.StochasticProfile (average strategy)
    trace: list                        # [(iteration, gap)] as measured
    gap: float                         # final measured gap (max single-seat BR gain)
    value: float                       # self-play value of the average profile
    iters_run: int
    capped: bool = False               # stop was wall_budget_s-driven (gap still exact)
    timings: dict | None = None        # build/iterate/export/br wall seconds
    debug: dict | None = field(default=None, repr=False)


def _bits_tuple(mask: int) -> tuple:
    out = []
    m = int(mask)
    while m:
        b = m & -m
        out.append(b.bit_length() - 1)
        m ^= b
    return tuple(out)


# =========================================================================== #
#  engine="loop" — the original recursive traversal (verified mirror)          #
# =========================================================================== #

class _Tree:
    """Flat public tree. Per node: actor, alive-world bookkeeping, children
    as (move, child_id, sel, gather) where sel indexes the node's local world
    positions and gather = flat strategy-slot index per selected world.
    Per info set: seat, hand, node key, legal moves, flat slot range."""

    def __init__(self):
        # nodes
        self.actor: list = []
        self.key: list = []          # tuple of tiles since root
        self.terminal: list = []
        self.decl_pts: list = []     # valid at terminals
        self.n_alive: list = []
        self.wiset: list = []        # (A,) global iset id per local world, or None
        self.children: list = []     # list of (move, child_id, sel, iset_ids, slots)
        self.pub: list = []          # _Pub per node (for V3 legality audits)
        # info sets
        self.i_seat: list = []
        self.i_hand: list = []
        self.i_node: list = []
        self.i_moves: list = []
        self.i_pinned: list = []
        self.i_pinned_probs: list = []
        # flat layout (filled by _finalize)
        self.off = self.slot_iset = self.nlegal_slot = None
        self.L = 0

    def _finalize(self):
        lens = np.array([len(m) for m in self.i_moves], dtype=np.int64)
        self.off = np.concatenate(([0], np.cumsum(lens)))[:-1]
        self.L = int(lens.sum())
        self.slot_iset = np.repeat(np.arange(len(lens)), lens)
        self.nlegal_slot = lens[self.slot_iset].astype(np.float64)
        for nid in range(len(self.actor)):
            self.children[nid] = [
                (m, cid, sel, self.off[iset_ids] + slots)
                for (m, cid, sel, iset_ids, slots) in self.children[nid]
            ]


def _build_tree(sub, pinned: dict) -> _Tree:
    luts = get_luts(sub.decl_id)
    t = _Tree()

    def new_node(p, pub, key, wcur, my_mask):
        nid = len(t.actor)
        t.actor.append(-1)
        t.key.append(key)
        t.terminal.append(p == 28)
        t.decl_pts.append(pub.decl_pts)
        t.n_alive.append(wcur.shape[0])
        t.wiset.append(None)
        t.children.append([])
        t.pub.append(pub)
        if p == 28:
            return nid

        actor = (pub.leader + pub.n_in_trick) % 4
        t.actor[nid] = actor
        A = wcur.shape[0]
        if actor == sub.me:
            hands = np.full(A, my_mask, dtype=np.int64)
        else:
            hands = wcur[:, sub.col_of[actor]]

        # info sets at this node: one per distinct hand, ascending. Pinned
        # seats expand only their support: zero-probability branches carry
        # zero counterfactual reach for every live seat and br_solve never
        # enters them, so pruning is exact.
        wiset = np.empty(A, dtype=np.int64)
        expand_of: dict = {}
        legal_of: dict = {}
        for h in np.unique(hands):
            h = int(h)
            iid = len(t.i_seat)
            legal = legal_tiles(h, pub.led_tile, sub.decl_id)
            legal_of[h] = legal
            t.i_seat.append(actor)
            t.i_hand.append(h)
            t.i_node.append(key)
            t.i_moves.append(legal)
            if actor in pinned:
                moves, probs = pinned[actor].dist(actor, h, key)
                if not set(moves) <= set(legal):
                    raise ValueError(
                        f"pinned profile illegal at seat={actor} hand={h:#x}")
                full = np.zeros(len(legal))
                for m, pr in zip(moves, probs):
                    full[legal.index(m)] = pr
                t.i_pinned.append(True)
                t.i_pinned_probs.append(full)
                expand_of[h] = tuple(m for m, pr in zip(moves, probs) if pr > 0)
            else:
                t.i_pinned.append(False)
                t.i_pinned_probs.append(None)
                expand_of[h] = legal
            wiset[hands == h] = iid
        t.wiset[nid] = wiset

        union = sorted(set().union(*expand_of.values()))
        for m in union:
            ok = np.array([m in expand_of[int(h)] for h in hands])
            sel = np.flatnonzero(ok)
            if len(sel) == 0:
                continue
            w2 = wcur[sel]
            my2 = my_mask
            if actor == sub.me:
                my2 = my_mask & ~(1 << m)
            else:
                w2 = w2.copy()
                w2[:, sub.col_of[actor]] &= ~(1 << m)
            iset_ids = wiset[sel]
            slots = np.array([legal_of[int(h)].index(m) for h in hands[sel]],
                             dtype=np.int64)
            cid = new_node(p + 1, _pub_step(pub, actor, m, luts, sub.bid_team),
                           key + (m,), w2, my2)
            t.children[nid].append((m, cid, sel, iset_ids, slots))
        return nid

    new_node(sub.p0, sub.pub0, (), sub.worlds.astype(np.int64), sub.my_mask0)
    t._finalize()
    return t


def _values(t: _Tree, payoff43, sig) -> np.ndarray:
    """Per-world expected payoff at the root under strategy slots `sig`."""
    def walk(nid):
        if t.terminal[nid]:
            return np.full(t.n_alive[nid], payoff43[t.decl_pts[nid]])
        v = np.zeros(t.n_alive[nid])
        for (_m, cid, sel, gid) in t.children[nid]:
            v[sel] += sig[gid] * walk(cid)
        return v
    return walk(0)


def _update_pass(t: _Tree, weights, payoff43, sig, u, cf, xI):
    """One full-width traversal accumulating seat u's counterfactual values
    (into cf, flat-slot indexed) and own-reach per info set (into xI).
    r_mu = chance x all-other-seat reach per world; r_u = u's own reach."""
    def walk(nid, r_mu, r_u):
        if t.terminal[nid]:
            return np.full(t.n_alive[nid], payoff43[t.decl_pts[nid]])
        s = t.actor[nid]
        v = np.zeros(t.n_alive[nid])
        for (_m, cid, sel, gid) in t.children[nid]:
            pr = sig[gid]
            if s == u:
                vc = walk(cid, r_mu[sel], r_u[sel] * pr)
            else:
                vc = walk(cid, r_mu[sel] * pr, r_u[sel])
            v[sel] += pr * vc
            if s == u:
                np.add.at(cf, gid, r_mu[sel] * vc)
        if s == u:
            iu, first = np.unique(t.wiset[nid], return_index=True)
            xI[iu] = r_u[first]
        return v

    walk(0, weights.astype(np.float64).copy(), np.ones(t.n_alive[0]))


# =========================================================================== #
#  engine="wave" — static SoA structure over the kernel's full-width walk      #
# =========================================================================== #

class _WaveStruct:
    """Everything the vectorized iteration needs, built once per subgame.

    Per wave transition j (wave j -> j+1), aligned with wave j+1's slots:
      PS[j]    parent slot index into wave j
      GID[j]   flat strategy-slot index (iset offset + ascending-move rank)
      uedge[j] {seat: index array of edges played by that seat}
    Per info set (global, wave-major): seat/hand/nlegal/off + slot_iset /
    nlegal_slot flat layouts (identical semantics to _Tree's)."""

    __slots__ = ("L", "S", "w0", "total_w", "leaf_pts", "PS", "GID", "uedge",
                 "iset_ju", "n_isets", "Lflat", "off", "slot_iset",
                 "nlegal_slot", "i_seat_arr", "woff", "wlen", "pin_slots",
                 "exp_entries", "exp_cols", "dbg")


def _build_wave(root, worlds, weights, pinned: dict, slot_budget,
                want_debug: bool, bulk_export: bool = False,
                kernels: bool = False) -> _WaveStruct:
    from hoyt.subgame import (
        build_subgame as _kernel_subgame,
        expand_full_width,
        paths_of,
    )
    # bulk export keys entries by the walk's 128-bit rolling path hash
    # (no python path tuples anywhere); pinned/debug keep the path route.
    bulk_export = bulk_export and not pinned and not want_debug
    ksub = _kernel_subgame(root, worlds, weights)
    kw = {} if slot_budget is None else {"slot_budget": slot_budget}
    res = expand_full_width(ksub, keep_slots=True,
                            need_path_ids=bulk_export, kernels=kernels,
                            **kw)
    waves = res["waves"]
    L = len(waves) - 1
    me = ksub.me
    col_arr = ksub.col_arr
    LED, CFB = ksub.LED, ksub.CFB
    p0 = ksub.p0

    ws = _WaveStruct()
    ws.L = L
    ws.S = [len(waves[j]["snode"]) for j in range(L + 1)]
    ws.w0 = ksub.weights[waves[0]["sworld"]].astype(np.float64)
    ws.total_w = float(ksub.weights.sum())
    leaf = res["leaf"]
    ws.leaf_pts = leaf["ptsvd"][leaf["snode"]].astype(np.int16)
    ws.PS, ws.GID, ws.uedge = [], [], []
    ws.iset_ju = {}
    ws.pin_slots = []          # (flat off, probs vector) fixed at pinned isets
    ws.exp_entries = []        # (seat, hand, path, moves, off, n) — nlegal >= 2
    ws.exp_cols = None         # columnar export (bulk_export), set in finalize
    ec = {"seat": [], "hand": [], "h1": [], "h2": [], "off": [], "cnt": [],
          "mv": []} if bulk_export else None
    ws.woff, ws.wlen = [], []
    ws.dbg = {"i_moves": [], "i_node": [], "i_seat": [], "i_hand": []} \
        if want_debug else None

    i_seat_parts, nleg_parts = [], []
    n_isets = 0
    Lflat = 0

    # per-node replayed state (small: node-count arrays, not slot arrays)
    mymask_n = np.array([ksub.my0], dtype=np.int64)
    led_n = np.array([ksub.led0], dtype=np.int64)

    # With pinned seats, the full-width walk contains subtrees below pinned
    # ZERO-probability edges. Pinned profiles need not cover info sets there
    # (they are unreachable while pinned seats follow the profile: zero
    # counterfactual reach in CFR, and BR walks drop zero-prob moves), so we
    # propagate slot reachability and only query pinned .dist where reached.
    reach = np.ones(ws.S[0], dtype=bool) if pinned else None

    for j in range(L):
        wj, wn = waves[j], waves[j + 1]
        snode, sw = wj["snode"], wj["sw"]
        parn = wn["parent"]                       # int32: index use only
        tile_n = wn["tile"].astype(np.int64)      # int8 stored; arithmetic
        pseat_n = wn["pseat"].astype(np.int64)    # needs the wide dtype
        Mj = len(wj["counts"])

        # actor per wave-j node: resident from the walk (int8 stored)
        actor_n = wj["actor"].astype(np.int64)

        # acting hand per slot (me's hand is node-public, replayed)
        acts = actor_n[snode]
        cols = col_arr[acts]
        hand = np.where(acts == me, mymask_n[snode],
                        sw[np.arange(ws.S[j]), np.where(cols < 0, 0, cols)])

        # info sets: unique (node, hand), ascending — kernel provider order
        key = (snode.astype(np.int64) << 28) | hand
        uq, fidx, inv = np.unique(key, return_index=True, return_inverse=True)
        nI = len(uq)
        u_node = (uq >> 28).astype(np.int64)
        u_hand = (uq & _ALL28).astype(np.int64)
        u_seat = acts[fidx]

        # legality per iset via LUTs (independent of the walk's edges)
        ls = led_n[u_node]
        fb = np.where(ls >= 0, CFB[ls], _ALL28)
        lm = u_hand & fb
        lm = np.where(lm != 0, lm, u_hand)
        nleg = np.bitwise_count(lm.astype(np.uint64)).astype(np.int64)
        off = Lflat + np.concatenate(([0], np.cumsum(nleg)))[:-1]

        # child edges -> parent slots: resident from the walk (rows/gidx,
        # perf-log 18n; formerly re-derived by (node, world) key search —
        # 2 s of searchsorted on the anchor)
        snode_c = wn["snode"]
        TC = tile_n[snode_c]
        seat_e = pseat_n[snode_c]
        PS = wn["pslot"].astype(np.int64)
        # cross-check: the walk's edge fan-out must equal LUT legality
        cnt_edges = np.bincount(PS, minlength=ws.S[j])
        if not np.array_equal(cnt_edges[fidx], nleg):
            raise AssertionError(f"wave {j}: edge count != legality count")

        # flat strategy-slot per edge: iset offset + ascending-move rank
        iset_e = inv[PS]
        below = (np.int64(1) << TC) - np.int64(1)
        rank = np.bitwise_count(
            (lm[iset_e] & below).astype(np.uint64)).astype(np.int64)
        gid = off[iset_e] + rank

        ws.PS.append(PS)
        ws.GID.append(gid)
        ws.uedge.append({u: np.flatnonzero(seat_e == u) for u in range(4)})
        ws.woff.append(Lflat)
        ws.wlen.append(int(nleg.sum()))
        for u in np.unique(u_seat):
            m = u_seat == u
            ws.iset_ju[(j, int(u))] = (n_isets + np.flatnonzero(m), fidx[m])
        i_seat_parts.append(u_seat.astype(np.int64))
        nleg_parts.append(nleg)

        # exportable isets (nlegal >= 2): columnar when bulk_export (hash
        # keys straight off the walk); else paths + move tuples (pinned /
        # debug / reference-impl export)
        if bulk_export:
            selE = np.flatnonzero(nleg >= 2)
            if len(selE):
                ec["seat"].append(u_seat[selE].astype(np.int8))
                ec["hand"].append(u_hand[selE])
                ec["h1"].append(wj["ph1"][u_node[selE]])
                ec["h2"].append(wj["ph2"][u_node[selE]])
                ec["off"].append(off[selE])
                cntE = nleg[selE]
                ec["cnt"].append(cntE)
                if kernels:
                    from hoyt.buildkernel import fw_fill
                    offE = np.concatenate(([0], np.cumsum(cntE)))
                    siE = np.empty(offE[-1], dtype=np.int64)
                    mvE = np.empty(offE[-1], dtype=np.int64)
                    fw_fill(lm[selE], offE, siE, mvE)
                    ec["mv"].append(mvE.astype(np.int8))
                else:
                    bmE = ((lm[selE, None] >> _AR28) & 1).astype(bool)
                    ec["mv"].append(np.nonzero(bmE)[1].astype(np.int8))
            wj["ph1"] = wj["ph2"] = None
        iset_reached = None
        if pinned:
            iset_reached = np.zeros(nI, dtype=bool)
            np.logical_or.at(iset_reached, inv, reach)
        need = nleg >= 2 if not bulk_export else np.zeros(nI, dtype=bool)
        pin_sel = np.zeros(nI, dtype=bool)
        if pinned:
            pin_sel = np.isin(u_seat, list(pinned)) & (nleg >= 2) \
                & iset_reached
            need = (need & iset_reached) | pin_sel
        if want_debug:
            need = np.ones(nI, dtype=bool)
        sel = np.flatnonzero(need)
        pin_prob_edge = None
        if len(sel):
            paths = [tuple(r) for r in paths_of(waves, j, u_node[sel]).tolist()]
            bm = ((lm[sel, None] >> _AR28) & 1).astype(bool)
            ii, tt = np.nonzero(bm)
            cut = np.searchsorted(ii, np.arange(1, len(sel)))
            moves_l = [tuple(a.tolist()) for a in np.split(tt, cut)]
            path_of = dict(zip(sel.tolist(), paths))
            moves_of = dict(zip(sel.tolist(), moves_l))
            if pinned and pin_sel.any():
                pin_prob_edge = np.ones(int(nleg.sum()))
                loc_off = off - Lflat
            for k in sel.tolist():
                if nleg[k] >= 2 and (iset_reached is None or iset_reached[k]):
                    ws.exp_entries.append(
                        (int(u_seat[k]), int(u_hand[k]), path_of[k],
                         moves_of[k], int(off[k]), int(nleg[k])))
                if pin_sel[k]:
                    seat_k = int(u_seat[k])
                    mv, pr = pinned[seat_k].dist(seat_k, int(u_hand[k]),
                                                 path_of[k])
                    full = np.zeros(int(nleg[k]))
                    lk = moves_of[k]
                    for m2, p2 in zip(mv, pr):
                        full[lk.index(int(m2))] = p2
                    ws.pin_slots.append((int(off[k]), full))
                    pin_prob_edge[loc_off[k]:loc_off[k] + int(nleg[k])] = full
            if want_debug:
                ws.dbg["i_moves"].extend(moves_of[k] for k in range(nI))
                ws.dbg["i_node"].extend(path_of[k] for k in range(nI))
                ws.dbg["i_seat"].extend(int(s) for s in u_seat)
                ws.dbg["i_hand"].extend(int(h) for h in u_hand)

        if pinned:
            # child reach: parent reached AND the edge is not a pinned
            # zero-probability move (support pruning, loop-engine parity)
            blocked = np.zeros(len(PS), dtype=bool)
            if pin_prob_edge is not None:
                pe = np.isin(seat_e, list(pinned))
                blocked[pe] = pin_prob_edge[(gid - Lflat)[pe]] == 0.0
            reach = reach[PS] & ~blocked

        n_isets += nI
        Lflat += int(nleg.sum())

        # advance replayed node state to wave j+1
        pos = (p0 + j) & 3
        if pos == 0:
            led_n = LED[tile_n]
        elif pos < 3:
            led_n = led_n[parn]
        else:
            led_n = np.full(len(parn), -1, dtype=np.int64)
        mm = mymask_n[parn]
        mymask_n = np.where(pseat_n == me,
                            mm & ~(np.int64(1) << tile_n), mm)

        # free the slot-level memory hogs as soon as the transition is done
        wj["sw"] = wj["snode"] = wj["sworld"] = None
        wj["pslot"] = wj["actor"] = None

    waves[L]["sw"] = waves[L]["snode"] = waves[L]["sworld"] = None
    waves[L]["pslot"] = None
    ws.n_isets = n_isets
    ws.Lflat = Lflat
    if bulk_export:
        cnt = np.concatenate(ec["cnt"]) if ec["cnt"] else \
            np.empty(0, dtype=np.int64)
        off_all = np.concatenate(ec["off"]) if ec["off"] else \
            np.empty(0, dtype=np.int64)
        total = int(cnt.sum())
        ramp = np.arange(total, dtype=np.int64) - np.repeat(
            np.concatenate(([0], np.cumsum(cnt)))[:-1], cnt)
        cat = lambda k, d: np.concatenate(ec[k]) if ec[k] else \
            np.empty(0, dtype=d)  # noqa: E731
        ws.exp_cols = {
            "seat": cat("seat", np.int8),
            "hand": cat("hand", np.int64),
            "h1": cat("h1", np.uint64),
            "h2": cat("h2", np.uint64),
            "moff": np.concatenate(([0], np.cumsum(cnt))),
            "mv": cat("mv", np.int8),
            "slot_idx": np.repeat(off_all, cnt) + ramp,
        }
    nleg_all = np.concatenate(nleg_parts) if nleg_parts else \
        np.empty(0, dtype=np.int64)
    ws.off = np.concatenate(([0], np.cumsum(nleg_all)))[:-1]
    ws.slot_iset = np.repeat(np.arange(n_isets), nleg_all)
    ws.nlegal_slot = nleg_all[ws.slot_iset].astype(np.float64)
    ws.i_seat_arr = np.concatenate(i_seat_parts) if i_seat_parts else \
        np.empty(0, dtype=np.int64)
    return ws


def _wave_values(ws: _WaveStruct, sig, payleaf) -> np.ndarray:
    """Backward-only pass: per-world expected payoff at the root."""
    v = payleaf
    for j in range(ws.L - 1, -1, -1):
        v = np.bincount(ws.PS[j], weights=sig[ws.GID[j]] * v,
                        minlength=ws.S[j])
    return v


def _wave_br(ws: _WaveStruct, asig, u, payleaf, sign_u, rmu_store) -> float:
    """Exact single-seat best response of seat u vs the profile `asig`,
    computed IN-STRUCT (perf-log 18m, Fable consult L1): the resident wave
    structure already contains everything exact BR needs, so gap pricing
    stops re-walking the subgame through impl.br_solve per measurement
    (which paid the stochastic-profile tree blowup, ~13 s/measurement on
    the anchor, 65-71% of solve wall).

    Forward: counterfactual reach r_mu (chance x all seats except u; u's
    edges at prob 1). Backward: at u's info sets, score each strategy slot
    by bincount of r_mu * child-value over the iset's worlds and pick the
    per-iset argmax of the SIGNED score (u maximizes its team's
    orientation); the chosen move's per-world child values propagate
    unweighted. Elsewhere sigma-weighted expectation. Ties pick the lowest
    move — value-identical by definition of a tie. Zero-reach isets choose
    arbitrarily and contribute zero (the same exactness argument as pinned
    support pruning). Cross-checked against impl.br_solve at 1e-9 (gate P4
    in test_cfr_parity; the loop engine still prices through br_solve as
    the standing oracle)."""
    L = ws.L
    r_mu = ws.w0
    for j in range(L):
        pr = asig[ws.GID[j]]
        ue = ws.uedge[j][u]
        rmu_store[j] = r_mu
        pm = pr
        if len(ue):
            pm = pr.copy()
            pm[ue] = 1.0
        r_mu = r_mu[ws.PS[j]] * pm
    v = payleaf
    for j in range(L - 1, -1, -1):
        pr = asig[ws.GID[j]]
        ue = ws.uedge[j][u]
        w = pr * v
        if len(ue):
            gid_l = ws.GID[j][ue] - ws.woff[j]
            sc = np.bincount(gid_l, weights=rmu_store[j][ws.PS[j][ue]] * v[ue],
                             minlength=ws.wlen[j])
            si = ws.slot_iset[ws.woff[j]:ws.woff[j] + ws.wlen[j]]
            imin = si[0]
            ssc = sign_u * sc
            gmax = np.full(int(si[-1]) - int(imin) + 1, -np.inf)
            np.maximum.at(gmax, si - imin, ssc)
            chosen = ssc == gmax[si - imin]
            idxs = np.flatnonzero(chosen)
            first = idxs[np.unique((si - imin)[idxs], return_index=True)[1]]
            sel = np.zeros(ws.wlen[j], dtype=bool)
            sel[first] = True
            w[ue] = np.where(sel[gid_l], v[ue], 0.0)
        v = np.bincount(ws.PS[j], weights=w, minlength=ws.S[j])
    return float(ws.w0 @ v) / ws.total_w


def _wave_pass(ws: _WaveStruct, sig, u, payleaf, cf, xI) -> None:
    """One update traversal for seat u: forward reaches, backward values,
    counterfactual accumulation into cf (flat slots) and own-reach into xI."""
    L = ws.L
    r_mu_store = [None] * L
    r_mu = ws.w0
    r_u = np.ones(ws.S[0])
    for j in range(L):
        ju = ws.iset_ju.get((j, u))
        if ju is not None:
            ids, reps = ju
            xI[ids] = r_u[reps]
        r_mu_store[j] = r_mu
        pr = sig[ws.GID[j]]
        ue = ws.uedge[j][u]
        pm = pr.copy()
        pm[ue] = 1.0
        r_mu = r_mu[ws.PS[j]] * pm
        pu = np.ones(len(pr))
        pu[ue] = pr[ue]
        r_u = r_u[ws.PS[j]] * pu
    v = payleaf
    for j in range(L - 1, -1, -1):
        pr = sig[ws.GID[j]]
        ue = ws.uedge[j][u]
        if len(ue):
            idx = ws.GID[j][ue] - ws.woff[j]
            contrib = r_mu_store[j][ws.PS[j][ue]] * v[ue]
            cf[ws.woff[j]:ws.woff[j] + ws.wlen[j]] += np.bincount(
                idx, weights=contrib, minlength=ws.wlen[j])
        v = np.bincount(ws.PS[j], weights=pr * v, minlength=ws.S[j])


# =========================================================================== #
#  engine="fused" — numba edge kernels + forced-slot-compressed updates (#82)  #
# =========================================================================== #

class _FusedLayout:
    """Per-solve iterate layout for the fused engine, derived from a
    _WaveStruct. Compressed slot space = non-forced slots only (nlegal >= 2),
    stable-sorted by seat so each seat's slots and info sets are contiguous
    slices; within a seat, slot and iset order match the flat wave-major
    order, so per-iset accumulations happen in the same order as the wave
    engine's bincounts (bitwise parity). Per wave: int32 parent slots,
    int32 compressed strategy ids (-1 = forced edge), int8 edge seats."""

    __slots__ = ("n_c", "n_ci", "c_flat", "c_soff", "c_isoff",
                 "c_isl_loc", "inv_nleg", "sig0_full",
                 "ps32", "cgid32", "eseat8", "c_iju", "w0",
                 "rmu", "ru2", "v2", "par", "vseg", "cf_ju", "iso")


def _build_fused(ws: _WaveStruct, sig0_full, par: bool = False,
                 cpu_buffers: bool = True, warm_jit: bool = True) -> _FusedLayout:
    fl = _FusedLayout()
    fl.par = par
    nleg_iset = np.diff(np.append(ws.off, ws.Lflat))
    slot_seat = ws.i_seat_arr[ws.slot_iset]
    flat_idx = np.flatnonzero(nleg_iset[ws.slot_iset] >= 2)
    order = np.argsort(slot_seat[flat_idx], kind="stable")
    fl.c_flat = flat_idx[order]
    c_seat = slot_seat[fl.c_flat]
    fl.n_c = len(fl.c_flat)
    fl.c_soff = np.searchsorted(c_seat, np.arange(5))
    cmap = np.full(ws.Lflat, -1, dtype=np.int32)
    cmap[fl.c_flat] = np.arange(fl.n_c, dtype=np.int32)

    # compressed isets: contiguous slot runs, ascending within each seat
    c_iset_g = ws.slot_iset[fl.c_flat]
    starts = np.empty(fl.n_c, dtype=bool)
    if fl.n_c:
        starts[0] = True
        starts[1:] = c_iset_g[1:] != c_iset_g[:-1]
    c_isl = np.cumsum(starts) - 1              # global compressed iset id
    fl.n_ci = int(c_isl[-1]) + 1 if fl.n_c else 0
    fl.c_isoff = np.empty(5, dtype=np.int64)
    fl.c_isoff[4] = fl.n_ci
    for u in range(3, -1, -1):
        lo = fl.c_soff[u]
        fl.c_isoff[u] = c_isl[lo] if lo < fl.c_soff[u + 1] \
            else fl.c_isoff[u + 1]
    fl.c_isl_loc = c_isl - fl.c_isoff[c_seat]  # per-seat-local iset ids

    # same ops as the wave engine's update (1.0 / nlegal), precomputed
    fl.inv_nleg = 1.0 / ws.nlegal_slot[fl.c_flat]
    fl.sig0_full = sig0_full                   # fp64; forced 1.0, pins set

    imap = np.full(ws.n_isets, -1, dtype=np.int64)
    if fl.n_c:
        imap[c_iset_g[starts]] = np.arange(fl.n_ci)
    fl.c_iju = {}
    for (j, u), (ids, reps) in ws.iset_ju.items():
        ci = imap[ids]
        keep = ci >= 0
        if keep.any():
            fl.c_iju[(j, u)] = (ci[keep], reps[keep])

    fl.ps32 = [p.astype(np.int32) for p in ws.PS]
    fl.cgid32 = [cmap[g] for g in ws.GID]
    fl.eseat8 = []
    for j in range(ws.L):
        e = np.empty(len(ws.PS[j]), dtype=np.int8)
        for u, ue in ws.uedge[j].items():
            e[ue] = u
        fl.eseat8.append(e)

    fl.w0 = ws.w0
    if cpu_buffers:
        fl.rmu = [np.empty(s) for s in ws.S]
        max_s = max(ws.S)
        fl.ru2 = (np.empty(max_s), np.empty(max_s))
        fl.v2 = (np.empty(max_s), np.empty(max_s))
    else:
        fl.rmu = fl.ru2 = fl.v2 = None

    if par:
        # P13 threading structure: every parallel fold's grouping is built
        # HERE, once, from the walk's arrays (stable argsorts) — the kernels
        # then own disjoint output ranges with ascending-edge order inside
        # each group, which makes them bitwise vs the sequential lane
        # regardless of thread count or scheduling (see iterkernel.py).
        fl.vseg = []
        for j in range(ws.L):
            ps = fl.ps32[j]
            cnt = np.bincount(ps, minlength=ws.S[j])
            if len(ps) == ws.S[j] and cnt.max() == 1:
                fl.vseg.append(None)       # bijection wave: pure-map path
            else:
                poff = np.zeros(ws.S[j] + 1, dtype=np.int64)
                np.cumsum(cnt, out=poff[1:])
                perm = np.argsort(ps, kind="stable").astype(np.int32)
                fl.vseg.append((poff, perm))
        fl.cf_ju = {}
        for j in range(ws.L):
            e8, cg = fl.eseat8[j], fl.cgid32[j]
            for u in range(4):
                sel = np.flatnonzero((e8 == u) & (cg >= 0))
                if not len(sel):
                    continue
                g = cg[sel]
                o = np.argsort(g, kind="stable")
                gs = g[o]
                st = np.empty(len(gs), dtype=bool)
                st[0] = True
                st[1:] = gs[1:] != gs[:-1]
                goff = np.append(np.flatnonzero(st),
                                 len(gs)).astype(np.int64)
                fl.cf_ju[(j, u)] = (goff, gs[st], sel[o].astype(np.int32))
        fl.iso = {}
        for u in range(4):
            sl = slice(fl.c_soff[u], fl.c_soff[u + 1])
            niu = int(fl.c_isoff[u + 1] - fl.c_isoff[u])
            fl.iso[u] = np.searchsorted(
                fl.c_isl_loc[sl], np.arange(niu + 1)).astype(np.int64)

    if warm_jit:
        # warm on 1-edge dummies so compile/cache-load lands in build, not
        # iteration. The Metal lane has separate JIT kernels and skips this.
        from hoyt.iterkernel import bwd_edges, fwd_edges
        z32, f32 = np.zeros(1, np.int32), np.full(1, -1, np.int32)
        z8 = np.zeros(1, np.int8)
        d = np.ones(1)
        fwd_edges(z32, f32, z8, 0, d, d.copy(), d.copy(),
                  np.empty(1), np.empty(1))
        bwd_edges(z32, f32, z8, 0, d, d.copy(), d.copy(),
                  np.empty(1), d.copy())
        if par:
            from hoyt.iterkernel import (bwd_v_map, bwd_v_seg, cf_seg,
                                         fwd_edges_par, rm_update_seg)
            z64 = np.zeros(2, np.int64)
            fwd_edges_par(z32, f32, z8, 0, d, d.copy(), d.copy(),
                          np.empty(1), np.empty(1))
            bwd_v_map(z32, f32, d, d.copy(), np.empty(1))
            bwd_v_seg(z64, z32, f32, d, d.copy(), np.empty(1))
            cf_seg(z64, z32, z32, z32, d, d.copy(), np.empty(1))
            rm_update_seg(z64, d.copy(), d.copy(), d.copy(), d.copy(),
                          np.empty(0), d.copy(), 1, 1)
    return fl


def _fused_pass(ws: _WaveStruct, fl: _FusedLayout, sig_c, u, payleaf,
                cf_c, xI_c) -> None:
    """One update traversal for seat u — the fused twin of _wave_pass."""
    from hoyt.iterkernel import (bwd_edges, bwd_v_map, bwd_v_seg, cf_seg,
                                 fwd_edges, fwd_edges_par)
    L = ws.L
    fl.rmu[0][:] = fl.w0
    ru = fl.ru2[0][:ws.S[0]]
    ru[:] = 1.0
    fwd = fwd_edges_par if fl.par else fwd_edges
    for j in range(L):
        cj = fl.c_iju.get((j, u))
        if cj is not None:
            cids, reps = cj
            xI_c[cids] = ru[reps]
        ru_n = fl.ru2[(j + 1) & 1][:ws.S[j + 1]]
        fwd(fl.ps32[j], fl.cgid32[j], fl.eseat8[j], u, sig_c,
            fl.rmu[j], ru, fl.rmu[j + 1], ru_n)
        ru = ru_n
    v = fl.v2[L & 1][:ws.S[L]]
    v[:] = payleaf
    for j in range(L - 1, -1, -1):
        v_p = fl.v2[j & 1][:ws.S[j]]
        if fl.par:
            cfj = fl.cf_ju.get((j, u))
            if cfj is not None:
                goff, gids, perm = cfj
                cf_seg(goff, gids, perm, fl.ps32[j], fl.rmu[j], v, cf_c)
            seg = fl.vseg[j]
            if seg is None:
                bwd_v_map(fl.ps32[j], fl.cgid32[j], sig_c, v, v_p)
            else:
                poff, vperm = seg
                bwd_v_seg(poff, vperm, fl.cgid32[j], sig_c, v, v_p)
        else:
            bwd_edges(fl.ps32[j], fl.cgid32[j], fl.eseat8[j], u, sig_c,
                      v, fl.rmu[j], v_p, cf_c)
        v = v_p


def _rm_plus_update_c(fl: _FusedLayout, u, sig_c, reg_c, avg_c, cf_c, xI_c,
                      sign_u, it) -> None:
    """The CFR+ update block on seat u's compressed contiguous slice —
    identical math (and, in fp64, identical bits) to _rm_plus_update
    restricted to seat u's non-forced slots; forced slots are provably
    inert there (reg == 0, sigma == 1 exactly), so nothing is lost."""
    sl = slice(fl.c_soff[u], fl.c_soff[u + 1])
    if fl.par:
        from hoyt.iterkernel import rm_update_seg
        rm_update_seg(fl.iso[u], sig_c[sl], reg_c[sl], avg_c[sl], cf_c[sl],
                      xI_c[fl.c_isoff[u]:fl.c_isoff[u + 1]],
                      fl.inv_nleg[sl], sign_u, it)
        return
    isl = fl.c_isl_loc[sl]
    niu = int(fl.c_isoff[u + 1] - fl.c_isoff[u])
    s = sig_c[sl]
    c = cf_c[sl]
    xIl = xI_c[fl.c_isoff[u]:fl.c_isoff[u + 1]]
    cfv = np.bincount(isl, weights=s * c, minlength=niu)
    avg_c[sl] += it * xIl[isl] * s
    reg = np.maximum(reg_c[sl] + sign_u * (c - cfv[isl]), 0.0)
    reg_c[sl] = reg
    tot = np.bincount(isl, weights=reg, minlength=niu)
    has = tot[isl] > 0
    sig_c[sl] = np.where(has, reg / np.where(has, tot[isl], 1.0),
                         fl.inv_nleg[sl])


def _avg_sig_fused(fl: _FusedLayout, avg_c, live_seats) -> np.ndarray:
    """Full-slot-space fp64 normalized average — bitwise what _avg_sig
    returns: forced slots 1.0, pinned slots their fixed probs (both live in
    sig0_full), live seats' non-forced slots avg/tot (uniform where the
    iset was never reached)."""
    out = fl.sig0_full.copy()
    for u in live_seats:
        sl = slice(fl.c_soff[u], fl.c_soff[u + 1])
        isl = fl.c_isl_loc[sl]
        niu = int(fl.c_isoff[u + 1] - fl.c_isoff[u])
        a = avg_c[sl]
        tot = np.bincount(isl, weights=a, minlength=niu)
        has = tot[isl] > 0
        out[fl.c_flat[sl]] = np.where(
            has, a / np.where(has, tot[isl], 1.0), fl.inv_nleg[sl])
    return out


# =========================================================================== #
#  cfr_solve                                                                   #
# =========================================================================== #

def cfr_solve(subgame, payoff43, iters: int = 200, target_gap: float | None = None,
              seed: int = 0, br_every: int = 10, impl=None,
              pinned: dict | None = None, debug: bool = False,
              engine: str = "fused", slot_budget: int | None = None,
              wall_budget_s: float | None = None,
              gap_exit: bool = False,
              threads: int | None = None) -> CFRResult:
    """CFR+ over all four seats' info sets; gap priced by impl.br_solve.

    Args:
        subgame: impl.build_subgame output; must expose .root/.worlds/.weights.
        payoff43: float64[43] leaf table on final declaring points.
        iters: iteration budget. One iteration = one alternating round
            (seats 0..3 each get a traversal + regret update, in seat order).
        target_gap: stop once the measured gap (max single-seat exact-BR gain
            vs the average profile, in the deviator's orientation) is <= this.
        seed: accepted for contract stability; the solve is fully
            deterministic (full-width traversals, no sampling), so it is unused.
        br_every: measure the gap (4 exact BR solves) every this many
            iterations; the final iteration is always measured.
        impl: module providing br_solve + StochasticProfile. Defaults to
            hoyt (the fast lane); tests pass hoyt.reference.
        pinned: optional {seat: profile}; those seats play the fixed profile,
            take no regret updates, and are excluded from the gap max
            (the 2-player-izable toy harness).
        debug: attach internals (regrets, average, iset table).
        engine: "fused" (numba edge kernels + forced-slot-compressed
            updates, default), "wave" (the pure-numpy mirror; fp64 results
            are bitwise identical to fused), or "loop" (the original
            recursive mirror, toy sizes only). "metal" runs float32 CFR+
            updates as custom Metal kernels and crosses to the host only at
            requested gap-audit boundaries; its contract is calibrated error,
            not fp64 bit replication.
        slot_budget: forwarded to the kernel walk (wave engine); None keeps
            the kernel default. A KernelMemoryError means chunk or cap —
            escalate, never sample.
        wall_budget_s: optional wall-clock budget for the WHOLE solve
            (build included), wave engine only. Checked at every iteration
            boundary and again after every gap measurement; when exceeded,
            the gap is measured once more and the solve stops with
            capped=True. The stop is a QUANTIFIED verdict — the average
            profile plus its exactly-priced single-seat BR gap — not a
            failure. A stopping measurement that also satisfies target_gap
            reports capped=False (convergence wins; when the budget expires
            exactly at the final iteration capped=True is still reported —
            the result is identical either way, the flag only names the
            binding constraint). Wave-only by design: the loop engine is
            the frozen parity mirror and its gates compare traces across
            engines; a wall-driven stop is timing-nondeterministic and can
            never be parity-pinned, so the loop raises ValueError instead
            of silently diverging (it is toy-sized only, where wall budgets
            are meaningless anyway). None (default) is behaviorally
            identical to the pre-knob solver.
        gap_exit: intermediate gap measurements price seats in descending
            last-known-gap order and stop at the first seat whose BR gain
            exceeds target_gap (perf-log 18o). Exact by construction: the
            continue/stop decision only needs "gap > target", convergence
            is only declared after all live seats are priced, and any
            measurement that can end the solve without convergence (iters
            exhausted, wall budget) prices all seats — so the stop
            iteration, final profile, value, and gap are IDENTICAL to
            gap_exit=False; only intermediate trace entries change (they
            record the certified-above-target partial max). Deterministic.
            wave/fused only: partial intermediate traces can't be
            parity-pinned against the loop mirror, so the loop raises.
        threads: run the fused iterate kernels on this many numba threads
            (perf-log P13). Bitwise identical to threads=None by
            construction — every parallel fold owns a disjoint output range
            with unchanged within-range accumulation order (the grouping is
            precomputed structure in _build_fused, never a runtime
            heuristic). Fused-only: the wave/loop mirrors are the pinned
            single-threaded references, so they raise.
    """
    del seed  # deterministic full-width solve; kept for contract stability
    if impl is None:
        import hoyt as impl  # the fast kernel lane
        if not hasattr(impl, "br_solve"):
            raise ImportError(
                "hoyt does not export br_solve yet; pass "
                "impl=hoyt.reference (toys) or the kernel module")
    payoff43 = np.asarray(payoff43, dtype=np.float64).reshape(-1)
    if payoff43.shape[0] != 43:
        raise ValueError("payoff43 must have 43 entries")
    pinned = {int(s): p for s, p in (pinned or {}).items()}
    if engine not in ("fused", "wave", "loop", "metal"):
        raise ValueError(
            f"engine must be 'fused', 'wave', 'loop' or 'metal', "
            f"got {engine!r}")
    if wall_budget_s is not None and engine == "loop":
        raise ValueError("wall_budget_s requires engine='fused' or 'wave' "
                         "(the loop engine is the frozen parity mirror; "
                         "wall-driven stops cannot be parity-pinned)")
    if gap_exit and engine == "loop":
        raise ValueError("gap_exit requires engine='fused' or 'wave' "
                         "(partial intermediate traces cannot be "
                         "parity-pinned against the loop mirror)")
    if threads is not None and engine != "fused":
        raise ValueError("threads requires engine='fused' (the wave and "
                         "loop mirrors and calibrated Metal lane do not "
                         "consume numba thread counts)")
    if engine in ("fused", "wave", "metal"):
        return _solve_wave(subgame, payoff43, iters, target_gap, br_every,
                           impl, pinned, debug, slot_budget, wall_budget_s,
                           fused=(engine == "fused"), gap_exit=gap_exit,
                           threads=threads, metal=(engine == "metal"))
    return _solve_loop(subgame, payoff43, iters, target_gap, br_every,
                       impl, pinned, debug)


def _rm_plus_update(sig, reg, avg, cf, xI, m, slot_iset, nlegal_slot,
                    n_isets, sign_u, it):
    """The shared CFR+ update block (identical math in both engines):
    linear-averaged strategy accumulation, regret-matching+ clamp, and
    strategy refresh for the updating seat's slots (mask m)."""
    cfv = np.zeros(n_isets)
    np.add.at(cfv, slot_iset, sig * cf)
    avg[m] += it * xI[slot_iset[m]] * sig[m]           # linear averaging
    reg[m] = np.maximum(reg[m] + sign_u * (cf[m] - cfv[slot_iset[m]]),
                        0.0)                            # regret-matching+
    tot = np.zeros(n_isets)
    np.add.at(tot, slot_iset, np.where(m, reg, 0.0))
    has = tot[slot_iset[m]] > 0
    sig[m] = np.where(has, reg[m] / np.where(has, tot[slot_iset[m]], 1.0),
                      1.0 / nlegal_slot[m])


def _avg_sig(sig, avg, slot_iset, nlegal_slot, n_isets, unpinned_mask):
    """Normalized average strategy; pinned slots keep their fixed sig."""
    out = sig.copy()
    tot = np.zeros(n_isets)
    np.add.at(tot, slot_iset, avg)
    has = tot[slot_iset] > 0
    m = unpinned_mask & has
    out[m] = avg[m] / tot[slot_iset][m]
    m0 = unpinned_mask & ~has
    out[m0] = 1.0 / nlegal_slot[m0]
    return out


def _solve_wave(subgame, payoff43, iters, target_gap, br_every, impl,
                pinned, debug, slot_budget, wall_budget_s=None,
                fused=False, gap_exit=False, threads=None,
                metal=False) -> CFRResult:
    tm = {"build": 0.0, "walk": 0.0, "layout": 0.0,
          "device_setup": 0.0, "iterate": 0.0, "export": 0.0,
          "br": 0.0, "value": 0.0}
    t0 = time.time()
    if fused and threads:
        # set before the build, not just before _build_fused: any numba
        # parallel kernel reached from _build_wave would otherwise run at
        # numba's default (all cores) and oversubscribe a multi-worker
        # sweep (perf-log 19d)
        import numba
        numba.set_num_threads(threads)
    ws = _build_wave(subgame.root, subgame.worlds, subgame.weights, pinned,
                     slot_budget, debug,
                     bulk_export=hasattr(impl.StochasticProfile, "set_bulk"),
                     kernels=fused or metal)
    tm["walk"] = time.time() - t0

    bid_team = int(subgame.root.bidder) % 2
    signs = {u: sign_of(u, bid_team) for u in range(4)}
    payleaf = payoff43[ws.leaf_pts]

    sig = 1.0 / ws.nlegal_slot
    slot_pinned = np.zeros(ws.Lflat, dtype=bool)
    for off, probs in ws.pin_slots:
        sig[off:off + len(probs)] = probs
        slot_pinned[off:off + len(probs)] = True
    # forced pinned isets (single legal) already have sig = 1.0; mark them
    if pinned:
        slot_seat = ws.i_seat_arr[ws.slot_iset]
        for s in pinned:
            slot_pinned |= (slot_seat == s)
    live_seats = [u for u in range(4) if u not in pinned]

    if fused or metal:
        tl = time.time()
        fl = _build_fused(ws, sig, par=bool(threads) or metal,
                          cpu_buffers=not metal, warm_jit=not metal)
        tm["layout"] = time.time() - tl
    if metal:
        from hoyt.metalkernel import MetalCFR
        td = time.time()
        metal_state = MetalCFR(ws, fl, payleaf, sig, live_seats)
        tm["device_setup"] = time.time() - td
    elif fused:
        sig_c = sig[fl.c_flat].copy()
        reg_c = np.zeros(fl.n_c)
        avg_c = np.zeros(fl.n_c)
        cf_c = np.zeros(fl.n_c)
        xI_c = np.zeros(fl.n_ci)
    else:
        reg = np.zeros(ws.Lflat)
        avg = np.zeros(ws.Lflat)
        slot_seat = ws.i_seat_arr[ws.slot_iset]
        upd = {u: (slot_seat == u) & ~slot_pinned for u in live_seats}
    tm["build"] = time.time() - t0

    def export(asig):
        t1 = time.time()
        prof = impl.StochasticProfile()
        if ws.exp_cols is not None:
            c = ws.exp_cols
            prof.set_bulk(c["seat"], c["hand"], c["h1"], c["h2"], c["moff"],
                          c["mv"], asig[c["slot_idx"]])
        else:
            set_ = prof.set
            for seat, hand, path, moves, off, n in ws.exp_entries:
                set_(seat, hand, path, moves, asig[off:off + n])
        tm["export"] += time.time() - t1
        return prof

    rmu_br = [None] * ws.L
    last_gap = {u: np.inf for u in live_seats}   # unpriced ⇒ price first

    def measure(asig, partial=False):
        # in-struct pricing (perf-log 18m / Fable consult L1): no export,
        # no impl.br_solve re-walk — the wave structure is already resident.
        # Cross-gated vs impl.br_solve at 1e-9 (P4); loop engine keeps the
        # br_solve path as the standing oracle.
        # partial (gap_exit, perf-log 18o): an INTERMEDIATE measurement only
        # decides continue-vs-stop, and gap > target ⇔ some seat's BR gain >
        # target — so price seats (largest last-known gap first) and exit at
        # the first crossing. Convergence is only ever declared after ALL
        # live seats priced, so a stopping gap is always the exact full max;
        # the exit cannot change the stop iteration or the final result.
        if metal:
            # Full device-side average value + single-seat BR audit. The
            # public profile crosses to the host once, after the stopping
            # iteration; intermediate certificates transfer only scalars.
            t1 = time.time()
            vbar, gap = metal_state.measure(signs)
            tm["br"] += time.time() - t1
            return vbar, gap
        t1 = time.time()
        v0 = _wave_values(ws, asig, payleaf)
        vbar = float(ws.w0 @ v0) / ws.total_w
        tm["value"] += time.time() - t1
        t1 = time.time()
        gap = 0.0
        for u in sorted(live_seats, key=lambda s: (-last_gap[s], s)):
            bru = _wave_br(ws, asig, u, payleaf, signs[u], rmu_br)
            g = signs[u] * (bru - vbar)
            last_gap[u] = g
            gap = max(gap, g)
            if partial and target_gap is not None and gap > target_gap:
                break
        tm["br"] += time.time() - t1
        return vbar, gap

    trace: list = []
    result = None
    it = 0
    capped = False
    over = (lambda: time.time() - t0 > wall_budget_s) \
        if wall_budget_s is not None else (lambda: False)
    if not fused and not metal:
        cf = np.empty(ws.Lflat)
        xI = np.empty(ws.n_isets)

    def averaged():
        if metal:
            return metal_state.average_numpy()
        if fused:
            return _avg_sig_fused(fl, avg_c, live_seats)
        return _avg_sig(sig, avg, ws.slot_iset, ws.nlegal_slot,
                        ws.n_isets, ~slot_pinned)

    for it in range(1, iters + 1):
        t1 = time.time()
        if metal:
            metal_state.round(signs, it)
        elif fused:
            for u in live_seats:
                cf_c[fl.c_soff[u]:fl.c_soff[u + 1]] = 0.0
                xI_c[fl.c_isoff[u]:fl.c_isoff[u + 1]] = 0.0
                _fused_pass(ws, fl, sig_c, u, payleaf, cf_c, xI_c)
                _rm_plus_update_c(fl, u, sig_c, reg_c, avg_c, cf_c, xI_c,
                                  signs[u], it)
        else:
            for u in live_seats:
                cf[:] = 0.0
                xI[:] = 0.0
                _wave_pass(ws, sig, u, payleaf, cf, xI)
                _rm_plus_update(sig, reg, avg, cf, xI, upd[u], ws.slot_iset,
                                ws.nlegal_slot, ws.n_isets, signs[u], it)
        tm["iterate"] += time.time() - t1
        budget_stop = over()
        if it % br_every == 0 or it == iters or budget_stop:
            asig = None if metal else averaged()
            # a measurement that can end the solve WITHOUT convergence
            # (iters exhausted / wall budget) must price all seats — a
            # capped row's final_gap is a full 4-seat verdict
            partial = gap_exit and it != iters and not budget_stop
            vbar, gap = measure(asig, partial)
            trace.append((it, gap))
            result = (asig, vbar, gap)
            if target_gap is not None and gap <= target_gap:
                break
            if over():                 # budget-driven stop, gap just priced
                if not partial:
                    capped = True
                    break
                # partial measurement can't cap: iterate on; the next
                # boundary check sees the expired budget and prices fully

    asig_f, vbar, gap = result
    if metal:
        asig_f = averaged()
    prof = export(asig_f)              # once, at the stop — not per measure
    dbg = None
    if debug:
        asig = averaged()
        isets = SimpleNamespace(
            off=ws.off, i_moves=ws.dbg["i_moves"], i_node=ws.dbg["i_node"],
            i_seat=ws.dbg["i_seat"], i_hand=ws.dbg["i_hand"])
        if metal:
            state = metal_state.debug_numpy()
            reg = np.zeros(ws.Lflat)
            reg[fl.c_flat] = state["reg_c"].astype(np.float64)
            avg = np.zeros(ws.Lflat)
            avg[fl.c_flat] = state["avg_c"].astype(np.float64)
            sig = fl.sig0_full.copy()
            sig[fl.c_flat] = state["sig_c"].astype(np.float64)
            dbg_extra = {"metal_peak_bytes": metal_state.peak_memory,
                         "metal_dtype": "float32"}
        elif fused:
            # scatter compressed state to full-slot space; forced slots
            # carry their exact invariants (reg 0.0, sig 1.0). avg differs
            # from the wave engine's on forced/unreached slots (dead weight
            # there); avg_sig is the parity-meaningful surface.
            reg = np.zeros(ws.Lflat)
            reg[fl.c_flat] = reg_c
            avg = np.zeros(ws.Lflat)
            avg[fl.c_flat] = avg_c
            sig = fl.sig0_full.copy()
            sig[fl.c_flat] = sig_c
            dbg_extra = {}
        else:
            dbg_extra = {}
        dbg = {"isets": isets, "reg": reg, "avg": avg, "avg_sig": asig,
               "sig": sig, "n_isets": ws.n_isets, **dbg_extra}
    return CFRResult(profile=prof, trace=trace, gap=gap, value=vbar,
                     iters_run=it, capped=capped, timings=tm, debug=dbg)


def _solve_loop(subgame, payoff43, iters, target_gap, br_every, impl,
                pinned, debug) -> CFRResult:
    # internal view derived ONLY from the documented .root/.worlds/.weights
    # surface; the caller's subgame object itself is passed to impl.br_solve.
    sub = _view_subgame(subgame.root, subgame.worlds, subgame.weights)
    bid_team = int(sub.root.bidder) % 2
    weights = np.asarray(sub.weights, dtype=np.float64).reshape(-1)
    total_w = float(weights.sum())

    t = _build_tree(sub, pinned)
    n_isets = len(t.i_seat)
    slot_seat = np.array(t.i_seat, dtype=np.int64)[t.slot_iset]
    slot_pinned = np.array(t.i_pinned, dtype=bool)[t.slot_iset]
    signs = {u: sign_of(u, bid_team) for u in range(4)}

    reg = np.zeros(t.L)
    avg = np.zeros(t.L)
    sig = 1.0 / t.nlegal_slot                     # uniform start
    for i in range(n_isets):
        if t.i_pinned[i]:
            sig[t.off[i]:t.off[i] + len(t.i_moves[i])] = t.i_pinned_probs[i]
    live_seats = [u for u in range(4) if u not in pinned]
    upd = {u: (slot_seat == u) & ~slot_pinned for u in live_seats}

    def export(asig):
        prof = impl.StochasticProfile()
        for i in range(n_isets):
            moves = t.i_moves[i]
            if len(moves) < 2:                    # forced: never consulted
                continue
            lo = t.off[i]
            prof.set(t.i_seat[i], t.i_hand[i], t.i_node[i],
                     moves, asig[lo:lo + len(moves)])
        return prof

    def measure(asig):
        prof = export(asig)
        vbar = float(weights @ _values(t, payoff43, asig)) / total_w
        gap = 0.0
        for u in live_seats:
            bru = impl.br_solve(subgame, prof, payoff43, hero=u,
                                want_strategy=False).value
            gap = max(gap, signs[u] * (bru - vbar))
        return prof, vbar, gap

    trace: list = []
    result = None
    it = 0
    for it in range(1, iters + 1):
        for u in live_seats:
            cf = np.zeros(t.L)
            xI = np.zeros(n_isets)
            _update_pass(t, weights, payoff43, sig, u, cf, xI)
            _rm_plus_update(sig, reg, avg, cf, xI, upd[u], t.slot_iset,
                            t.nlegal_slot, n_isets, signs[u], it)
        if it % br_every == 0 or it == iters:
            asig = _avg_sig(sig, avg, t.slot_iset, t.nlegal_slot, n_isets,
                            ~slot_pinned)
            prof, vbar, gap = measure(asig)
            trace.append((it, gap))
            result = (prof, vbar, gap)
            if target_gap is not None and gap <= target_gap:
                break

    prof, vbar, gap = result
    dbg = None
    if debug:
        asig = _avg_sig(sig, avg, t.slot_iset, t.nlegal_slot, n_isets,
                        ~slot_pinned)
        isets = SimpleNamespace(off=t.off, i_moves=t.i_moves,
                                i_node=t.i_node, i_seat=t.i_seat,
                                i_hand=t.i_hand)
        dbg = {"isets": isets, "tree": t, "reg": reg, "avg": avg,
               "avg_sig": asig, "sig": sig, "n_isets": n_isets}
    return CFRResult(profile=prof, trace=trace, gap=gap, value=vbar,
                     iters_run=it, debug=dbg)
