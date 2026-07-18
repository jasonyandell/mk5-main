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

Two interchangeable iteration engines (identical math, parity-gated in
walt/tests/test_cfr_parity.py):

- engine="wave" (default): the static public tree + info-set index is
  reconstructed from `hoyt.expand_full_width`'s SoA waves (parent
  slots matched by sorted (node, world) keys; per-edge strategy slots by
  masked-popcount move ranks against per-info-set legality). Iterations are
  pure per-wave gathers + bincounts — zero python per-node work.
- engine="loop": the original recursive python traversal, kept verbatim as
  the verified correctness mirror for parity tests. Toy sizes only.

Exported profiles skip FORCED decisions (single legal move): both BR
implementations play forced moves without consulting the profile, so
entries there are dead weight (~85% of info sets at H4 scale).

INJECTABILITY
-------------
cfr_solve touches the subgame ONLY through `.root/.worlds/.weights`, and
touches the injected `impl` module ONLY through:

    impl.br_solve(subgame, profile, payoff43, hero=seat,
                  want_strategy=False)                    # gap pricing
    impl.StochasticProfile()                              # export format
    profile.set(seat, hand_mask, node, moves, probs)      # one call per iset

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
                want_debug: bool, bulk_export: bool = False) -> _WaveStruct:
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
                            need_path_ids=bulk_export, **kw)
    waves = res["waves"]
    L = len(waves) - 1
    N = int(ksub.worlds.shape[0])
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
    ws.leaf_pts = leaf["ptsvd"][leaf["snode"]]
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
        snode, sworld, sw = wj["snode"], wj["sworld"], wj["sw"]
        parn, tile_n, pseat_n = wn["parent"], wn["tile"], wn["pseat"]
        Mj = len(wj["counts"])

        # actor per wave-j node = seat of its first child edge
        firstchild = np.searchsorted(parn, np.arange(Mj))
        actor_n = pseat_n[firstchild]

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

        # child edges -> parent slots, by (node, world) key (worlds are
        # unique within a node and node-major slot order is preserved)
        snode_c, sworld_c = wn["snode"], wn["sworld"]
        TC = tile_n[snode_c]
        seat_e = pseat_n[snode_c]
        kp = snode.astype(np.int64) * N + sworld
        kc = parn[snode_c].astype(np.int64) * N + sworld_c
        if kp.size > 1 and not (kp[1:] > kp[:-1]).all():
            sp = np.argsort(kp, kind="stable")
            PS = sp[np.searchsorted(kp[sp], kc)]
        else:
            PS = np.searchsorted(kp, kc)
        if not np.array_equal(kp[PS], kc):
            raise AssertionError(
                f"wave {j}: parent-slot reconstruction failed (kernel wave "
                "ordering changed?)")
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
                ec["cnt"].append(nleg[selE])
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

    waves[L]["sw"] = waves[L]["snode"] = waves[L]["sworld"] = None
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
#  cfr_solve                                                                   #
# =========================================================================== #

def cfr_solve(subgame, payoff43, iters: int = 200, target_gap: float | None = None,
              seed: int = 0, br_every: int = 10, impl=None,
              pinned: dict | None = None, debug: bool = False,
              engine: str = "wave", slot_budget: int | None = None) -> CFRResult:
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
        engine: "wave" (vectorized over the kernel's SoA walk, default) or
            "loop" (the original recursive mirror, toy sizes only).
        slot_budget: forwarded to the kernel walk (wave engine); None keeps
            the kernel default. A KernelMemoryError means chunk or cap —
            escalate, never sample.
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
    if engine not in ("wave", "loop"):
        raise ValueError(f"engine must be 'wave' or 'loop', got {engine!r}")
    if engine == "wave":
        return _solve_wave(subgame, payoff43, iters, target_gap, br_every,
                           impl, pinned, debug, slot_budget)
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
                pinned, debug, slot_budget) -> CFRResult:
    tm = {"build": 0.0, "iterate": 0.0, "export": 0.0, "br": 0.0,
          "value": 0.0}
    t0 = time.time()
    ws = _build_wave(subgame.root, subgame.worlds, subgame.weights, pinned,
                     slot_budget, debug,
                     bulk_export=hasattr(impl.StochasticProfile, "set_bulk"))
    tm["build"] = time.time() - t0

    bid_team = int(subgame.root.bidder) % 2
    signs = {u: sign_of(u, bid_team) for u in range(4)}
    payleaf = payoff43[ws.leaf_pts]

    reg = np.zeros(ws.Lflat)
    avg = np.zeros(ws.Lflat)
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
    slot_seat = ws.i_seat_arr[ws.slot_iset]
    live_seats = [u for u in range(4) if u not in pinned]
    upd = {u: (slot_seat == u) & ~slot_pinned for u in live_seats}

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

    def measure(asig):
        prof = export(asig)
        t1 = time.time()
        v0 = _wave_values(ws, asig, payleaf)
        vbar = float(ws.w0 @ v0) / ws.total_w
        tm["value"] += time.time() - t1
        t1 = time.time()
        gap = 0.0
        for u in live_seats:
            bru = impl.br_solve(subgame, prof, payoff43, hero=u,
                                want_strategy=False).value
            gap = max(gap, signs[u] * (bru - vbar))
        tm["br"] += time.time() - t1
        return prof, vbar, gap

    trace: list = []
    result = None
    it = 0
    cf = np.empty(ws.Lflat)
    xI = np.empty(ws.n_isets)
    for it in range(1, iters + 1):
        t1 = time.time()
        for u in live_seats:
            cf[:] = 0.0
            xI[:] = 0.0
            _wave_pass(ws, sig, u, payleaf, cf, xI)
            _rm_plus_update(sig, reg, avg, cf, xI, upd[u], ws.slot_iset,
                            ws.nlegal_slot, ws.n_isets, signs[u], it)
        tm["iterate"] += time.time() - t1
        if it % br_every == 0 or it == iters:
            asig = _avg_sig(sig, avg, ws.slot_iset, ws.nlegal_slot,
                            ws.n_isets, ~slot_pinned)
            prof, vbar, gap = measure(asig)
            trace.append((it, gap))
            result = (prof, vbar, gap)
            if target_gap is not None and gap <= target_gap:
                break

    prof, vbar, gap = result
    dbg = None
    if debug:
        asig = _avg_sig(sig, avg, ws.slot_iset, ws.nlegal_slot, ws.n_isets,
                        ~slot_pinned)
        isets = SimpleNamespace(
            off=ws.off, i_moves=ws.dbg["i_moves"], i_node=ws.dbg["i_node"],
            i_seat=ws.dbg["i_seat"], i_hand=ws.dbg["i_hand"])
        dbg = {"isets": isets, "reg": reg, "avg": avg, "avg_sig": asig,
               "sig": sig, "n_isets": ws.n_isets}
    return CFRResult(profile=prof, trace=trace, gap=gap, value=vbar,
                     iters_run=it, timings=tm, debug=dbg)


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
