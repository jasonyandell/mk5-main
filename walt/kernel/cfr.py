"""walt/kernel/cfr.py — CFR+ over a walt endgame subgame, priced by exact BR.

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

INJECTABILITY
-------------
cfr_solve touches the subgame ONLY through `.root/.worlds/.weights` (it
builds its own public tree + info-set index from those via walt.tables), and
touches the injected `impl` module ONLY through:

    impl.br_solve(subgame, profile, payoff43, hero=seat)  # gap pricing
    impl.StochasticProfile()                              # export format
    profile.set(seat, hand_mask, node, moves, probs)      # one call per iset

where node = tuple of domino ids played since the root (the profile-domain
convention in walt/kernel/reference.py; the kernel lane hashes it
internally). The orchestrator runs the SAME cfr_solve on the fast kernel
via:

    import walt.kernel as K
    sub = K.build_subgame(root, worlds, weights)
    res = cfr_solve(sub, payoff43, iters=500, target_gap=0.05, impl=K)

`impl` defaults to walt.kernel (the fast lane); tests inject
walt.kernel.reference. Zero torch anywhere in this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from walt.kernel.reference import (  # shared public-state algebra + protocol
    _pub_step,
    build_subgame as _view_subgame,
    legal_tiles,
    sign_of,
)
from walt.tables import get_luts

__all__ = ["CFRResult", "cfr_solve"]


@dataclass
class CFRResult:
    profile: object                    # impl.StochasticProfile (average strategy)
    trace: list                        # [(iteration, gap)] as measured
    gap: float                         # final measured gap (max single-seat BR gain)
    value: float                       # self-play value of the average profile
    iters_run: int
    debug: dict | None = field(default=None, repr=False)


# --------------------------------------------------------------------------- #
#  Public tree + info-set index                                                #
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
#  Traversals                                                                  #
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
#  cfr_solve                                                                   #
# --------------------------------------------------------------------------- #

def cfr_solve(subgame, payoff43, iters: int = 200, target_gap: float | None = None,
              seed: int = 0, br_every: int = 10, impl=None,
              pinned: dict | None = None, debug: bool = False) -> CFRResult:
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
            walt.kernel (the fast lane); tests pass walt.kernel.reference.
        pinned: optional {seat: profile}; those seats play the fixed profile,
            take no regret updates, and are excluded from the gap max
            (the 2-player-izable toy harness).
        debug: attach internals (tree, regrets, average, iset table).
    """
    del seed  # deterministic full-width solve; kept for contract stability
    if impl is None:
        import walt.kernel as impl  # the fast kernel lane
        if not hasattr(impl, "br_solve"):
            raise ImportError(
                "walt.kernel does not export br_solve yet; pass "
                "impl=walt.kernel.reference (toys) or the kernel module")
    payoff43 = np.asarray(payoff43, dtype=np.float64).reshape(-1)
    if payoff43.shape[0] != 43:
        raise ValueError("payoff43 must have 43 entries")
    pinned = {int(s): p for s, p in (pinned or {}).items()}
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

    # flat strategy state
    reg = np.zeros(t.L)
    avg = np.zeros(t.L)
    sig = 1.0 / t.nlegal_slot                     # uniform start
    for i in range(n_isets):
        if t.i_pinned[i]:
            sig[t.off[i]:t.off[i] + len(t.i_moves[i])] = t.i_pinned_probs[i]
    live_seats = [u for u in range(4) if u not in pinned]
    upd = {u: (slot_seat == u) & ~slot_pinned for u in live_seats}

    def avg_sig():
        out = sig.copy()                          # pinned slots stay fixed
        tot = np.zeros(n_isets)
        np.add.at(tot, t.slot_iset, avg)
        m = ~slot_pinned
        has = tot[t.slot_iset] > 0
        out[m & has] = avg[m & has] / tot[t.slot_iset][m & has]
        out[m & ~has] = 1.0 / t.nlegal_slot[m & ~has]
        return out

    def export(asig):
        prof = impl.StochasticProfile()
        for i in range(n_isets):
            lo = t.off[i]
            moves = t.i_moves[i]
            prof.set(t.i_seat[i], t.i_hand[i], t.i_node[i],
                     moves, asig[lo:lo + len(moves)])
        return prof

    def measure(asig):
        prof = export(asig)
        vbar = float(weights @ _values(t, payoff43, asig)) / total_w
        gap = 0.0
        for u in live_seats:
            bru = impl.br_solve(subgame, prof, payoff43, hero=u).value
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
            m = upd[u]
            cfv = np.zeros(n_isets)
            np.add.at(cfv, t.slot_iset, sig * cf)
            avg[m] += it * xI[t.slot_iset[m]] * sig[m]       # linear averaging
            reg[m] = np.maximum(reg[m] + signs[u] * (cf[m] - cfv[t.slot_iset[m]]),
                                0.0)                          # regret-matching+
            tot = np.zeros(n_isets)
            np.add.at(tot, t.slot_iset, np.where(m, reg, 0.0))
            has = tot[t.slot_iset[m]] > 0
            new = np.where(has, reg[m] / np.where(has, tot[t.slot_iset[m]], 1.0),
                           1.0 / t.nlegal_slot[m])
            sig[m] = new
        if it % br_every == 0 or it == iters:
            asig = avg_sig()
            prof, vbar, gap = measure(asig)
            trace.append((it, gap))
            result = (prof, vbar, gap)
            if target_gap is not None and gap <= target_gap:
                break

    prof, vbar, gap = result
    dbg = None
    if debug:
        dbg = {"tree": t, "reg": reg, "avg": avg, "avg_sig": avg_sig(),
               "sig": sig, "n_isets": n_isets}
    return CFRResult(profile=prof, trace=trace, gap=gap, value=vbar,
                     iters_run=it, debug=dbg)
