"""hoyt/equivcensus.py — the exact-equivalence census instrument.

The question (2026-07-20): do H4 endgame roots/worlds admit exact
equivalence classes ("every legal domino combination" need not be solved)?
The instrument: the RESIDUAL SIGNATURE — live tiles as a colored relational
structure carrying everything the game can ever read (per-lead legality
rows, trick-order comparisons as dense ranks, counts, holders, void
constraints, seat/payoff header) and nothing else. Any signature
isomorphism is a game isomorphism: suit-algebra-spec §5-§7 shows the play
phase touches tiles only via led_suit / can_follow / tau-comparisons /
count. See wiki [[endgame-equivalence-census]] for the measured verdicts.

Verdicts on the frozen evalset (hoyt/evalset_h4_v1.jsonl, 200 roots):

- CROSS-ROOT: 200/200 distinct canonical classes in every regime (strict,
  points-payoff, relaxed, no-void diagnostic). Compression 1.000x.
- WITHIN-ROOT WORLDS: zero hidden-moving automorphisms anywhere; world
  orbits 3,206,646 -> 3,206,646 (exactly 1.000x). Mechanism, not accident:
  in a 4-seat trick game any two comparable hidden tiles can co-occur in a
  trick (different holders in some world), so their relative rank is always
  game-live. Exact fungibility of hidden tiles is structurally impossible
  except for mutually-incomparable "dead trash" pairs (0 observed at H4).
- WITHIN-HAND: interchangeable my-tile pairs exist in 25/200 roots (29
  pairs incl. two S3 triples) under the relaxed signature (me-me trick
  comparisons dropped — two tiles of one hand never resolve a trick).
  Receipt: br_solve vs the uniform field ties every pair's root value
  BITWISE (29/29 at dv = 0.0; non-paired moves differ, median 1.54 pts).
- PIP RELABELING IS NOT A PER-DEAL ISOMORPHISM: the lead rule reads pip
  order (a mixed tile leads its max pip) and in-suit ranks are pip sums,
  so S7 transport (suit-algebra-spec §9) is distribution-level only:
  0/200 pip-relabeled roots matched their originals. The game re-reads
  its pips at every mixed lead.

Instrument certification: canonical keys invariant under index relabeling
(A1), planted symmetries found exactly (A2) — hoyt/tests/test_equivcensus.py.

Scope (honesty line): exactness claims are u-lane (physics worlds, uniform
weights). Sigma-filtered world sets (jud in the loop) and rng world caps
are not signature-respecting; nets read pips.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from walt.tables import get_luts
from walt.worlds import _void_forbidden, enumerate_worlds, seat_order

# --------------------------------------------------------------- signature


class Signature:
    """The residual signature of one trick-boundary root.

    relations: C[s, t] = can_follow(t, led_suit(s)); Q[s, t] = dense rank of
    tau(t) within lead-row s (sloughs tie at the bottom). colors0 carries
    (holder, count, per-seat void-forbidden bits). Headers pin the seat/
    payoff frame: regime "S" (any payoff43) includes bid_value and banked
    points; regime "P" (points payoff, arange(43)) drops them — banked
    declaring points are a pure value offset under a linear payoff.

    relax_meme=True drops my-tile-vs-my-tile rank comparisons (sound at
    value level: one play per seat per trick, so two tiles of one hand
    never resolve a trick against each other).
    """

    def __init__(self, root, use_voids: bool = True,
                 relax_meme: bool = False):
        assert len(root.current_trick) == 0, "census assumes trick boundary"
        luts = get_luts(root.decl_id)
        played = {d for (_s, d) in root.play_history}
        mine = set(int(t) for t in root.my_hand)
        self.live = sorted(set(range(28)) - played)
        self.n = len(self.live)
        self.hidden = sorted(set(self.live) - mine)

        made = {s: 0 for s in range(4)}
        for (s, _d) in root.play_history:
            made[s] += 1

        forb = _void_forbidden(root)
        rel_of = {s: (s - root.me) % 4 for s in seat_order(root)}
        forb_rel = {rel_of[s]: int(m) for s, m in forb.items()}

        cols = []
        for t in self.live:
            holder = 0 if t in mine else 1
            fb = tuple(((forb_rel[r] >> t) & 1) if (use_voids and holder == 1)
                       else 0 for r in (1, 2, 3))
            cols.append((holder, int(luts.count[t]), fb))
        self.colors0 = cols

        n = self.n
        C = np.zeros((n, n), dtype=np.int8)
        Q = np.zeros((n, n), dtype=np.int8)
        hid_idx = [i for i, t in enumerate(self.live) if t not in mine]
        for i, s in enumerate(self.live):
            ls = int(luts.led_suit[s])
            taus = [int(luts.rank[ls, t]) for t in self.live]
            C[i] = [int(luts.can_follow[ls, t]) for t in self.live]
            if not relax_meme:
                uniq = {v: k for k, v in enumerate(sorted(set(taus)))}
                Q[i] = [uniq[v] for v in taus]
            else:
                # hidden tiles: dense rank among hidden (even slots); my
                # tiles: position vs the hidden scale only (odd = strictly
                # between; even = tied with that hidden value)
                hid_taus = sorted(set(taus[j] for j in hid_idx))
                rank_h = {v: 2 * k + 2 for k, v in enumerate(hid_taus)}
                for j, t in enumerate(self.live):
                    v = taus[j]
                    if t not in mine or v in rank_h:
                        Q[i, j] = rank_h[v]
                    else:
                        below = sum(1 for hv in hid_taus if hv < v)
                        Q[i, j] = 2 * below + 1
        self.C, self.Q = C, Q

        me_declares = int(root.me % 2 == root.bidder % 2)
        rel_leader = (root.trick_leader - root.me) % 4
        bid_team = root.bidder % 2
        self.header_S = ("S", me_declares, rel_leader, int(root.bid_value),
                         int(root.team_points[bid_team]),
                         int(root.team_points[1 - bid_team]))
        self.header_P = ("P", me_declares, rel_leader)


# ------------------------------------------------ canonicalization + Aut


def _refine(colors, C, Q):
    """Iterated color refinement; stable integer colors renumbered by
    sorted signature, hence root-independent."""
    n = len(colors)
    colors = list(colors)
    prev_classes = -1
    while True:
        sigs = []
        for t in range(n):
            neigh = sorted(
                (colors[u], int(C[t, u]), int(Q[t, u]),
                 int(C[u, t]), int(Q[u, t]))
                for u in range(n) if u != t)
            sigs.append((colors[t], tuple(neigh)))
        order = {s: i for i, s in enumerate(sorted(set(sigs)))}
        colors = [order[s] for s in sigs]
        if len(order) == prev_classes:
            return colors
        prev_classes = len(order)


def _encode(perm, colors0, C, Q, header):
    idx = np.array(perm)
    return (header, tuple(colors0[p] for p in perm),
            C[np.ix_(idx, idx)].tobytes(), Q[np.ix_(idx, idx)].tobytes())


def canonical_key(sig, header):
    """Min encoding over all signature isomorphisms (individualization-
    refinement backtracking). Complete: equal keys iff isomorphic."""
    best = [None]

    def rec(colors):
        colors = _refine(colors, sig.C, sig.Q)
        classes: dict[int, list] = {}
        for i, c in enumerate(colors):
            classes.setdefault(c, []).append(i)
        multi = sorted(k for k, v in classes.items() if len(v) > 1)
        if not multi:
            perm = sorted(range(sig.n), key=lambda i: colors[i])
            enc = _encode(perm, sig.colors0, sig.C, sig.Q, header)
            if best[0] is None or enc < best[0]:
                best[0] = enc
            return
        for t in classes[multi[0]]:
            rec([(c, 0 if i == t else 1) for i, c in enumerate(colors)])

    rec([(c,) for c in sig.colors0])
    return best[0]


def automorphisms(sig):
    """All tile bijections preserving colors0, C, Q (seats fixed; holder
    color keeps my hand setwise-fixed). Returns perms over range(sig.n)."""
    colors = _refine([(c,) for c in sig.colors0], sig.C, sig.Q)
    classes: dict[int, list] = {}
    for i, c in enumerate(colors):
        classes.setdefault(c, []).append(i)
    order = sorted(range(sig.n), key=lambda i: (len(classes[colors[i]]), i))
    C, Q = sig.C, sig.Q
    auts = []

    def bt(k, phi):
        if k == sig.n:
            auts.append(tuple(phi[i] for i in range(sig.n)))
            return
        s = order[k]
        for t in classes[colors[s]]:
            if t in phi.values():
                continue
            ok = all(C[s, a] == C[t, b] and C[a, s] == C[b, t]
                     and Q[s, a] == Q[t, b] and Q[a, s] == Q[b, t]
                     for a, b in phi.items()) \
                and C[s, s] == C[t, t] and Q[s, s] == Q[t, t]
            if ok:
                phi[s] = t
                bt(k + 1, phi)
                del phi[s]

    bt(0, {})
    return auts


def interchangeable_pairs(root) -> set[tuple[int, int]]:
    """Provably value-tied my-tile pairs (relaxed-signature orbits)."""
    sig = Signature(root, relax_meme=True)
    pairs = set()
    for g in automorphisms(sig):
        for i in range(sig.n):
            if g[i] > i:
                pairs.add((sig.live[i], sig.live[g[i]]))
    return pairs


# ------------------------------------------------------------ world orbits


def world_orbit_count(root, sig, auts) -> tuple[int, int]:
    """(n_worlds_u, n_orbits) under the automorphism action on hidden
    tiles (seats fixed; worlds are walt/worlds.py (N, 3) masks)."""
    worlds = enumerate_worlds(root)
    N = len(worlds)
    if len(auts) <= 1:
        return N, N
    perms = []
    for g in auts:
        p = np.arange(28, dtype=np.uint32)
        for i, t in enumerate(sig.live):
            p[t] = sig.live[g[i]]
        perms.append(p)

    def apply(masks, p):
        out = np.zeros_like(masks)
        for b in range(28):
            out |= (((masks >> np.uint32(b)) & np.uint32(1))
                    << np.uint32(int(p[b])))
        return out

    keys = None
    for p in perms:
        m = apply(worlds, p)
        hi = (m[:, 0].astype(np.uint64) << np.uint64(32)) | m[:, 1].astype(np.uint64)
        kk = np.stack([hi, m[:, 2].astype(np.uint64)], axis=1)
        if keys is None:
            keys = kk
        else:
            lt = (kk[:, 0] < keys[:, 0]) | (
                (kk[:, 0] == keys[:, 0]) & (kk[:, 1] < keys[:, 1]))
            keys[lt] = kk[lt]
    return N, len(np.unique(keys, axis=0))


# ------------------------------------------------------------------ receipt


def value_tie_receipt(root, cap: int = 32):
    """Route the interchangeability claim through the solver: hero-BR vs
    the uniform field (symmetric by construction) must tie each pair's
    root value. Returns [(a, b, |dv|)]. Any world subset is valid — the
    automorphism is identity on hidden tiles."""
    import hoyt as K
    from hoyt.br import payoff_points
    from hoyt.profiles import StochasticProfile
    from walt.grade import _world_cap_rng

    pairs = interchangeable_pairs(root)
    if not pairs:
        return []
    worlds = enumerate_worlds(root)
    if len(worlds) > cap:
        idx = _world_cap_rng(root, cap).choice(len(worlds), size=cap,
                                               replace=False)
        worlds = worlds[np.sort(idx)]
    wt = np.full(len(worlds), 1.0 / len(worlds))
    sub = K.build_subgame(root, worlds, wt)
    br = K.br_solve(sub, StochasticProfile(uniform_fallback=True),
                    payoff_points())
    return [(a, b, abs(br.root_values[a] - br.root_values[b]))
            for a, b in sorted(pairs)]


# --------------------------------------------------------------------- CLI


def _root_of(rd: dict):
    from walt.contracts import EndgameRoot
    return EndgameRoot(
        decl_id=rd["decl_id"], bidder=rd["bidder"], bid_value=rd["bid_value"],
        bids=tuple(rd["bids"]), dealer=rd["dealer"], me=rd["me"],
        my_hand=tuple(rd["my_hand"]),
        play_history=tuple((s, d) for s, d in rd["play_history"]),
        trick_leader=rd["trick_leader"],
        current_trick=tuple(rd["current_trick"]),
        team_points=tuple(rd["team_points"]))


def main():
    evalset = Path(__file__).parent / "evalset_h4_v1.jsonl"
    recs = [json.loads(x) for x in open(evalset)]
    print(f"{len(recs)} roots")
    keys = {"S": {}, "P": {}, "relaxed": {}}
    aut_hist: Counter = Counter()
    tot_N = tot_orb = 0
    ties = 0
    for i, rec in enumerate(recs):
        root = _root_of(rec["root"])
        sig = Signature(root)
        rel = Signature(root, relax_meme=True)
        keys["S"].setdefault(canonical_key(sig, sig.header_S), []).append(rec["seed"])
        keys["P"].setdefault(canonical_key(sig, sig.header_P), []).append(rec["seed"])
        keys["relaxed"].setdefault(canonical_key(rel, rel.header_P), []).append(rec["seed"])
        auts = automorphisms(rel)
        aut_hist[len(auts)] += 1
        N, orb = world_orbit_count(root, sig, automorphisms(sig))
        tot_N, tot_orb = tot_N + N, tot_orb + orb
        ties += len(interchangeable_pairs(root)) if len(auts) > 1 else 0
        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{len(recs)}", flush=True)
    for name, kk in keys.items():
        print(f"regime {name}: {len(kk)} classes / {len(recs)} "
              f"(collapse {len(recs) / len(kk):.3f}x)")
    print(f"relaxed |Aut| histogram: {dict(sorted(aut_hist.items()))}")
    print(f"worlds -> orbits: {tot_N} -> {tot_orb} ({tot_N / tot_orb:.3f}x)")
    print(f"interchangeable my-tile pairs: {ties}")


if __name__ == "__main__":
    main()
