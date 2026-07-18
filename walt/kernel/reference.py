"""walt/kernel/reference.py — transparent pure-python mirror of CONTRACTS.md.

A small, slow, obviously-correct implementation of the kernel interface
(`build_subgame` / `br_solve` / `SigmaTable` / `StochasticProfile`) for TOY
sizes only (≤2 tricks, ≤12 worlds). It is the correctness mirror the CFR lane
develops against, and an independent cross-check of the kernel lane. No
performance goals; clarity over everything. Zero torch; rules only via
walt.tables (which tabulates forge.oracle.tables — never reimplemented here).

Conventions shared with the kernel lane (the profile-domain contract):

- **node key** = tuple of domino ids played SINCE THE ROOT, in play order.
  The acting seats are implied by position (play rotates from the root's
  current trick position; trick winners lead), so tiles alone identify the
  public node.
- **profile domain** = (seat, hand_mask, node): hand_mask is the acting
  seat's CURRENT remaining uint32 hand mask at that node (as a python int).
- **profile protocol**: both SigmaTable and StochasticProfile expose
  `.dist(seat, hand_mask, node) -> (moves, probs)` with `moves` an ascending
  tuple of domino ids and `probs` a float array summing to 1. br_solve and
  cfr_solve consume profiles only through `.dist`.

Orientation (walt parity): all values are E[payoff43[final declaring points]]
in DECLARING orientation, normalized by total world weight, never flipped.
A seat maximizes sign*value with sign = +1 iff seat%2 == bidder%2. Tie rule:
moves ascending, first strict sign*v > sign*best wins.

Info-set consistency is structural: the recursion walks the PUBLIC tree, so
every world that reaches a node shares the acting seat's information set
partition at that node. At profile nodes, worlds spread over children per
their own hand's distribution and are MERGED per public child before
recursing (no strategy fusion for any downstream decision). A hidden best
responder decomposes exactly into independent solves per root hand class,
because its information sets never span two of its possible root hands.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from walt.tables import get_luts, hand_to_mask, legal_moves_mask, mask_to_tiles

__all__ = [
    "SigmaTable",
    "StochasticProfile",
    "Subgame",
    "BRResult",
    "build_subgame",
    "br_solve",
    "profile_value",
    "sign_of",
    "legal_tiles",
]


def sign_of(seat: int, bid_team: int) -> float:
    """+1 iff the seat is on the declaring team (walt orientation rule)."""
    return 1.0 if seat % 2 == bid_team else -1.0


def legal_tiles(hand_mask: int, led_tile, decl_id: int) -> tuple:
    """Ascending domino ids legal from hand_mask (walt.tables semantics)."""
    return tuple(mask_to_tiles(legal_moves_mask(np.uint32(hand_mask), led_tile, decl_id)))


# --------------------------------------------------------------------------- #
#  Profiles                                                                    #
# --------------------------------------------------------------------------- #

class SigmaTable:
    """Deterministic profile: (seat, hand_mask, node) -> move."""

    def __init__(self, table: dict | None = None):
        self.table: dict = dict(table) if table else {}

    def set(self, seat: int, hand_mask: int, node: tuple, move: int) -> None:
        self.table[(int(seat), int(hand_mask), tuple(node))] = int(move)

    def move(self, seat: int, hand_mask: int, node: tuple) -> int:
        key = (int(seat), int(hand_mask), tuple(node))
        if key not in self.table:
            raise KeyError(f"SigmaTable has no entry for {key}")
        return self.table[key]

    def dist(self, seat: int, hand_mask: int, node: tuple):
        return (self.move(seat, hand_mask, node),), np.array([1.0])

    def __len__(self) -> int:
        return len(self.table)


class StochasticProfile:
    """Mixed profile: (seat, hand_mask, node) -> (moves, probs)."""

    def __init__(self, entries: dict | None = None):
        # entries: {(seat, hand_mask, node): (moves tuple, probs array)}
        self.entries: dict = {}
        for key, (moves, probs) in (entries or {}).items():
            self.set(key[0], key[1], key[2], moves, probs)

    def set(self, seat: int, hand_mask: int, node: tuple, moves, probs) -> None:
        moves = tuple(int(m) for m in moves)
        probs = np.asarray(probs, dtype=np.float64).reshape(-1)
        if len(moves) != len(probs):
            raise ValueError("moves and probs must align")
        if abs(float(probs.sum()) - 1.0) > 1e-9:
            raise ValueError(f"probs must sum to 1, got {probs.sum()}")
        if any(moves[i] >= moves[i + 1] for i in range(len(moves) - 1)):
            raise ValueError("moves must be strictly ascending")
        self.entries[(int(seat), int(hand_mask), tuple(node))] = (moves, probs)

    def dist(self, seat: int, hand_mask: int, node: tuple):
        key = (int(seat), int(hand_mask), tuple(node))
        if key not in self.entries:
            raise KeyError(f"StochasticProfile has no entry for {key}")
        return self.entries[key]

    def __len__(self) -> int:
        return len(self.entries)


# --------------------------------------------------------------------------- #
#  Subgame                                                                     #
# --------------------------------------------------------------------------- #

# Public trick-state carried through the recursion. led_tile is the LED
# domino id (None at trick start); decl_pts is banked declaring points
# including everything credited before the root.
@dataclass(frozen=True)
class _Pub:
    leader: int
    n_in_trick: int
    led_tile: int | None
    brank: int          # best rank so far this trick
    bseat: int          # seat holding the best rank
    tcnt: int           # count points riding on this trick
    decl_pts: int


@dataclass
class Subgame:
    """build_subgame output. cfr_solve relies on .root/.worlds/.weights."""

    root: object
    worlds: np.ndarray        # (N, 3) int64 CURRENT hand masks, ascending non-me seats
    weights: np.ndarray       # (N,) float64
    # derived
    decl_id: int = field(init=False)
    me: int = field(init=False)
    bidder: int = field(init=False)
    bid_team: int = field(init=False)
    seats: tuple = field(init=False)     # the three non-me seats, ascending
    col_of: dict = field(init=False)     # seat -> worlds column
    p0: int = field(init=False)          # plays made before the root
    my_mask0: int = field(init=False)
    pub0: _Pub = field(init=False)

    def __post_init__(self):
        root = self.root
        self.decl_id = int(root.decl_id)
        self.me = int(root.me)
        self.bidder = int(root.bidder)
        self.bid_team = self.bidder % 2
        self.seats = tuple(s for s in range(4) if s != self.me)
        self.col_of = {s: i for i, s in enumerate(self.seats)}
        self.p0 = len(root.play_history)
        self.my_mask0 = int(hand_to_mask(root.my_hand))

        luts = get_luts(self.decl_id)
        ct = tuple(int(d) for d in root.current_trick)
        leader = int(root.trick_leader)
        if (leader + len(ct)) % 4 != self.me:
            raise ValueError("root must be a decision of root.me")
        led_tile, brank, bseat, tcnt = None, -1, -1, 0
        if ct:
            led_tile = ct[0]
            suit = int(luts.led_suit[led_tile])
            for i, t in enumerate(ct):
                r = int(luts.rank[suit, t])
                if r > brank:
                    brank, bseat = r, (leader + i) % 4
                tcnt += int(luts.count[t])
        self.pub0 = _Pub(leader, len(ct), led_tile, brank, bseat, tcnt,
                         int(root.team_points[self.bid_team]))


def build_subgame(root, worlds, weights) -> Subgame:
    worlds = np.asarray(worlds, dtype=np.int64).reshape(-1, 3)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if worlds.shape[0] == 0:
        raise ValueError("build_subgame needs at least one world")
    if weights.shape[0] != worlds.shape[0]:
        raise ValueError("weights must match worlds")
    if not (weights > 0).all():
        raise ValueError("weights must be positive")
    return Subgame(root=root, worlds=worlds, weights=weights)


def _pub_step(pub: _Pub, seat: int, tile: int, luts, bid_team: int) -> _Pub:
    """Advance the public trick state by one play (zeb-exact via LUTs)."""
    if pub.n_in_trick == 0:
        led_tile, suit = tile, int(luts.led_suit[tile])
        brank, bseat = int(luts.rank[suit, tile]), seat
        tcnt = int(luts.count[tile])
    else:
        led_tile = pub.led_tile
        suit = int(luts.led_suit[led_tile])
        r = int(luts.rank[suit, tile])
        brank, bseat = (r, seat) if r > pub.brank else (pub.brank, pub.bseat)
        tcnt = pub.tcnt + int(luts.count[tile])
    n = pub.n_in_trick + 1
    if n == 4:  # trick complete: first-max already enforced by strict '>'
        pts = tcnt + 1
        decl = pub.decl_pts + (pts if bseat % 2 == bid_team else 0)
        return _Pub(bseat, 0, None, -1, -1, 0, decl)
    return _Pub(pub.leader, n, led_tile, brank, bseat, tcnt, pub.decl_pts)


# --------------------------------------------------------------------------- #
#  Exact best response                                                         #
# --------------------------------------------------------------------------- #

@dataclass
class BRResult:
    value: float                 # E[payoff] declaring orientation, normalized
    best_move: int | None        # hero's root move (None when hero != root.me)
    root_values: dict            # move -> value, only when hero == root.me
    strategy: SigmaTable         # hero's BR strategy over reachable info sets


def br_solve(subgame: Subgame, profile, payoff43, hero: int | None = None) -> BRResult:
    """Exact best response of `hero`; every other seat plays `profile`.

    hero defaults to root.me. When hero is a hidden seat, the BR decomposes
    into independent solves per possible root hand of hero (its info sets
    never span two root hands); value is the weighted sum over all classes.
    """
    sub = subgame
    payoff43 = np.asarray(payoff43, dtype=np.float64).reshape(-1)
    if payoff43.shape[0] != 43:
        raise ValueError("payoff43 must have 43 entries")
    hero = sub.me if hero is None else int(hero)
    sgn = sign_of(hero, sub.bid_team)
    luts = get_luts(sub.decl_id)
    total_w = float(sub.weights.sum())
    hero_hidden = hero != sub.me
    hcol = sub.col_of[hero] if hero_hidden else None

    strategy = SigmaTable()
    root_values: dict = {}
    root_best: list = [None]  # [move]; captured at p == p0 when hero == me

    def walk(p, pub, node, wcur, reach, my_mask):
        if p == 28:
            return float(reach.sum()) * payoff43[pub.decl_pts]
        actor = (pub.leader + pub.n_in_trick) % 4

        if actor == hero:
            hand = my_mask if not hero_hidden else int(wcur[0, hcol])
            best = None
            best_m = None
            for m in legal_tiles(hand, pub.led_tile, sub.decl_id):
                if hero_hidden:
                    w2 = wcur.copy()
                    w2[:, hcol] &= ~(1 << m)
                    my2 = my_mask
                else:
                    w2 = wcur
                    my2 = my_mask & ~(1 << m)
                v = walk(p + 1, _pub_step(pub, actor, m, luts, sub.bid_team),
                         node + (m,), w2, reach, my2)
                if p == sub.p0 and not hero_hidden:
                    root_values[m] = v / total_w
                if best is None or sgn * v > sgn * best:
                    best, best_m = v, m
            strategy.set(hero, hand, node, best_m)
            if p == sub.p0 and not hero_hidden:
                root_best[0] = best_m
            return best

        # profile seat: spread each hand-group over its distribution, then
        # MERGE all worlds reaching the same public child before recursing.
        if actor == sub.me:
            hands = np.full(len(reach), my_mask, dtype=np.int64)
        else:
            hands = wcur[:, sub.col_of[actor]]
        child: dict = {}  # move -> (list of sel arrays, list of reach arrays)
        for h in np.unique(hands):
            sel = np.flatnonzero(hands == h)
            legal = legal_tiles(int(h), pub.led_tile, sub.decl_id)
            moves, probs = profile.dist(actor, int(h), node)
            if not set(moves) <= set(legal):
                raise ValueError(
                    f"profile plays illegal move at seat={actor} hand={h:#x} "
                    f"node={node}: {moves} vs legal {legal}")
            for m, pr in zip(moves, probs):
                if pr <= 0.0:
                    continue
                sels, rs = child.setdefault(int(m), ([], []))
                sels.append(sel)
                rs.append(reach[sel] * pr)
        total = 0.0
        for m in sorted(child):
            sels, rs = child[m]
            sel = np.concatenate(sels)
            r2 = np.concatenate(rs)
            w2 = wcur[sel]
            my2 = my_mask
            if actor == sub.me:
                my2 = my_mask & ~(1 << m)
            else:
                w2 = w2.copy()
                w2[:, sub.col_of[actor]] &= ~(1 << m)
            total += walk(p + 1, _pub_step(pub, actor, m, luts, sub.bid_team),
                          node + (m,), w2, r2, my2)
        return total

    if hero_hidden:
        total = 0.0
        for h in np.unique(sub.worlds[:, hcol]):
            sel = np.flatnonzero(sub.worlds[:, hcol] == h)
            total += walk(sub.p0, sub.pub0, (), sub.worlds[sel],
                          sub.weights[sel].copy(), sub.my_mask0)
        return BRResult(total / total_w, None, {}, strategy)

    value = walk(sub.p0, sub.pub0, (), sub.worlds,
                 sub.weights.copy(), sub.my_mask0)
    return BRResult(value / total_w, root_best[0], root_values, strategy)


# --------------------------------------------------------------------------- #
#  Expected value when EVERY seat follows a profile                            #
# --------------------------------------------------------------------------- #

def profile_value(subgame: Subgame, profiles, payoff43) -> float:
    """E[payoff] (declaring orientation, normalized) when all four seats play
    `profiles` — either one profile object covering every seat, or a dict
    seat -> profile object."""
    sub = subgame
    payoff43 = np.asarray(payoff43, dtype=np.float64).reshape(-1)
    luts = get_luts(sub.decl_id)

    def dist_for(seat, hand, node):
        prof = profiles[seat] if isinstance(profiles, dict) else profiles
        return prof.dist(seat, hand, node)

    def walk(p, pub, node, wcur, reach, my_mask):
        if p == 28:
            return float(reach.sum()) * payoff43[pub.decl_pts]
        actor = (pub.leader + pub.n_in_trick) % 4
        if actor == sub.me:
            hands = np.full(len(reach), my_mask, dtype=np.int64)
        else:
            hands = wcur[:, sub.col_of[actor]]
        child: dict = {}
        for h in np.unique(hands):
            sel = np.flatnonzero(hands == h)
            legal = legal_tiles(int(h), pub.led_tile, sub.decl_id)
            moves, probs = dist_for(actor, int(h), node)
            if not set(moves) <= set(legal):
                raise ValueError(
                    f"profile plays illegal move at seat={actor} hand={h:#x}")
            for m, pr in zip(moves, probs):
                if pr <= 0.0:
                    continue
                sels, rs = child.setdefault(int(m), ([], []))
                sels.append(sel)
                rs.append(reach[sel] * pr)
        total = 0.0
        for m in sorted(child):
            sels, rs = child[m]
            sel = np.concatenate(sels)
            r2 = np.concatenate(rs)
            w2 = wcur[sel]
            my2 = my_mask
            if actor == sub.me:
                my2 = my_mask & ~(1 << m)
            else:
                w2 = w2.copy()
                w2[:, sub.col_of[actor]] &= ~(1 << m)
            total += walk(p + 1, _pub_step(pub, actor, m, luts, sub.bid_team),
                          node + (m,), w2, r2, my2)
        return total

    total = walk(sub.p0, sub.pub0, (), sub.worlds, sub.weights.copy(),
                 sub.my_mask0)
    return total / float(sub.weights.sum())
