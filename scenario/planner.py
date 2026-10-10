"""The finite planning rule and the ladder of players.

One decision invocation: form a belief (a weighted set of scenarios), solve
the continuation problem over cells of scenarios the player cannot tell
apart, return the first action of the best plan.

    V(C) = sum w·u                      finished
         = sum w·L                      at the search frontier
         = max_a sum_o V(C_{a,o})       the deciding player chooses
         = sum_o V(C_o)                 another seat acts

Two operations: choose ONE action across the whole cell; ADD the values of
the distinguishable continuations. At another seat's turn each scenario's
modeled player acts from its own information, and the cell partitions by
the tile the deciding player sees.

The ladder: rung k's modeled seats are rung k-1 players; rung -1 is random
legal play, drawn from a tape keyed on the public record so that one
scenario plays one tile at one record in every branch (worlds partition,
they never multiply). A modeled player's decision is a pure function of
(rung, seat, its remaining hand, the record) and is cached as such.

Values are kept in bidder-team units: with the `make` utility a cell's value
is the belief mass in which the bidding side makes its contract; with
`points` it is the mass-weighted signed point margin. A seat on the bidding
team maximizes, a defender minimizes — the same u read from two sides.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from forge.zeb.game import current_player
from forge.zeb.types import ZebGameState

from scenario.fiber import sample_worlds
from scenario.record import Record, record_from_state, remaining_hand
from scenario.rules import Rules, rules_for, tiles_of

World = tuple[int, int, int, int]                       # remaining-hand masks, absolute seats
Belief = Callable[[Record, int, list[World]], list[float]]  # (record, viewer, worlds) -> weights

HAND_TOTAL = 42
MASK64 = (1 << 64) - 1


@dataclass(frozen=True)
class Rung:
    """The hats: one rung's declared computation."""

    n_worlds: int           # scenarios this rung samples at an invocation
    horizon: int | None     # plays searched explicitly from the invocation root; None = to the end


@dataclass(frozen=True)
class Config:
    rungs: tuple[Rung, ...]     # rungs[k] configures D_k; the live player is the last rung
    utility: str = "make"       # "make" (bidding side makes it) or "points" (signed margin)
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.rungs:
            raise ValueError("a player needs at least one rung")
        if self.utility not in ("make", "points"):
            raise ValueError(f"unknown utility {self.utility!r}")

    @property
    def level(self) -> int:
        return len(self.rungs) - 1


def _splitmix(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return x ^ (x >> 31)


def _mix(*parts: int) -> int:
    h = 0
    for p in parts:
        h = _splitmix(h ^ (p & MASK64))
    return h


def _tape_key(rec: Record) -> int:
    packed = 0
    for t in rec.trick:
        packed = (packed << 5) | t
    return _mix(rec.played, rec.leader, packed, rec.banked[0], rec.banked[1])


class Walt:
    """A particular Walt: a Config plus a belief constructor."""

    def __init__(self, config: Config, belief: Belief | None = None):
        self.config = config
        self.belief = belief
        self._pi: dict[tuple, int] = {}   # modeled decisions, by (rung, seat, hand, record key)
        self.nodes = 0                     # cells evaluated in the last decision

    # -- the player at the table --------------------------------------------

    def decide(self, state: ZebGameState, bid: int) -> tuple[int, dict[int, float]]:
        """Return (tile, action values) for the seat to play in a PLAYING state."""
        rec = record_from_state(state)
        viewer = rec.seat
        if viewer != current_player(state):
            raise AssertionError("record seat disagrees with engine")
        own = remaining_hand(state, viewer)
        rules = rules_for(state.decl_id)
        legal = tiles_of(rules.legal(own, rec.led))
        self._pi.clear()
        self.nodes = 0
        if len(legal) == 1:
            return legal[0], {legal[0]: float("nan")}
        inv = self._invocation(self.config.level, viewer, own, rec, rules, bid, state.bidder % 2)
        q = inv.action_values()
        return _best(q, inv.maximize), q

    # -- the ladder ------------------------------------------------------------

    def _invocation(self, k: int, viewer: int, own: int, rec: Record, rules: Rules,
                    bid: int, bidder_team: int) -> "_Invocation":
        rung = self.config.rungs[k]
        seed = _mix(self.config.seed, k, viewer, own, hash(rec.key()))
        rng = random.Random(seed)
        sizes = (rec.hand_size(0), rec.hand_size(1), rec.hand_size(2), rec.hand_size(3))
        worlds = sample_worlds(viewer, own, rec.played, sizes, rec.forbid, rung.n_worlds, rng)
        weights = self.belief(rec, viewer, worlds) if self.belief else [1.0] * len(worlds)
        tapes = [_mix(seed, i) for i in range(len(worlds))]
        return _Invocation(self, k, viewer, rec, rules, bid, bidder_team, worlds, weights, tapes)

    def _modeled_move(self, k: int, seat: int, own: int, rec: Record, rules: Rules,
                      bid: int, bidder_team: int) -> int:
        """What a rung-k player holding `own` at `seat` plays at this record."""
        key = (k, seat, own, rec.key())
        tile = self._pi.get(key)
        if tile is None:
            legal = tiles_of(rules.legal(own, rec.led))
            if len(legal) == 1:
                tile = legal[0]
            else:
                inv = self._invocation(k, seat, own, rec, rules, bid, bidder_team)
                tile = _best(inv.action_values(), inv.maximize)
            self._pi[key] = tile
        return tile


def _best(q: dict[int, float], maximize: bool) -> int:
    """The tie rule: among equal-valued moves, the lowest tile id."""
    items = sorted(q.items())
    if maximize:
        return max(items, key=lambda kv: kv[1])[0]
    return min(items, key=lambda kv: kv[1])[0]


class _Invocation:
    """One planning problem: a viewer, a rung, a sample, and its memo."""

    def __init__(self, walt: Walt, k: int, viewer: int, root: Record, rules: Rules,
                 bid: int, bidder_team: int, worlds: list[World],
                 weights: list[float], tapes: list[int]):
        self.walt = walt
        self.k = k
        self.viewer = viewer
        self.root = root
        self.rules = rules
        self.bid = bid
        self.bidder_team = bidder_team
        self.maximize = viewer % 2 == bidder_team
        self.worlds = worlds
        self.weights = weights
        self.tapes = tapes
        self.make = walt.config.utility == "make"
        self.horizon = walt.config.rungs[k].horizon
        self.root_plays = root.plays_so_far()
        self.memo: dict[tuple, float] = {}

    # -- values -------------------------------------------------------------------

    def action_values(self) -> dict[int, float]:
        """Q(a) for each legal first action: value per unit of belief mass."""
        own = self.worlds[0][self.viewer] & ~self.root.played
        everyone = (1 << len(self.worlds)) - 1
        total = self._mass(everyone)
        return {
            t: self.value(self.root.advance(t, self.rules), everyone) / total
            for t in tiles_of(self.rules.legal(own, self.root.led))
        }

    def value(self, rec: Record, alive: int) -> float:
        key = (rec.key(), alive)
        v = self.memo.get(key)
        if v is not None:
            return v
        v = self._value(rec, alive)
        self.memo[key] = v
        return v

    def _value(self, rec: Record, alive: int) -> float:
        self.walt.nodes += 1
        if self.make:
            # Decided contracts: the rest of the hand cannot change the outcome.
            if rec.banked[self.bidder_team] >= self.bid:
                return self._mass(alive)
            if rec.banked[1 - self.bidder_team] > HAND_TOTAL - self.bid:
                return 0.0
        if rec.finished:
            return self._score(rec) * self._mass(alive)
        if self.horizon is not None and rec.plays_so_far() - self.root_plays >= self.horizon:
            return sum(self.weights[r] * self._playout(r, rec) for r in _bits(alive))
        seat = rec.seat
        if seat == self.viewer:
            hand = self.worlds[0][seat] & ~rec.played
            vals = [self.value(rec.advance(t, self.rules), alive)
                    for t in tiles_of(self.rules.legal(hand, rec.led))]
            return max(vals) if self.maximize else min(vals)
        buckets: dict[int, int] = {}
        for r in _bits(alive):
            hand = self.worlds[r][seat] & ~rec.played
            tile = self._field_move(seat, r, hand, rec)
            buckets[tile] = buckets.get(tile, 0) | (1 << r)
        return sum(self.value(rec.advance(t, self.rules), cell) for t, cell in buckets.items())

    # -- the field ----------------------------------------------------------------

    def _field_move(self, seat: int, r: int, hand: int, rec: Record) -> int:
        legal = self.rules.legal(hand, rec.led)
        if self.k == 0:
            return self._dice(r, rec, legal)
        return self.walt._modeled_move(
            self.k - 1, seat, hand, rec, self.rules, self.bid, self.bidder_team,
        )

    def _dice(self, r: int, rec: Record, legal: int) -> int:
        """Random legal play from scenario r's tape, keyed on the record."""
        tiles = tiles_of(legal)
        if len(tiles) == 1:
            return tiles[0]
        return tiles[_mix(self.tapes[r], _tape_key(rec)) % len(tiles)]

    # -- leaves --------------------------------------------------------------------

    def _playout(self, r: int, rec: Record) -> float:
        """The frontier continuation: every seat (the viewer too) plays from the tape."""
        world = self.worlds[r]
        while not rec.finished:
            if self.make:
                if rec.banked[self.bidder_team] >= self.bid:
                    return 1.0
                if rec.banked[1 - self.bidder_team] > HAND_TOTAL - self.bid:
                    return 0.0
            seat = rec.seat
            legal = self.rules.legal(world[seat] & ~rec.played, rec.led)
            rec = rec.advance(self._dice(r, rec, legal), self.rules)
        return self._score(rec)

    def _score(self, rec: Record) -> float:
        bt = self.bidder_team
        if self.make:
            return 1.0 if rec.banked[bt] >= self.bid else 0.0
        return float(rec.banked[bt] - rec.banked[1 - bt])

    def _mass(self, alive: int) -> float:
        return sum(self.weights[r] for r in _bits(alive))


def _bits(mask: int):
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low
