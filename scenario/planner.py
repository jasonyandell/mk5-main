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

The max is honest only while the cell holds many scenarios: a max over a
one-world cell is perfect-information play for that world, so a plan whose
cells have thinned out is clairvoyant from there on. A rung's `floor` is the
guard. Below the root, when the deciding player's cell holds fewer than
`floor` scenarios, it does not maximize; it plays as a rung k-1 player
holding its own remaining hand at that record would — the move it would
find by sampling afresh, as a modeled seat does — one move for the whole
cell, so V(C) = V(C_m). The root decision is never floored: the root cell is
the whole sample.

The ladder: rung k's modeled seats are rung k-1 players; rung -1 is random
legal play, drawn from a tape keyed on the public record so that one
scenario plays one tile at one record in every branch (worlds partition,
they never multiply) — or, when the config names a field net
(`scenario.field`), rung -1 is that net: a learned policy reading the
modeled seat's own information, which also takes rung 0's floored moves.
The net is a field only; leaves are still scored by the recursion and the
frontier playout. A modeled player's decision is a pure function of
(rung, seat, its remaining hand, the record) and is cached as such.

The belief β weights the LIVE player's scenarios; modeled minds weight
theirs uniformly (compatibility only, the g ≡ 0 limit), as the shipped
player does. β has the tilt form b = u·e^g: u is the uniform (exact, no
invalid worlds — the fiber is enumerated outright when it is no larger than
the sample), and g is a log-likelihood of what has been observed. The
built-in `Tilt` scores the hidden seats' discretionary plays under a field
model — `dice` (each legal tile equally likely, so forced-looking plays
weigh more) or `field` (the rung-0 player's own choice, ε-greedy); any
further evidence (the auction, a learned head) multiplies in through
`Walt(belief=callable)`, which sees the whole public `Situation`. Every random draw — scenario sampling and the dice tape — comes
from SplitMix64 streams seeded by the invocation, so a decision is a pure
function of (config, state, bid) that any implementation can reproduce
bit for bit; `scenario/core` is the compiled one, checked against this file.

Values are kept in bidder-team units: with the `make` utility a cell's value
is the belief mass in which the bidding side makes its contract; with
`points` it is the mass-weighted signed point margin. A seat on the bidding
team maximizes, a defender minimizes — the same u read from two sides.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from forge.zeb.game import current_player
from forge.zeb.types import ZebGameState

from scenario.field import FieldNet
from scenario.fiber import sample_worlds
from scenario.record import Record, empty_record, record_from_state, remaining_hand
from scenario.rules import ALL_TILES, Rules, bit, mask_of, rules_for, tiles_of

World = tuple[int, int, int, int]                       # remaining-hand masks, absolute seats

HAND_TOTAL = 42
MASK64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15


@dataclass(frozen=True)
class Rung:
    """The hats: one rung's declared computation."""

    n_worlds: int           # scenarios this rung samples at an invocation
    horizon: int | None     # plays searched explicitly from the invocation root; None = to the end
    floor: int = 1          # below the root, a cell this thin hands the decider's move down a rung

    def __post_init__(self) -> None:
        if self.floor < 1:
            raise ValueError("a cell floor is at least 1")


@dataclass(frozen=True)
class Tilt:
    """g, the built-in evidence: how the hidden seats' discretionary plays are
    scored in each scenario. `none` is uniform; `dice` gives a play with n
    legal options likelihood 1/n; `field` gives it ε/n + (1−ε)·[the rung-0
    player holding that hand would have played it]."""

    plays: str = "none"
    eps: float = 0.1

    def __post_init__(self) -> None:
        if self.plays not in ("none", "dice", "field"):
            raise ValueError(f"unknown tilt {self.plays!r}")
        if not 0.0 < self.eps <= 1.0:
            raise ValueError("eps must lie in (0, 1]")


@dataclass(frozen=True)
class Config:
    rungs: tuple[Rung, ...]     # rungs[k] configures D_k; the live player is the last rung
    utility: str = "make"       # "make" (bidding side makes it) or "points" (signed margin)
    seed: int = 0
    tilt: Tilt = Tilt()
    field: FieldNet | None = None   # rung -1: the field net, or None for random legal play

    def __post_init__(self) -> None:
        if not self.rungs:
            raise ValueError("a player needs at least one rung")
        if self.utility not in ("make", "points"):
            raise ValueError(f"unknown utility {self.utility!r}")
        if self.tilt.plays == "field" and self.level == 0:
            raise ValueError("the field tilt models plays with rung 0; a one-rung player has no modeled rung")
        if self.rungs[0].floor > 1 and self.field is None:
            raise ValueError("rung 0 hands a floored move down only to a field net")

    @property
    def level(self) -> int:
        return len(self.rungs) - 1


def _splitmix(x: int) -> int:
    x = (x + GOLDEN) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return x ^ (x >> 31)


def _mix(*parts: int) -> int:
    h = 0
    for p in parts:
        h = _splitmix(h ^ (p & MASK64))
    return h


class SplitMix:
    """The SplitMix64 stream: `_splitmix` of a state that advances by GOLDEN."""

    __slots__ = ("state",)

    def __init__(self, seed: int):
        self.state = seed & MASK64

    def next_u64(self) -> int:
        out = _splitmix(self.state)
        self.state = (self.state + GOLDEN) & MASK64
        return out

    def random(self) -> float:
        """Uniform on [0, 1) with 53 random bits."""
        return (self.next_u64() >> 11) * 2.0 ** -53


def _pack(played: int, leader: int, trick: tuple[int, ...], banked: tuple[int, int]) -> int:
    """The public record (less voids) as one integer below 2**59: the tape key."""
    v = played | (leader << 28) | (len(trick) << 30)
    for i, t in enumerate(trick):
        v |= t << (32 + 5 * i)
    return v | (banked[0] << 47) | (banked[1] << 53)


def _tape_key(rec: Record) -> int:
    return _pack(rec.played, rec.leader, rec.trick, rec.banked)


@dataclass(frozen=True)
class Situation:
    """Everything public at a decision, as the belief sees it."""

    decl_id: int
    viewer: int
    bid: int
    bidder: int                              # the declaring seat
    record: Record
    plays: tuple[tuple[int, int], ...]       # (seat, tile) in play order

    @property
    def bidder_team(self) -> int:
        return self.bidder % 2

    def played_by(self, seat: int) -> int:
        return mask_of(t for s, t in self.plays if s == seat)


Belief = Callable[[Situation, list[World]], list[float]]  # evidence beyond the tilt: weights per world


def situation_from_state(state: ZebGameState, bid: int) -> Situation:
    rec = record_from_state(state)
    return Situation(
        decl_id=state.decl_id, viewer=rec.seat, bid=int(bid), bidder=state.bidder,
        record=rec, plays=tuple((int(s), int(t)) for s, t in state.play_history),
    )


@dataclass(frozen=True)
class Analysis:
    """One decision, opened up. `worlds` are the live rung's scenarios in
    sample order and `weights` the belief over them exactly as the planner
    weighs them (tilt × external evidence, unnormalized). `outcomes[a][r]`
    is the bidder-team final margin scenario r reaches when the root action
    is a and play continues under the plan — the viewer's moves chosen once
    per cell, the other seats as modeled — under every utility, so that
    q[a] = Σ_r weights[r]·u(outcomes[a][r]) / Σ_r weights[r] with u the
    identity under `points` and [margin ≥ 2·bid − 42] under `make`. Where
    the plan stops before the hand ends (the contract decided under `make`,
    or the horizon) the rest of the hand is played from the tape: beyond
    that point the margin is the tape's, not the plan's."""

    tile: int
    q: dict[int, float]
    worlds: list[World]
    weights: list[float]
    outcomes: dict[int, tuple[float, ...]]


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
        inv = self._root(state, bid)
        if isinstance(inv, int):
            return inv, {inv: float("nan")}
        q = inv.action_values()
        return _best(q, inv.maximize), q

    def analyze(self, state: ZebGameState, bid: int) -> "Analysis":
        """`decide`, opened up: the same tile and action values, plus where
        each root action sends every live scenario under the plan. A forced
        move is not planned (as in `decide`): its analysis holds no scenarios."""
        inv = self._root(state, bid)
        if isinstance(inv, int):
            return Analysis(inv, {inv: float("nan")}, [], [], {inv: ()})
        q = inv.action_values()
        return Analysis(_best(q, inv.maximize), q, inv.worlds, inv.weights, inv.outcomes())

    def _root(self, state: ZebGameState, bid: int) -> "_Invocation | int":
        """The live invocation at a decision — or the tile, when it is forced."""
        sit = situation_from_state(state, bid)
        rec, viewer = sit.record, sit.viewer
        if viewer != current_player(state):
            raise AssertionError("record seat disagrees with engine")
        own = remaining_hand(state, viewer)
        rules = rules_for(state.decl_id)
        legal = tiles_of(rules.legal(own, rec.led))
        self._pi.clear()
        self.nodes = 0
        if len(legal) == 1:
            return legal[0]
        return self._invocation(self.config.level, viewer, own, rec, rules, bid, sit.bidder_team, sit)

    # -- the belief --------------------------------------------------------------

    def tilt_weights(self, sit: Situation, worlds: list[World], rules: Rules) -> list[float]:
        """e^g for each scenario: the likelihood of the hidden seats' observed
        discretionary plays, replayed from the first lead with each seat
        holding the hand the scenario implies it was dealt."""
        tilt = self.config.tilt
        w = [1.0] * len(worlds)
        if tilt.plays == "none" or not sit.plays:
            return w
        dice = tilt.plays == "dice"
        eps = tilt.eps
        full = [[hands[s] | sit.played_by(s) for s in range(4)] for hands in worlds]
        rec = empty_record(sit.plays[0][0])
        for seat, tile in sit.plays:
            if seat != sit.viewer:
                for r, hands in enumerate(full):
                    legal = rules.legal(hands[seat], rec.led)
                    n = legal.bit_count()
                    if n > 1:
                        if dice:
                            w[r] *= 1.0 / n
                        else:
                            m = self._modeled_move(0, seat, hands[seat], rec, rules, sit.bid, sit.bidder_team)
                            w[r] *= eps / n + (1.0 - eps) * (1.0 if m == tile else 0.0)
                    hands[seat] &= ~bit(tile)
            rec = rec.advance(tile, rules)
        return w

    # -- the ladder ------------------------------------------------------------

    def _invocation(self, k: int, viewer: int, own: int, rec: Record, rules: Rules,
                    bid: int, bidder_team: int, sit: Situation | None = None) -> "_Invocation":
        """`sit` is given at the live rung only: modeled minds weigh their
        scenarios uniformly."""
        rung = self.config.rungs[k]
        seed = _mix(self.config.seed, k, viewer, own, _tape_key(rec), *rec.forbid)
        rng = SplitMix(seed)
        sizes = (rec.hand_size(0), rec.hand_size(1), rec.hand_size(2), rec.hand_size(3))
        worlds = sample_worlds(viewer, own, rec.played, sizes, rec.forbid, rung.n_worlds, rng)
        weights = [1.0] * len(worlds)
        if sit is not None:
            weights = self.tilt_weights(sit, worlds, rules)
            if self.belief is not None:
                weights = [w * x for w, x in zip(weights, self.belief(sit, worlds))]
        tapes = [_mix(seed, i) for i in range(len(worlds))]
        return _Invocation(self, k, viewer, rec, rules, bid, bidder_team, worlds, weights, tapes)

    def _modeled_move(self, k: int, seat: int, own: int, rec: Record, rules: Rules,
                      bid: int, bidder_team: int) -> int:
        """What a rung-k player holding `own` at `seat` plays at this record;
        rung -1 is the field net."""
        key = (k, seat, own, rec.key())
        tile = self._pi.get(key)
        if tile is None:
            legal = tiles_of(rules.legal(own, rec.led))
            if len(legal) == 1:
                tile = legal[0]
            elif k < 0:
                assert self.config.field is not None, "rung -1 is random play, not a mind"
                tile = self.config.field.move(rules, seat, own, rec, bid, bidder_team)
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
        self.floor = walt.config.rungs[k].floor
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
        u = self._settled(rec)
        if u is not None:
            return u * self._mass(alive)
        if self._at_frontier(rec):
            return sum(self.weights[r] * self._playout(r, rec) for r in _bits(alive))
        seat = rec.seat
        if seat == self.viewer:
            if self._floored(alive):
                return self.value(rec.advance(self._handed_down(rec), self.rules), alive)
            hand = self.worlds[0][seat] & ~rec.played
            vals = [self.value(rec.advance(t, self.rules), alive)
                    for t in tiles_of(self.rules.legal(hand, rec.led))]
            return max(vals) if self.maximize else min(vals)
        return sum(self.value(rec.advance(t, self.rules), cell)
                   for t, cell in self._partition(seat, rec, alive).items())

    def _settled(self, rec: Record) -> float | None:
        """The utility every scenario shares once the hand is over — or, under
        `make`, once the contract is decided and the rest cannot change it."""
        if self.make:
            if rec.banked[self.bidder_team] >= self.bid:
                return 1.0
            if rec.banked[1 - self.bidder_team] > HAND_TOTAL - self.bid:
                return 0.0
        if rec.finished:
            return self._score(rec)
        return None

    def _at_frontier(self, rec: Record) -> bool:
        return self.horizon is not None and rec.plays_so_far() - self.root_plays >= self.horizon

    def _partition(self, seat: int, rec: Record, alive: int) -> dict[int, int]:
        """The field acts: the cell split by the tile the viewer sees, in
        first-seen order."""
        buckets: dict[int, int] = {}
        for r in _bits(alive):
            hand = self.worlds[r][seat] & ~rec.played
            tile = self._field_move(seat, r, hand, rec)
            buckets[tile] = buckets.get(tile, 0) | (1 << r)
        return buckets

    # -- the trace ----------------------------------------------------------------

    def outcomes(self) -> dict[int, tuple[float, ...]]:
        """Per legal root action, the leaf value each scenario reaches under
        the plan. A replay of the solved tree, not a second solve: every cell
        it visits is already in the memo, and the policy subtree it walks
        holds at most one path per scenario."""
        own = self.worlds[0][self.viewer] & ~self.root.played
        everyone = (1 << len(self.worlds)) - 1
        out = {}
        for t in tiles_of(self.rules.legal(own, self.root.led)):
            leaf = [0.0] * len(self.worlds)
            self._trace(self.root.advance(t, self.rules), everyone, leaf)
            out[t] = tuple(leaf)
        return out

    def _trace(self, rec: Record, alive: int, leaf: list[float]) -> None:
        if rec.finished:
            for r in _bits(alive):
                leaf[r] = self._margin(rec)
            return
        if self._settled(rec) is not None or self._at_frontier(rec):
            for r in _bits(alive):
                leaf[r] = self._playout(r, rec, settle=False)
            return
        seat = rec.seat
        if seat == self.viewer:
            if self._floored(alive):
                self._trace(rec.advance(self._handed_down(rec), self.rules), alive, leaf)
                return
            hand = self.worlds[0][seat] & ~rec.played
            q = {t: self.value(rec.advance(t, self.rules), alive)
                 for t in tiles_of(self.rules.legal(hand, rec.led))}
            self._trace(rec.advance(_best(q, self.maximize), self.rules), alive, leaf)
            return
        for t, cell in self._partition(seat, rec, alive).items():
            self._trace(rec.advance(t, self.rules), cell, leaf)

    # -- the floor ----------------------------------------------------------------

    def _floored(self, alive: int) -> bool:
        """Too few scenarios left in the cell for a max over them to be honest."""
        return alive.bit_count() < self.floor

    def _handed_down(self, rec: Record) -> int:
        """The decider's move in a floored cell: the rung k-1 player's (the
        field net's at rung 0), holding the decider's remaining hand (the same
        in every scenario of the cell)."""
        hand = self.worlds[0][self.viewer] & ~rec.played
        return self.walt._modeled_move(
            self.k - 1, self.viewer, hand, rec, self.rules, self.bid, self.bidder_team,
        )

    # -- the field ----------------------------------------------------------------

    def _field_move(self, seat: int, r: int, hand: int, rec: Record) -> int:
        if self.k == 0 and self.walt.config.field is None:
            return self._dice(r, rec, self.rules.legal(hand, rec.led))
        return self.walt._modeled_move(
            self.k - 1, seat, hand, rec, self.rules, self.bid, self.bidder_team,
        )

    def _dice(self, r: int, rec: Record, legal: int) -> int:
        """Random legal play from scenario r's tape, keyed on the record."""
        if legal & (legal - 1) == 0:  # one legal tile
            return legal.bit_length() - 1
        tiles = tiles_of(legal)
        return tiles[_splitmix(self.tapes[r] ^ _tape_key(rec)) % len(tiles)]

    # -- leaves --------------------------------------------------------------------

    def _playout(self, r: int, rec: Record, settle: bool = True) -> float:
        """The frontier continuation: every seat (the viewer too) plays from
        the tape. The same record semantics as `Record.advance`, unrolled on
        plain integers — this loop is the search's hot spot, and no modeled
        mind samples inside it, so the void ledger need not be kept. With
        `settle` the utility is returned as soon as it is decided; without
        it the hand is played out and the margin returned (the trace)."""
        world = self.worlds[r]
        rules = self.rules
        tape = self.tapes[r]
        bt = self.bidder_team
        make, bid = self.make and settle, self.bid
        played, leader, trick = rec.played, rec.leader, list(rec.trick)
        banked = list(rec.banked)
        while played != ALL_TILES:
            if make:
                if banked[bt] >= bid:
                    return 1.0
                if banked[1 - bt] > HAND_TOTAL - bid:
                    return 0.0
            seat = (leader + len(trick)) % 4
            hand = world[seat] & ~played
            if trick:
                legal = hand & rules.follow_bits[rules.led_suit[trick[0]]] or hand
            else:
                legal = hand
            if legal & (legal - 1) == 0:
                tile = legal.bit_length() - 1
            else:
                tiles = tiles_of(legal)
                key = _pack(played, leader, trick, (banked[0], banked[1]))
                tile = tiles[_splitmix(tape ^ key) % len(tiles)]
            played |= 1 << tile
            trick.append(tile)
            if len(trick) == 4:
                t4 = (trick[0], trick[1], trick[2], trick[3])
                leader = (leader + rules.winner_offset(t4)) % 4
                banked[leader % 2] += rules.trick_points(t4)
                trick = []
        if make:
            return 1.0 if banked[bt] >= bid else 0.0
        return float(banked[bt] - banked[1 - bt])

    def _score(self, rec: Record) -> float:
        if self.make:
            return 1.0 if rec.banked[self.bidder_team] >= self.bid else 0.0
        return self._margin(rec)

    def _margin(self, rec: Record) -> float:
        bt = self.bidder_team
        return float(rec.banked[bt] - rec.banked[1 - bt])

    def _mass(self, alive: int) -> float:
        return sum(self.weights[r] for r in _bits(alive))


def _bits(mask: int):
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low
