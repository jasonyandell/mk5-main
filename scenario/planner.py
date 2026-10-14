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
the whole sample. The floor guards only a sample: when the fiber is small
enough to be enumerated outright (late in the hand), every cell is the exact
posterior under the model, a thin cell means the record really does narrow
the worlds that far, and the max over it is a best response rather than
clairvoyance — so an enumerated invocation is never floored.

The ladder: rung k's modeled seats are rung k-1 players; rung -1 is random
legal play, drawn from a tape keyed on the public record so that one
scenario plays one tile at one record in every branch (worlds partition,
they never multiply) — or, when the config names a field net
(`scenario.field`), rung -1 is that net: a learned policy reading the
modeled seat's own information, which also takes rung 0's floored moves.
The net is a field only; leaves are still scored by the recursion and the
frontier playout. The net's field may be made stochastic, F_σ with
`Config.field_eps` = ε: a modeled seat explores with probability ε, drawing
uniformly over its legal tiles from the same record-keyed tape as the dice,
and otherwise plays the net's move, so

    P_σ(t | hand, record, world) = ε/n + (1 − ε)·[t = the net's move]

over the n legal tiles, positive on every one when ε > 0, so the posterior
under F_σ (`Walt.field_likelihood`) has full support on the fiber. The
draw is the tape's, so it is one tile per (world, record), and worlds still
partition. ε = 0 is the deterministic net, byte for byte. A modeled
player's decision is a pure function of (rung, seat, its remaining hand,
the record) and is cached as such.

The belief β weights the LIVE player's scenarios; modeled minds weight
theirs uniformly (compatibility only, the g ≡ 0 limit), as the shipped
player does. β has the tilt form b = u·e^g: u is the uniform (exact, no
invalid worlds — the fiber is enumerated outright when it is no larger than
the sample), and g is a log-likelihood of what has been observed. The
other seats' plays have one likelihood, the field's (`field_likelihood`),
and it enters once, as the measure the scenarios are drawn from (below);
any further evidence (the auction, a learned head) multiplies in through
`Walt(belief=callable)`, which sees the whole public `Situation`.

The live rung's scenarios themselves may be drawn from the field's
posterior instead of the uniform (`Config.draw = "posterior"`, a one-rung
player with a field net): β^F(world | record) ∝ u(world)·L(world), L the
probability that F_σ plays what the other seats played (`field_likelihood`).
A strong L must be sampled from, never used to reweight a uniform sample
(the weights collapse onto a handful of worlds), so: when the fiber is
enumerated its scenarios are weighted by L normalized — the exact posterior;
when it is sampled, the scenarios are n draws from β^F by rejection from
the uniform stream (a proposal is kept when a uniform coin falls below its
L ≤ 1; at ε = 0, L is the net's 0/1 filter and no coin is drawn), equally
weighted. A decision that has not found n within `draw_budget` proposals,
or whose fiber holds no world the field explains, plans over the uniform
sample instead, and says so (`Walt.fallbacks`).

A `Reading` (`Config.reading`) makes the posterior's likelihood the net's
set instead of its move: at each of another seat's discretionary plays, A is
the legal tiles the net scores within `margin` of its best for that seat's
hand in the world, and the play weighs [tile ∈ A] (the consistent
likelihood) or [tile ∈ A]/|A| (the uniform one: a pick uniform within the
set, under which the set's size is evidence — restricted choice), as Silas
reads a policy's allowed set (`silas/README.md`, The record as evidence).
Its `eps` softens the set as F_σ softens the move — a play weighs
ε/n + (1 − ε)·[tile ∈ A]·(1 or 1/|A|) — so at ε > 0 no world vanishes and
the posterior never empties (#145); ε = 0 is the sharp set. A sharp reading
need not fit the real deal's own seats, and a soft one is still sharp
enough that reweighting collapses, so its draw adds a middle path: L normalized on a fiber of at most n;
n draws with replacement from the whole fiber weighted by L when it holds at
most `READ_ENUM` members (enumerated: exact draws however few worlds L
keeps); rejection from the uniform stream above that (a coin under the
uniform likelihood).

A convention (`Config.convention`, `scenario.convention`) overlays the live
seat: as sender, at a gated position it plays the plan's best tile among
the rule's allowed ones (the q reported is the plan's, unchanged;
`Walt.sent` says the override was in force); as reader, its worlds are
drawn from the rule-only posterior — the same rejection from the uniform
stream, under the 0/1 likelihood that the partner's gated decisions obeyed
the rule (`convention_likelihood`), no coin drawn, with the same budget and
fallback. A reader draws from the uniform otherwise, never from β^F: the
field's likelihood would model the partner's moves a second time.

The tie rule ≺ (`Config.tie`) picks among the decision's exact ties: root
actions whose q is the best q as a double, in the decision's own sample (no
noise bar; under uniform weights, equal counts of worlds made). `lowest`, the
default, takes the lowest tile id. `margin` — make, then margin — takes the
tie with the best mean bidder-team margin for the decider's side under the
plan, M(t) = Σ_r w_r·m_r(t), m_r(t) the margin scenario r reaches when the
root plays t and the plan continues (`Analysis.outcomes`: where the plan
stops, the contract decided or the horizon, the tape plays the rest), then
the lowest tile id. It acts at every decision a Walt makes — the live root, a
modeled rung's, a convention sender's — and never inside the plan, whose
interior choices keep the lowest id among equal values (#120).

Every random draw — scenario sampling, the posterior's coins and the dice
tape — comes from SplitMix64 streams seeded by the invocation's coordinate
(viewer, own hand, record; never the rung), so a decision is a pure function
of (config, state, bid) that any implementation can reproduce bit for bit; `scenario/core` is the compiled one, checked against this file.

Values are kept in bidder-team units: with the `make` utility a cell's value
is the belief mass in which the bidding side makes its contract; with
`points` it is the mass-weighted signed point margin. A seat on the bidding
team maximizes, a defender minimizes — the same u read from two sides.
"""
from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from itertools import accumulate
from typing import Callable

from forge.zeb.game import current_player
from forge.zeb.types import ZebGameState

from scenario.convention import Convention
from scenario.field import FieldNet
from scenario.fiber import Fiber, sample_worlds
from scenario.record import Record, empty_record, record_from_state, remaining_hand
from scenario.rules import ALL_TILES, Rules, bit, mask_of, rules_for, tiles_of

World = tuple[int, int, int, int]                       # remaining-hand masks, absolute seats
DRAWS = ("uniform", "posterior")
LIKELIHOODS = ("consistent", "uniform")
TIES = ("lowest", "margin")   # the tie rule ≺: the lowest tile id, or make then margin
DRAW_BUDGET = 1 << 22   # posterior: uniform proposals per decision before falling back
READ_ENUM = 1 << 20     # a reading draws from the whole fiber, enumerated, when it holds at most this many

HAND_TOTAL = 42
MASK64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15
COIN = 0x636F696E            # "coin": the posterior's acceptance stream is _mix(seed, COIN)


@dataclass(frozen=True)
class Rung:
    """The hats: one rung's declared computation."""

    n_worlds: int           # scenarios this rung samples at an invocation
    horizon: int | None     # plays searched explicitly from the invocation root; None = to the end
    floor: int = 1          # below the root, a sampled cell this thin hands the decider's move down a rung

    def __post_init__(self) -> None:
        if self.floor < 1:
            raise ValueError("a cell floor is at least 1")


@dataclass(frozen=True)
class Reading:
    """The posterior's likelihood as the field net's set A, the legal tiles
    the net scores within `margin` of its best (in the net's integer score
    units; a margin wider than every gap is every legal tile: the
    any-legal-tile policy), softened by `eps` as F_σ softens the net's move.
    A play of one of n legal tiles weighs

        ε/n + (1 − ε)·[tile ∈ A]·w,   w = 1 (`consistent`), 1/|A| (`uniform`)

    so ε = 0 is the sharp set (a play outside A weighs 0) and any ε > 0 is
    positive on every world: the posterior never empties. Under the uniform
    likelihood it is the probability that a seat which explores uniformly
    with probability ε, else picks uniformly within A, plays the tile; at
    |A| = 1 both are F_σ's P_σ."""

    likelihood: str
    margin: int
    eps: float = 0.0

    def __post_init__(self) -> None:
        if self.likelihood not in LIKELIHOODS:
            raise ValueError(f"the likelihood is one of {LIKELIHOODS}, not {self.likelihood!r}")
        if not 0 <= self.margin < 1 << 62:
            raise ValueError("a reading's margin lies in [0, 2^62)")
        if not 0.0 <= self.eps <= 1.0:
            raise ValueError("a reading's eps lies in [0, 1]")

    def weight(self, tile: int, allowed: int, n: int) -> float:
        """The play's factor of the likelihood: tile among n legal tiles, A = `allowed`."""
        inside = (allowed >> tile) & 1
        w = 0.0 if not inside else 1.0 if self.likelihood == "consistent" else 1.0 / allowed.bit_count()
        return self.eps / n + (1.0 - self.eps) * w


@dataclass(frozen=True)
class Config:
    rungs: tuple[Rung, ...]     # rungs[k] configures D_k; the live player is the last rung
    utility: str = "make"       # "make" (bidding side makes it) or "points" (signed margin)
    seed: int = 0
    field: FieldNet | None = None   # rung -1: the field net, or None for random legal play
    field_eps: float = 0.0      # ε of the stochastic field F_σ; 0 is the deterministic net
    draw: str = "uniform"       # the live rung's worlds: "uniform" on the fiber, or "posterior" β^F
    draw_budget: int = DRAW_BUDGET
    convention: Convention | None = None   # the live seat's convention overlay; None is off
    reading: Reading | None = None   # the posterior's likelihood as the net's set; None is F_σ's
    tie: str = "lowest"         # the tie rule among a decision's exact ties: "lowest" or "margin"

    def __post_init__(self) -> None:
        if not 0.0 <= self.field_eps <= 1.0:
            raise ValueError("field_eps must lie in [0, 1]")
        if self.field_eps > 0.0 and self.field is None:
            raise ValueError("field_eps needs a field net: the random field is the dice already")
        if not self.rungs:
            raise ValueError("a player needs at least one rung")
        if self.utility not in ("make", "points"):
            raise ValueError(f"unknown utility {self.utility!r}")
        if self.rungs[0].floor > 1 and self.field is None:
            raise ValueError("rung 0 hands a floored move down only to a field net")
        if self.draw not in DRAWS:
            raise ValueError(f"unknown draw {self.draw!r}")
        if self.draw == "posterior" and (self.level != 0 or self.field is None):
            raise ValueError("the posterior is the field net's, drawn by a one-rung player")
        if self.draw_budget < 1:
            raise ValueError("the draw budget is at least one proposal")
        if self.reading is not None and self.draw != "posterior":
            raise ValueError("a reading is the posterior's likelihood: it needs draw=\"posterior\"")
        if self.convention is not None and self.convention.read and self.draw != "uniform":
            raise ValueError("a convention reader draws from the uniform: the field's posterior "
                             "would model the partner's moves twice")
        if self.tie not in TIES:
            raise ValueError(f"unknown tie rule {self.tie!r}")
        if self.tie == "margin" and self.utility != "make":
            raise ValueError("make-then-margin breaks ties of the make utility")

    @property
    def level(self) -> int:
        return len(self.rungs) - 1

    @property
    def field_coin(self) -> int:
        """F_σ's explore threshold on a 53-bit draw, ⌈ε·2^53⌉: the coin
        explores with probability ε̂ = field_coin / 2^53, which is ε for every
        multiple of 2^-53 (0.5, 0.25, …) and within 2^-53 of it otherwise.
        Scaling by a power of two is exact, so the core computes the same
        integer."""
        return math.ceil(self.field_eps * 2.0 ** 53)

    @property
    def field_eps_exact(self) -> float:
        """ε̂, the probability the coin realizes (exact in a double)."""
        return self.field_coin * 2.0 ** -53


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


Belief = Callable[[Situation, list[World]], list[float]]  # evidence beyond the plays: weights per world


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
    weighs them (the posterior's × external evidence, unnormalized). `outcomes[a][r]`
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
    line: dict[int, tuple["Step", ...]] | None = None   # per root action, the plan's line, when asked


@dataclass(frozen=True)
class Step:
    """One cell of the plan's line where the viewer moves: the plays since
    the root that reach it (the root action first; the other seats' tiles
    among them are the observations the plan branched on to get here), the
    tile the plan plays there, the scenarios in the cell (indices into
    `Analysis.worlds`, ascending), and whether the floor handed the move
    down a rung instead of maximizing. A root action's steps come in the
    trace's depth-first order, the observed tiles at a branch in first-seen
    order."""

    plays: tuple[tuple[int, int], ...]      # (seat, tile), from the root action on
    tile: int
    cell: tuple[int, ...]
    floored: bool


class Walt:
    """A particular Walt: a Config plus a belief constructor."""

    def __init__(self, config: Config, belief: Belief | None = None):
        self.config = config
        self.belief = belief
        self._pi: dict[tuple, int] = {}   # modeled decisions, by (rung, seat, hand, record key)
        self._sets: dict[tuple, int] = {}  # a reading's sets, by (seat, hand, record key)
        self._facs: dict[tuple, float] = {}  # a reading's per-seat factors, by (seat, hand now, record key)
        self.nodes = 0                     # cells evaluated in the last decision
        self.proposals = 0                 # posterior or reader: uniform proposals drawn in the last decision
        self.fallbacks = 0                 # posterior or reader: 1 when the last decision planned over the uniform
        self.sent = 0                      # convention: 1 when the sender's rule chose the last decision's tile

    # -- the player at the table --------------------------------------------

    def decide(self, state: ZebGameState, bid: int) -> tuple[int, dict[int, float]]:
        """Return (tile, action values) for the seat to play in a PLAYING state."""
        inv = self._root(state, bid)
        if isinstance(inv, int):
            return inv, {inv: float("nan")}
        q = inv.action_values()
        return self._choice(q, inv), q

    def analyze(self, state: ZebGameState, bid: int, line: bool = False) -> "Analysis":
        """`decide`, opened up: the same tile and action values, plus where
        each root action sends every live scenario under the plan — and, with
        `line`, the plan's line itself (`Step`s): what the viewer plays at
        each of its cells, and on which observations the plan branches. The
        line is read off the same replay, so asking for it changes nothing
        else. A forced move is not planned (as in `decide`): its analysis
        holds no scenarios."""
        inv = self._root(state, bid)
        if isinstance(inv, int):
            return Analysis(inv, {inv: float("nan")}, [], [], {inv: ()}, {inv: ()} if line else None)
        q = inv.action_values()
        steps: dict[int, list[Step]] | None = {} if line else None
        outcomes = inv.outcomes(steps)
        return Analysis(self._choice(q, inv), q, inv.worlds, inv.weights, outcomes,
                        None if steps is None else {t: tuple(v) for t, v in steps.items()})

    def _choice(self, q: dict[int, float], inv: "_Invocation") -> int:
        """The plan's best tile — or, where the convention's sender gate
        holds, the plan's best among the tiles the rule allows."""
        conv = self.config.convention
        if conv is None or not conv.send:
            return inv.choose(q)
        own = inv.worlds[0][inv.viewer]
        allowed = conv.sends(inv.rules, inv.viewer, inv.bidder, own, inv.root)
        if allowed is None:
            return inv.choose(q)
        self.sent = 1
        return inv.choose({t: v for t, v in q.items() if (allowed >> t) & 1})

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
        self._sets.clear()
        self._facs.clear()
        self.nodes = self.proposals = self.fallbacks = self.sent = 0
        if len(legal) == 1:
            return legal[0]
        return self._invocation(self.config.level, viewer, own, rec, rules, bid, sit.bidder_team, sit)

    # -- the belief --------------------------------------------------------------

    def field_likelihood(self, sit: Situation, worlds: list[World]) -> list[float]:
        """For each world, the probability that the field this one-rung plan
        models — F_σ — plays what the other seats played:

            L(world) = ∏ P_σ(observed tile | that seat's hand in the world, record so far)

        over the modeled seats' past plays, P_σ = ε̂/n + (1 − ε̂)·[the net's
        move] (`Config.field_eps_exact`), or 1/n under the random field. A
        forced play has P_σ = 1. With ε > 0 it is positive on every world in
        the fiber; with ε = 0 it is the 0/1 indicator that the net played
        every observed move. Prior × this, normalized, is the posterior β^F."""
        cfg = self.config
        if cfg.level != 0:
            raise ValueError("the field likelihood is the one-rung plan's: deeper rungs model seats with a solver")
        rules = rules_for(sit.decl_id)
        eps = cfg.field_eps_exact if cfg.field is not None else 1.0
        w = [1.0] * len(worlds)
        for r, seat, hand, rec, tile, n in _hidden_plays(sit, worlds, rules):
            hit = eps < 1.0 and self._modeled_move(-1, seat, hand, rec, rules, sit.bid, sit.bidder_team) == tile
            w[r] *= eps / n + (1.0 - eps) * (1.0 if hit else 0.0)
        return w

    def reading_likelihood(self, sit: Situation, worlds: list[World]) -> list[float]:
        """For each world, the likelihood of the other seats' plays under the
        config's `Reading`: with A the legal tiles the net scores within the
        margin of its best for that seat's hand in the world at that record,
        the product over the discretionary plays (a forced play weighs 1) of
        `Reading.weight` — ε/n + (1 − ε)·[tile ∈ A]·(1 or 1/|A|) — taken seat
        by seat (`_factor`, the product of that seat's plays in play order),
        the seats' factors multiplied in seat order, as Silas's readings
        factorize."""
        cfg = self.config
        if cfg.reading is None or cfg.field is None:
            raise ValueError("a reading reads the field net's sets")
        rules = rules_for(sit.decl_id)
        ev = _evidence(sit, rules)
        out = []
        for hands in worlds:
            p = 1.0
            for seat in range(4):
                if seat != sit.viewer:
                    p *= self._factor(sit, rules, ev, seat, hands[seat])
            out.append(p)
        return out

    def _factor(self, sit: Situation, rules: Rules, ev: list, seat: int, now: int) -> float:
        """A seat's factor of the reading's likelihood, given the hand it
        holds now: the product of `Reading.weight` over its discretionary
        plays, in play order (0 at the first play outside a sharp set)."""
        key = (seat, now, sit.record.key())
        p = self._facs.get(key)
        if p is not None:
            return p
        reading, field = self.config.reading, self.config.field
        assert reading is not None and field is not None
        p, mine = 1.0, sit.played_by(seat)
        for s, rec, tile, before in ev:
            if s != seat:
                continue
            hand = (now | mine) & ~before
            n = rules.legal(hand, rec.led).bit_count()
            if n > 1:
                skey = (seat, hand, rec.key())
                a = self._sets.get(skey)
                if a is None:
                    a = self._sets[skey] = field.allowed(rules, seat, hand, rec, sit.bid, sit.bidder_team,
                                                         reading.margin)
                p *= reading.weight(tile, a, n)
                if p == 0.0:
                    break
        self._facs[key] = p
        return p

    def convention_likelihood(self, sit: Situation, worlds: list[World]) -> list[float]:
        """For each world, 1 if every gated decision the viewer's partner made
        obeyed the convention's reading — replayed with the hand the world
        gives the partner — else 0. Ungated decisions, and every decision of
        the opponents, are not evidence."""
        conv = self.config.convention
        if conv is None:
            raise ValueError("no convention to read")
        rules = rules_for(sit.decl_id)
        partner = (sit.viewer + 2) % 4
        w = [1.0] * len(worlds)
        for r, seat, hand, rec, tile, _ in _hidden_plays(sit, worlds, rules):
            if seat == partner and w[r]:
                allowed = conv.reads(rules, seat, sit.bidder, hand, rec)
                if allowed is not None and not (allowed >> tile) & 1:
                    w[r] = 0.0
        return w

    # -- the ladder ------------------------------------------------------------

    def _invocation(self, k: int, viewer: int, own: int, rec: Record, rules: Rules,
                    bid: int, bidder_team: int, sit: Situation | None = None) -> "_Invocation":
        """`sit` is given at the live rung only: modeled minds weigh their
        scenarios uniformly."""
        rung = self.config.rungs[k]
        # Not the rung: every rung draws the same worlds and tapes at a
        # coordinate, so the ladder iterates one operator (scenario/lean/).
        seed = _mix(self.config.seed, viewer, own, _tape_key(rec), *rec.forbid)
        rng = SplitMix(seed)
        sizes = (rec.hand_size(0), rec.hand_size(1), rec.hand_size(2), rec.hand_size(3))
        worlds, exact = sample_worlds(viewer, own, rec.played, sizes, rec.forbid, rung.n_worlds, rng)
        weights = [1.0] * len(worlds)
        conv = self.config.convention
        reading = self.config.reading
        if sit is not None and reading is not None:
            worlds, weights = self._drawn(sit, worlds, exact, seed, self.reading_likelihood,
                                          reading.likelihood == "uniform" or reading.eps > 0.0, READ_ENUM)
        elif sit is not None and self.config.draw == "posterior":
            worlds, weights = self._drawn(sit, worlds, exact, seed, self.field_likelihood,
                                          self.config.field_coin > 0)
        elif sit is not None and conv is not None and conv.read:
            worlds, weights = self._drawn(sit, worlds, exact, seed, self.convention_likelihood, False)
        if sit is not None and self.belief is not None:
            weights = [w * x for w, x in zip(weights, self.belief(sit, worlds))]
        tapes = [_mix(seed, i) for i in range(len(worlds))]
        floor = 1 if exact else rung.floor
        bidder = sit.bidder if sit is not None else None
        return _Invocation(self, k, viewer, rec, rules, bid, bidder_team, worlds, weights, tapes, floor, bidder)

    def _drawn(self, sit: Situation, uniform: list[World], exact: bool, seed: int,
               likelihood: Belief, coined: bool, cap: int = 0) -> tuple[list[World], list[float]]:
        """The live rung's scenarios under uniform × `likelihood` (the field's
        β^F, the reading's, or the convention reader's filter) and their
        weights, given the uniform ones: the fiber weighted by L normalized
        when `exact`; else, when the fiber holds at most `cap` members, n draws
        with replacement from the whole fiber weighted by L (enumerated, so
        the draws are exact however sharp L is: a coin on `_mix(seed, COIN)`
        scaled by the total, the first member whose running sum exceeds it);
        else n draws by rejection, equally weighted — a proposal is kept when
        a coin falls below its L ≤ 1 (`coined`), or, with no coin, when L is
        positive — or the uniform, when L vanishes on the whole fiber or the
        budget runs out."""
        n = len(uniform)
        if exact:
            like = likelihood(sit, uniform)
            total = sum(like)
            if total > 0:
                return uniform, [x / total for x in like]
            self.fallbacks = 1
            return uniform, [1.0] * n
        cfg, rec = self.config, sit.record
        fiber = Fiber(sit.viewer, uniform[0][sit.viewer], rec.played,
                      tuple(rec.hand_size(s) for s in range(4)), rec.forbid)
        if fiber.size <= cap:
            every = fiber.worlds()
            self.proposals = len(every)
            cum = list(accumulate(likelihood(sit, every)))
            total = cum[-1]
            if total > 0:
                coins = SplitMix(_mix(seed, COIN))
                kept = []
                for _ in range(n):
                    i = bisect_right(cum, coins.random() * total)
                    kept.append(every[i if i < len(every) else bisect_left(cum, total)])
                return kept, [1.0] * n
            self.fallbacks = 1
            return uniform, [1.0] * n
        rng = SplitMix(seed)
        coins = SplitMix(_mix(seed, COIN)) if coined else None
        kept: list[World] = []
        while len(kept) < n and self.proposals < cfg.draw_budget:
            world = fiber.draw(rng)
            self.proposals += 1
            u = coins.random() if coins is not None else 0.0
            if u < likelihood(sit, [world])[0]:
                kept.append(world)
        if len(kept) < n:
            self.fallbacks = 1
            return uniform, [1.0] * n
        return kept, [1.0] * n

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
                tile = inv.choose(inv.action_values())
            self._pi[key] = tile
        return tile


def _hidden_plays(sit: Situation, worlds: list[World], rules: Rules):
    """The other seats' discretionary plays as each world explains them:
    (world, seat, the seat's hand then, record before the play, tile, legal
    count n > 1), replayed from the first lead with each seat holding the
    hand the world implies it was dealt."""
    if not sit.plays:
        return
    full = [[hands[s] | sit.played_by(s) for s in range(4)] for hands in worlds]
    rec = empty_record(sit.plays[0][0])
    for seat, tile in sit.plays:
        if seat != sit.viewer:
            for r, hands in enumerate(full):
                n = rules.legal(hands[seat], rec.led).bit_count()
                if n > 1:
                    yield r, seat, hands[seat], rec, tile, n
                hands[seat] &= ~bit(tile)
        rec = rec.advance(tile, rules)


def _evidence(sit: Situation, rules: Rules) -> list[tuple[int, Record, int, int]]:
    """The other seats' plays from the viewer's chair: (seat, the record
    before the play, the tile, the tiles that seat had played before it)."""
    out: list[tuple[int, Record, int, int]] = []
    if not sit.plays:
        return out
    rec, before = empty_record(sit.plays[0][0]), [0, 0, 0, 0]
    for seat, tile in sit.plays:
        if seat != sit.viewer:
            out.append((seat, rec, tile, before[seat]))
        before[seat] |= bit(tile)
        rec = rec.advance(tile, rules)
    return out


def _best(q: dict[int, float], maximize: bool) -> int:
    """The best move; among equal-valued moves, the lowest tile id."""
    items = sorted(q.items())
    if maximize:
        return max(items, key=lambda kv: kv[1])[0]
    return min(items, key=lambda kv: kv[1])[0]


class _Invocation:
    """One planning problem: a viewer, a rung, a sample, and its memo."""

    def __init__(self, walt: Walt, k: int, viewer: int, root: Record, rules: Rules,
                 bid: int, bidder_team: int, worlds: list[World],
                 weights: list[float], tapes: list[int], floor: int, bidder: int | None = None):
        self.walt = walt
        self.bidder = bidder    # the declaring seat, known at the live rung (the convention's sender reads it)
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
        self.floor = floor
        self.root_plays = root.plays_so_far()
        self.memo: dict[tuple, float] = {}

    # -- the decision -----------------------------------------------------------

    def choose(self, q: dict[int, float]) -> int:
        """The decision among root actions valued `q`: the best, and among
        exact ties the tie rule's pick (`Config.tie`)."""
        tile = _best(q, self.maximize)
        if self.walt.config.tie == "lowest":
            return tile
        tied = [t for t, v in sorted(q.items()) if v == q[tile]]
        if len(tied) < 2:
            return tile
        return _best({t: self.mean_margin(t) for t in tied}, self.maximize)

    def mean_margin(self, t: int) -> float:
        """M(t) = Σ_r w_r·m_r(t): the weighted bidder-team margin the live
        scenarios reach when the root plays t and the plan continues (one
        root action's `outcomes`), unnormalized — every action shares the mass."""
        leaf = [0.0] * len(self.worlds)
        self._trace(self.root.advance(t, self.rules), (1 << len(self.worlds)) - 1, leaf)
        return sum(self.weights[r] * leaf[r] for r in range(len(self.worlds)))

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

    def outcomes(self, line: dict[int, list[Step]] | None = None) -> dict[int, tuple[float, ...]]:
        """Per legal root action, the leaf value each scenario reaches under
        the plan. A replay of the solved tree, not a second solve: every cell
        it visits is already in the memo, and the policy subtree it walks
        holds at most one path per scenario. Given `line`, each root
        action's viewer cells are recorded into it, as `Step`s, on the way."""
        own = self.worlds[0][self.viewer] & ~self.root.played
        everyone = (1 << len(self.worlds)) - 1
        out = {}
        for t in tiles_of(self.rules.legal(own, self.root.led)):
            leaf = [0.0] * len(self.worlds)
            steps = None
            if line is not None:
                steps = line[t] = []
            self._trace(self.root.advance(t, self.rules), everyone, leaf, steps, ((self.viewer, t),))
            out[t] = tuple(leaf)
        return out

    def _trace(self, rec: Record, alive: int, leaf: list[float],
               steps: list[Step] | None = None, path: tuple[tuple[int, int], ...] = ()) -> None:
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
            floored = self._floored(alive)
            if floored:
                tile = self._handed_down(rec)
            else:
                hand = self.worlds[0][seat] & ~rec.played
                q = {t: self.value(rec.advance(t, self.rules), alive)
                     for t in tiles_of(self.rules.legal(hand, rec.led))}
                tile = _best(q, self.maximize)
            if steps is not None:
                steps.append(Step(path, tile, tuple(_bits(alive)), floored))
                path = path + ((seat, tile),)
            self._trace(rec.advance(tile, self.rules), alive, leaf, steps, path)
            return
        for t, cell in self._partition(seat, rec, alive).items():
            self._trace(rec.advance(t, self.rules), cell, leaf, steps,
                        path + ((seat, t),) if steps is not None else path)

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
        """The modeled seat's tile in scenario r. At rung 0 the field is the
        dice, or F_σ: the dice when scenario r's coin at this record explores,
        else the net (ε = 0 never draws the coin)."""
        cfg = self.walt.config
        if self.k == 0 and (cfg.field is None or self._explores(r, rec)):
            return self._dice(r, rec, self.rules.legal(hand, rec.led))
        return self.walt._modeled_move(
            self.k - 1, seat, hand, rec, self.rules, self.bid, self.bidder_team,
        )

    def _explores(self, r: int, rec: Record) -> bool:
        """F_σ's coin in scenario r at this record: the top 53 bits of one more
        SplitMix step past the dice's draw, below ⌈ε·2^53⌉. It reads the
        tape and the record only, never the hand, so it is independent of
        the dice's uniform tile up to the hash."""
        coin = self.walt.config.field_coin
        return coin > 0 and _splitmix(_splitmix(self.tapes[r] ^ _tape_key(rec))) >> 11 < coin

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
