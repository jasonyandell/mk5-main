# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""Zeb's reason report (#124): why Zeb plays its tile and not another, in Silas.

One decision, two tiles: Zeb's (the plan's choice) and a named one (the
book's, say). The report reads `Walt.analyze` — every live scenario's ending
under the plan, for every root tile, and the plan's line — and says:

- **The line.** What the plan does next after each tile, and on how many
  distinct observations it branches.
- **The split.** The worlds where Zeb's tile succeeds for the mover and the
  named one fails (side A), and the reverse (side B), with exact counts and
  weight shares. Success is the mover's: the contract made for the bidding
  side, set for the defenders.
- **What must be true, in Silas.** A declared, finite space of Silas queries
  over the hidden hands (`space`: holdings, voids, the open trick's seats
  still to play, count at risk, the book prelude's nouns as holdings; and
  the conjunction and disjunction of every two), each evaluated by Silas on
  the decision's belief — Zeb's own scenarios and weights, labeled
  `sampled(n, seed)` — restricted to each side. The facts whose masses on
  the two sides differ most are the reason, with the share of each side
  they explain (coverage). Each comes with its exact probability under the
  uniform over the fiber, where Silas can resolve it.
- **Found, or not.** A reason must separate the sides by at least
  `SEPARATION` (1/2) and by more than the space's best on `RELABELINGS`
  random relabelings of the same scenarios (its chance level), both set
  before any position was looked at. Below that the report says *no reason
  found* and shows the best candidates anyway.
- **A counterexample world on each side**, as hands, with the plan's path
  through it after each tile.
- **The noise bar.** The same decision re-planned at another seed and one
  rung stronger (`noise_configs`); the bar is the furthest the pair's price
  moves. Silas's three-valued comparison decides: the probability that Zeb's
  tile beats the other in a world (ties halved), against one half, with half
  the bar as the band. Inside the bar the report says *indistinguishable*
  and gives no reason.
- **Ties.** An exact tie is *indifferent — no reason* (#120).

Every number carries its authority. Counts, shares and side masses are read
off one sample (`sampled(n, seed)`, n the live scenarios, seed the plan's);
priors over the fiber are `exact`. The report reads pmake, so the plan must
use the `make` utility.

    python -u -m scenario.reason --help
"""
from __future__ import annotations

import argparse
import dataclasses
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

import numpy as np
from scenario_core import SilasAuthority, SilasBelief, SilasError, SilasQuery, silas_compare

from forge.zeb.types import ZebGameState

from scenario.planner import HAND_TOTAL, Analysis, Belief, Config, Rung, Situation, Step, Walt, situation_from_state
from scenario.record import Record, remaining_hand
from scenario.rules import ALL_TILES, bit, rules_for, tiles_of
from scenario.silas import frame

BUDGET = 10_000_000          # Silas work units per summary
PRIOR_CAP = 200_000          # points of an exact prior's resolving belief
BAR_GRAIN = 10 ** 6          # the bar is rounded up to a multiple of 1/BAR_GRAIN
STRONGER = Rung(8, 4)        # the rung added below a one-rung plan (the arena's m8, g4)
TRUTH_BLOCK = 512            # scenarios per Silas pass reading each fact world by world (2^i weights, exact in f64)

# Set before any position was looked at (2026-10-13). A reason separates the
# two columns by at least SEPARATION, |P(fact | one) − P(fact | two)| ≥ 1/2,
# and by more than the space's best on every one of RELABELINGS random
# relabelings of the same scenarios into columns of the same sizes (seeded
# CHANCE_SEED): the separation a space this size reaches by chance on sides
# this small (one fact against one-fact relabelings, two against two). The
# reason is the best single fact that passes, or the best two-fact
# explanation that passes when it separates strictly more.
SEPARATION = Fraction(1, 2)
RELABELINGS = 20
CHANCE_SEED = 124

INDIFFERENT = "indifferent"            # an exact tie: no reason (#120)
INDISTINGUISHABLE = "indistinguishable"  # the price lies inside the noise bar: no reason
REASON = "reason"

ONE_FACT = "one fact"                  # a single fact of the space reaches the threshold
TWO_FACTS = "two facts"                # a conjunction or disjunction of two does, and separates more
NO_REASON = "no reason found"          # none does: the best candidates are shown anyway


# -- names ----------------------------------------------------------------------------


def tile_name(t: int) -> str:
    """Silas's spelling of a tile id: high pip first (`6-5`)."""
    high = max(h for h in range(7) if h * (h + 1) // 2 <= t)
    return f"{high}-{t - high * (high + 1) // 2}"


def context_name(k: int) -> str:
    """Zeb's led suit k as a Silas context: `q0`..`q6`, `q*` for the called suit."""
    return "q*" if k == 7 else f"q{k}"


_SUIT = ("blanks", "ones", "deuces", "treys", "fours", "fives", "sixes", "trump")


def _hands(world, viewer: int) -> str:
    return "; ".join(
        f"S{s}{' (viewer)' if s == viewer else ''}: " + " ".join(tile_name(t) for t in sorted(tiles_of(world[s]), reverse=True))
        for s in range(4)
    )


def _plays(plays: Sequence[tuple[int, int]]) -> str:
    return " · ".join(f"S{s} {tile_name(t)}" for s, t in plays)


def _num(x: Fraction) -> str:
    if x.denominator <= 10_000:
        return f"{x.numerator}/{x.denominator} ≈ {float(x):.3f}" if x.denominator > 1 else str(x.numerator)
    return f"≈ {float(x):.4f}"


# -- the candidate space ----------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One Silas query of the declared space."""

    text: str     # the query, which is its name
    gloss: str    # the same in words
    reads: int    # the pool tiles it reads (the exact prior resolves these)
    atoms: int    # its size, for breaking ties toward the simpler fact
    scope: str = ""        # the same fact as one clause, for composing two (`two_fact`)
    book: bool = False     # it calls the book prelude (`(include book)`)
    family: str = ""       # the family of the declared space that generated it


def _fact(roles: str, body: str, gloss: str, reads: int, atoms: int, family: str, book: bool = False) -> Candidate:
    """A candidate written as one clause: `body` under its own roles."""
    scope = f"(some (roles {roles}) {body})" if roles else body
    head = "(include book) " if book else ""
    return Candidate(f"{head}(query (roles) (out) (pattern {scope}))", gloss, reads, atoms, scope, book, family)


def candidates(decl_id: int, viewer: int, own: int, played: int, sizes: Sequence[int]) -> list[Candidate]:
    """The first declared space, holdings and voids, at a decision of seat
    `viewer` holding `own`:

    1. `holds`: seat X holds t — X a hidden seat, t a hidden tile — and an
       opponent (either) holds t;
    2. `holds with`: X holds t and another tile of context q — q a context
       t follows, with another hidden tile in it — and the same for an
       opponent (one opponent holding both);
    3. `void`: X holds no tile of context q, and an opponent is void in q —
       q a context with a hidden tile in it.

    Seats are named absolutely (`S1`); the opponent forms name the viewer's
    opponents by role (`opponent me foe`). `space` adds the rest."""
    rules = rules_for(decl_id)
    hidden = ALL_TILES & ~played & ~own
    seats = [s for s in range(4) if s != viewer and sizes[s] > 0]
    contexts = [k for k in range(8) if rules.follow_bits[k] & hidden]
    who = [(f"S{s}", f"S{s} ({'partner' if s == (viewer + 2) % 4 else 'opponent'})", "") for s in seats]
    who.append(("foe", "an opponent", "(viewer me) (opponent me foe) "))
    roles = {"foe": "(chair me) (chair foe)"}
    out: list[Candidate] = []

    def fact(holder: str, extra_roles: str, body: str, gloss: str, reads: int, atoms: int, family: str) -> Candidate:
        r = " ".join(x for x in (roles.get(holder, ""), extra_roles) if x)
        scope = f"(some (roles {r}) {body})" if r else body
        return Candidate(f"(query (roles{' ' + r if r else ''}) (out) (pattern {body}))",
                         gloss, reads, atoms, scope, False, family)

    for holder, said, lead in who:
        base = 2 if holder == "foe" else 0
        for t in tiles_of(hidden):
            name = tile_name(t)
            out.append(fact(holder, "", f"{lead}(holds {holder} {name})", f"{said} holds {name}", bit(t), base + 1,
                            "holds"))
    for holder, said, lead in who:
        base = 2 if holder == "foe" else 0
        for t in tiles_of(hidden):
            name = tile_name(t)
            for k in contexts:
                if not (rules.follow_bits[k] >> t) & 1 or not (rules.follow_bits[k] & hidden & ~bit(t)):
                    continue
                # `other` is bound by the suit before the world is read, so an
                # exact prior resolves only the suit's tiles.
                body = (f"{lead}(holds {holder} {name}) (in other {context_name(k)}) "
                        f"(not (tile other {name})) (holds {holder} other)")
                out.append(fact(holder, "(domino other)", body, f"{said} holds {name} and another of {_SUIT[k]}",
                                bit(t) | (rules.follow_bits[k] & hidden), base + 4, "holds with"))
    for holder, said, lead in who:
        base = 2 if holder == "foe" else 0
        for k in contexts:
            out.append(fact(holder, "", f"{lead}(void {holder} {context_name(k)})", f"{said} is void in {_SUIT[k]}",
                            rules.follow_bits[k] & hidden, base + 1, "void"))
    return out


def _yet_to_play(viewer: int, rec: Record) -> list[int]:
    """The seats still to play to the open trick after the viewer, in order."""
    return [(viewer + i) % 4 for i in range(1, 4 - len(rec.trick))]


def _seat_said(s: int, viewer: int, after: Sequence[int]) -> str:
    side = "partner" if s == (viewer + 2) % 4 else "opponent"
    if len(after) == 1:
        side += ", the last to play"
    elif s == after[0]:
        side += ", next to play"
    elif s == after[-1]:
        side += ", the last to play"
    return f"S{s} ({side})"


def position_facts(decl_id: int, viewer: int, own: int, rec: Record, tiles: Sequence[int]) -> list[Candidate]:
    """The second declared space: the open trick and the seats still to play
    to it, each fact a Silas query over the world. `tiles` are the viewer's
    tiles the facts are relative to (the report's pair); the viewer's legal
    count tiles are added. With S a seat still to play after the viewer:

    4. `position`: following, S holds a tile of the led suit; S is void in
       the led suit and holds trump; S can beat the trick as it stands (for
       the partner: the partner can still win this trick); S can take the
       trick after the viewer plays t (S may play a tile, following suit or
       void in it, that beats both t and the trick's winner). Leading, the
       suit is the one t leads: S can beat t led; S is void in t's suit and
       holds trump; an opponent can beat t led.
    5. `count at risk`: a count tile the viewer could give (`own-legal`,
       `count-tile`) is capturable by the next seat; and the tile-relative
       facts of 4 for each legal count tile.

    Nothing else is generated; a seat that has played to the trick is not
    asked about."""
    rules = rules_for(decl_id)
    hidden = ALL_TILES & ~rec.played & ~own
    trumps = rules.follow_bits[7]
    after = _yet_to_play(viewer, rec)
    if not after:
        return []
    counted = [t for t in tiles_of(rules.legal(own, rec.led)) if rules.count[t]]
    relative = list(dict.fromkeys([*tiles, *counted]))
    nxt = f"S{after[0]}"
    out: list[Candidate] = []

    def playable(s: str) -> str:   # s may play x to this trick: x follows q, or s is void in q
        return f"(none (roles) (not (in x q)) (not (void {s} q)))"

    def family(t: int) -> str:
        return "count at risk" if t in counted else "position"

    if rec.trick:
        k = rules.led_suit[rec.led]
        reads = (rules.follow_bits[k] | trumps) & hidden
        win = "(led-context q) (current-winner w) (trick-play w win)"
        for s in after:
            said, S = _seat_said(s, viewer, after), f"S{s}"
            out.append(_fact("(context q) (domino x)", f"(led-context q) (in x q) (holds {S} x)",
                             f"{said} holds a tile of the led suit ({_SUIT[k]})", rules.follow_bits[k] & hidden, 2,
                             "position"))
            if k != 7:
                out.append(_fact("(context q) (context tr) (domino x)",
                                 f"(led-context q) (trump tr) (void {S} q) (in x tr) (holds {S} x)",
                                 f"{said} is void in the led suit ({_SUIT[k]}) and holds trump", reads, 3,
                                 "position"))
            can = "can still win this trick" if s == (viewer + 2) % 4 else "can beat the trick as it stands"
            out.append(_fact("(context q) (chair w) (domino win) (domino x)",
                             f"{win} (holds {S} x) {playable(S)} (beats x win q)", f"{said} {can}", reads, 3,
                             "position"))
            for t in relative:
                name = tile_name(t)
                out.append(_fact("(context q) (chair w) (domino win) (domino x)",
                                 f"{win} (holds {S} x) {playable(S)} (beats x {name} q) (beats x win q)",
                                 f"{said} can take the trick after I play {name}", reads, 4, family(t)))
        if counted:
            out.append(_fact("(domino c) (context q) (chair w) (domino win) (domino x)",
                             f"(own-legal c) (count-tile c) {win} (holds {nxt} x) {playable(nxt)} "
                             "(beats x c q) (beats x win q)",
                             f"a count tile I could give is capturable by the next seat ({nxt})", reads, 5,
                             "count at risk", book=True))
        return out
    for t in relative:
        name, k = tile_name(t), rules.led_suit[t]
        reads = (rules.follow_bits[k] | trumps) & hidden
        for s in after:
            said, S = _seat_said(s, viewer, after), f"S{s}"
            out.append(_fact("(context q) (domino x)",
                             f"(leads-context {name} q) (holds {S} x) {playable(S)} (beats x {name} q)",
                             f"{said} can beat {name} led", reads, 3, family(t)))
            if k != 7:
                out.append(_fact("(context q) (context tr) (domino x)",
                                 f"(leads-context {name} q) (trump tr) (void {S} q) (in x tr) (holds {S} x)",
                                 f"{said} is void in the suit {name} leads ({_SUIT[k]}) and holds trump", reads, 4,
                                 "position"))
        out.append(_fact("(chair me) (chair s) (context q) (domino x)",
                         f"(viewer me) (opponent me s) (leads-context {name} q) (holds s x) {playable('s')} "
                         f"(beats x {name} q)", f"an opponent can beat {name} led", reads, 5, family(t)))
    if counted:
        out.append(_fact("(domino c) (context q) (domino x)",
                         f"(own-legal c) (count-tile c) (leads-context c q) (holds {nxt} x) {playable(nxt)} "
                         "(beats x c q)", f"a count tile I could lead is capturable by the next seat ({nxt})",
                         hidden, 5, "count at risk", book=True))
    return out


DOUBLES = sum(bit(h * (h + 1) // 2 + h) for h in range(7))


def prelude_facts(decl_id: int, viewer: int, own: int, played: int, sizes: Sequence[int]) -> list[Candidate]:
    """The third declared space: the book prelude's nouns (`silas/lib/book.silas`)
    as holdings. A prelude concept that reads only the frame (the safe trick,
    can-take-it, may-yet-trump, the dead double, can-lose) is the same in
    every world of one decision, so it cannot separate two sets of them; the
    prelude enters the space as what a seat holds:

    6. `prelude`: X holds a count tile (`count-tile`), a ten, a trump
       (`trump-tile`), the boss trump (`boss-trump`), a trump none of the
       viewer's beats, a double, the top live tile of suit q (`boss`) — X a
       hidden seat or an opponent (either).

    A trump concept is a scope, read after the atoms around it, so the
    called suit's `in` binds the tile first and an exact prior resolves
    only the called suit (under no trump the fact is false and drops out)."""
    rules = rules_for(decl_id)
    hidden = ALL_TILES & ~played & ~own
    trumps = rules.follow_bits[7] & hidden
    counts = sum(bit(t) for t in tiles_of(hidden) if rules.count[t])
    tens = sum(bit(t) for t in tiles_of(hidden) if rules.count[t] == 10)
    seats = [s for s in range(4) if s != viewer and sizes[s] > 0]
    who = [(f"S{s}", "", "", f"S{s} ({'partner' if s == (viewer + 2) % 4 else 'opponent'})", 0) for s in seats]
    who.append(("foe", "(chair me) (chair foe) ", "(viewer me) (opponent me foe) ", "an opponent", 2))
    out: list[Candidate] = []
    for holder, roles, lead, said, base in who:
        def add(concept: str, gloss: str, reads: int, atoms: int) -> None:
            if reads:
                out.append(_fact(f"{roles}(domino d)", f"{lead}{concept} (holds {holder} d)", f"{said} {gloss}",
                                 reads, base + atoms, "prelude", book=True))
        add("(count-tile d)", "holds a count tile", counts, 2)
        add("(count d 10)", "holds a ten", tens, 2)
        add("(in d q*) (trump-tile d)", "holds a trump", trumps, 2)
        add("(in d q*) (boss-trump d)", "holds the boss trump", trumps, 2)
        add("(in d q*) (trump-tile d) "
            "(none (roles (domino e)) (mine e) (some (roles (context t)) (trump t) (beats e d t)))",
            "holds a trump none of mine beats", trumps, 3)
        add("(double d)", "holds a double", DOUBLES & hidden, 2)
        for k in range(7):
            add(f"(boss d {context_name(k)})", f"holds the top live tile of {_SUIT[k]}",
                rules.follow_bits[k] & hidden & ~trumps, 2)
    return out


def space(sit: Situation, own: int, tiles: Sequence[int]) -> list[Candidate]:
    """The declared space of single facts at a decision: `candidates`,
    `position_facts` (relative to `tiles`) and `prelude_facts`. The two-fact
    candidates are the conjunction and the disjunction of every two of them
    (`two_fact`). Declared up front; nothing else is searched."""
    rec = sit.record
    sizes = [rec.hand_size(s) for s in range(4)]
    return (candidates(sit.decl_id, sit.viewer, own, rec.played, sizes)
            + position_facts(sit.decl_id, sit.viewer, own, rec, tiles)
            + prelude_facts(sit.decl_id, sit.viewer, own, rec.played, sizes))


def two_fact(a: Candidate, b: Candidate, op: str) -> Candidate:
    """`a and b` (one pattern holding both clauses) or `a or b` (a pattern
    each, Silas's union of cases), as one Silas query. Each fact keeps its
    roles in its own `some` scope, so the two share none: a scope's roles
    are distinct from one another, not from the roles around them."""
    head = "(include book) " if a.book or b.book else ""
    body = f"(pattern {a.scope} {b.scope})" if op == "and" else f"(pattern {a.scope}) (pattern {b.scope})"
    return Candidate(f"{head}(query (roles) (out) {body})", f"{a.gloss} {op} {b.gloss}", a.reads | b.reads,
                     a.atoms + b.atoms, "", a.book or b.book, op)


# -- the report -------------------------------------------------------------------------


@dataclass(frozen=True)
class Fact:
    """A candidate, evaluated: its probability on each side (Silas
    estimates, sampled), how many worlds of each side it holds in, its
    probability over the whole sample, and its exact prior over the fiber
    (None when Silas refuses to resolve it within the cap)."""

    candidate: Candidate
    on_a: Fraction            # on the first of `Reason.columns` (side A, unless A is empty)
    on_b: Fraction            # on the second (side B, unless B is empty)
    count_a: int
    count_b: int
    overall: Fraction
    prior: Fraction | None
    authority: str
    prior_authority: str | None
    alike: tuple[str, ...] = ()   # the other single facts of the space that read the same on both columns
                                  # ("not ..." when as its negation): the same split, named otherwise

    @property
    def separation(self) -> Fraction:
        return abs(self.on_a - self.on_b)

    @property
    def coverage(self) -> tuple[Fraction, Fraction]:
        """The share of each column the fact explains, read toward the column
        where it holds more: the share of that column where it holds, and of
        the other where it fails. Their sum less one is the separation."""
        if self.on_a >= self.on_b:
            return self.on_a, 1 - self.on_b
        return 1 - self.on_a, self.on_b


@dataclass(frozen=True)
class Example:
    """One world of a side, as hands, and the plan's path through it after each tile."""

    index: int
    world: tuple[int, int, int, int]
    weight: float
    paths: dict[int, tuple[tuple[int, int], ...]]
    margins: dict[int, float]
    fact_holds: bool | None


@dataclass(frozen=True)
class Arm:
    """One re-plan of the noise bar: its config, label, and the pair's price
    (Zeb's tile minus the other, the mover's pmake)."""

    label: str
    config: Config
    q: dict[int, float]
    price: float


@dataclass(frozen=True)
class Reason:
    decl_id: int
    viewer: int
    bidder: int
    bid: int
    plays: tuple[tuple[int, int], ...]
    own: int
    config: Config
    tile: int                 # Zeb's
    other: int                # the named tile
    q: dict[int, float]
    price: float              # the mover's pmake, Zeb's tile minus the other
    authority: str            # sampled(n, seed): the plan's sample
    n_worlds: int
    side_a: tuple[int, ...]   # Zeb's tile succeeds, the other fails
    side_b: tuple[int, ...]   # the other succeeds, Zeb's tile fails
    share_a: Fraction
    share_b: Fraction
    columns: tuple[str, str]  # the two sets the facts separate: ("A", "B"), or, when one side is empty,
                              # the other against "the rest" (the scenarios where the tiles agree)
    facts: tuple[Fact, ...]   # the top single facts, best first (empty when the price gives no reason)
    pairs: tuple[Fact, ...]   # the top two-fact explanations, best first
    found: str                # ONE_FACT, TWO_FACTS or NO_REASON ("" when the price gives no reason)
    chance: tuple[float, float]   # the space's best single and two-fact separation over the relabelings
    searched: tuple[int, int, int, int]   # single facts declared, distinct on the sample, refused; pairs kept
    examples: tuple[Example | None, Example | None]
    lines: dict[int, tuple[Step, ...]]
    weights: tuple[float, ...]
    arms: tuple[Arm, ...]
    bar: Fraction
    beats: Fraction           # P(Zeb's tile beats the other in a world, ties halved), sampled
    verdict: str              # REASON, INDISTINGUISHABLE or INDIFFERENT
    compare: str              # Silas's verdict against the bar: better / worse / indistinguishable


def noise_configs(config: Config) -> list[tuple[str, Config]]:
    """The noise bar's re-plans: the same plan at the next seed, and one rung
    stronger — a copy of the bottom modeled rung added beneath the ladder,
    or `STRONGER` (n8, horizon 4, the arena's inner rung) under a one-rung
    plan, so the modeled seats plan instead of playing the field."""
    if config.draw != "uniform":
        raise ValueError("a stronger rung cannot draw from the field's posterior (a one-rung draw): name the bar")
    inner = config.rungs[0] if config.level >= 1 else STRONGER
    return [
        (f"seed {config.seed + 1}", dataclasses.replace(config, seed=config.seed + 1)),
        (f"one rung stronger (k{config.level + 1}, added rung n{inner.n_worlds}, horizon {inner.horizon})",
         dataclasses.replace(config, rungs=(inner,) + config.rungs)),
    ]


def _walt(config: Config, belief: Belief | None, compiled: bool):
    if compiled:
        from scenario.compiled import CompiledWalt
        return CompiledWalt(config, belief=belief)
    return Walt(config, belief=belief)


def describe(config: Config) -> str:
    rungs = ", ".join(f"n{r.n_worlds}/h{'∞' if r.horizon is None else r.horizon}/f{r.floor}" for r in config.rungs)
    field = "dice" if config.field is None else ("net" if config.field_eps == 0 else f"net ε={config.field_eps:g}")
    return f"k{config.level} [{rungs}], {config.utility}, seed {config.seed}, field {field}, draw {config.draw}"


def reason(
    state: ZebGameState, bid: int, other: int, config: Config, *,
    bar: Sequence[tuple[str, Config]] | None = None, belief: Belief | None = None,
    compiled: bool = True, top: int = 5,
) -> Reason:
    """The reason report for the decision at `state` under contract `bid`:
    Zeb's tile (the plan's choice under `config`) against `other`. `bar`
    names the noise bar's re-plans (default `noise_configs(config)`)."""
    if config.utility != "make":
        raise ValueError("the reason report reads pmake: plan under the make utility")
    walt = _walt(config, belief, compiled)
    a: Analysis = walt.analyze(state, bid, line=True)
    if not a.worlds:
        raise ValueError("a forced move has no reason to report")
    if other not in a.q:
        raise ValueError(f"{tile_name(other)} is not a legal tile here")
    tile = a.tile
    if other == tile:
        raise ValueError("the named tile is Zeb's own")
    sit = situation_from_state(state, bid)
    viewer, own = sit.viewer, remaining_hand(state, sit.viewer)
    sign = 1 if viewer % 2 == sit.bidder % 2 else -1
    threshold = 2 * bid - HAND_TOTAL

    def succeeds(t: int) -> list[bool]:
        return [(o >= threshold) == (sign > 0) for o in a.outcomes[t]]

    zs, os_ = succeeds(tile), succeeds(other)
    n = len(a.worlds)
    side_a = tuple(r for r in range(n) if zs[r] and not os_[r])
    side_b = tuple(r for r in range(n) if os_[r] and not zs[r])
    w = [Fraction(x) for x in a.weights]          # each float read exactly
    total = sum(w)
    share_a = sum((w[r] for r in side_a), Fraction(0)) / total
    share_b = sum((w[r] for r in side_b), Fraction(0)) / total
    price = sign * (a.q[tile] - a.q[other])

    sampled = SilasAuthority.sampled(n, config.seed)
    f = frame(sit, own)
    whole = SilasBelief.from_worlds(f, a.worlds, a.weights, sampled)
    left, right = _paired(whole, a, zs, os_, tile, other, config)

    arms = []
    for label, cfg in (noise_configs(config) if bar is None else bar):
        _, q = _walt(cfg, belief, compiled).decide(state, bid)
        arms.append(Arm(label, cfg, q, sign * (q[tile] - q[other])))
    spread = max((abs(arm.price - price) for arm in arms), default=0.0)
    band = Fraction(math.ceil(spread * BAR_GRAIN), BAR_GRAIN)
    if silas_compare(left, right, Fraction(0)) == "indistinguishable":
        verdict, compared = INDIFFERENT, "indistinguishable"
    else:
        compared = silas_compare(left, right, band / 2)
        verdict = REASON if compared != "indistinguishable" else INDISTINGUISHABLE

    found = ""
    facts: tuple[Fact, ...] = ()
    pairs: tuple[Fact, ...] = ()
    chance, searched = (0.0, 0.0), (0, 0, 0, 0)
    examples: tuple[Example | None, Example | None] = (None, None)
    columns = ("A", "B")
    if verdict == REASON:
        one, two = side_a, side_b
        if not side_a or not side_b:   # one tile dominates: set its side against the worlds where they agree
            agree = tuple(r for r in range(n) if zs[r] == os_[r])
            one, two, columns = (side_a, agree, ("A", "the rest")) if side_a else (agree, side_b, ("the rest", "B"))
        found, facts, pairs, chance, searched = _explain(f, a, one, two, sampled, sit, own, (tile, other), top)
        best = pairs[0] if found == TWO_FACTS else (facts[0] if facts else None)
        examples = (_example(f, a, side_a, best, True, sampled, (tile, other), viewer),
                    _example(f, a, side_b, best, False, sampled, (tile, other), viewer))
    return Reason(
        decl_id=sit.decl_id, viewer=viewer, bidder=sit.bidder, bid=bid, plays=sit.plays, own=own,
        config=config, tile=tile, other=other, q=dict(a.q), price=price, authority=sampled.label,
        n_worlds=n, side_a=side_a, side_b=side_b, share_a=share_a, share_b=share_b, columns=columns,
        facts=facts, pairs=pairs, found=found, chance=chance, searched=searched,
        examples=examples, lines=a.line or {}, weights=tuple(a.weights), arms=tuple(arms),
        bar=band, beats=left.value - right.value + Fraction(1, 2), verdict=verdict, compare=compared,
    )


@dataclass(frozen=True)
class Scenarios:
    """A decision's scenarios and their weights, read off a pour rather than
    a plan (`scenario.metasplat`): what the fact search reads of an Analysis."""

    worlds: list[tuple[int, int, int, int]]
    weights: list[float]


def explain(sit: Situation, own: int, cell: Scenarios, one: Sequence[int], two: Sequence[int],
            tiles: tuple[int, int], authority, top: int = 5):
    """The report's separating-fact search on any two sets of a decision's
    scenarios — the same declared space, threshold and chance level as
    `reason` — without re-planning. Returns (found, facts, pairs, chance,
    searched)."""
    return _explain(frame(sit, own), cell, one, two, authority, sit, own, tiles, top)


def _model(config: Config, what: str) -> str:
    return f"zeb plan ({describe(config)}): {what}"


def _evidence(belief, a: Analysis, score: Sequence[float], model: str):
    """Silas's evidence probability of a per-scenario likelihood on the
    decision's belief. A world drawn twice gets its copies' weighted mean,
    which is exact unless the copies' endings differ (then a double)."""
    copies: dict[tuple, list[tuple[float, float]]] = {}
    for world, w, s in zip(a.worlds, a.weights, score):
        copies.setdefault(tuple(world), []).append((w, s))
    like = {k: c[0][1] if len({s for _, s in c}) == 1 else sum(w * s for w, s in c) / sum(w for w, _ in c)
            for k, c in copies.items()}
    _, estimate = belief.condition_likelihood("evidence", model, list(like), list(like.values()))
    return estimate


def _paired(belief, a: Analysis, zs, os_, tile: int, other: int, config: Config):
    """Two Silas estimates whose difference is P(Zeb's tile beats the other in
    a world, a tie counting one half) − 1/2, which is half the pair's price:
    that probability (the evidence of the likelihood (1 + [Zeb's succeeds] −
    [the other succeeds]) / 2) and the constant 1/2 — or, when the other
    beats Zeb's tile in every world (a zero-mass evidence Silas refuses), 1/2
    and the reverse probability."""
    score = [(1 + z - o) / 2 for z, o in zip(zs, os_)]
    half = _evidence(belief, a, [0.5] * len(score), _model(config, "the constant 1/2"))
    if any(score):
        name = f"{tile_name(tile)} beats {tile_name(other)}, ties halved"
        return _evidence(belief, a, score, _model(config, name)), half
    name = f"{tile_name(other)} beats {tile_name(tile)}, ties halved"
    return half, _evidence(belief, a, [1 - s for s in score], _model(config, name))


def _side(f, a: Analysis, side: Sequence[int], authority, unit: bool = False):
    return SilasBelief.from_worlds(f, [a.worlds[r] for r in side],
                                   [1.0] * len(side) if unit else [a.weights[r] for r in side], authority)


def _truths(f, a: Analysis, facts: Sequence[Candidate], authority) -> tuple[list[tuple[Candidate, int]], int]:
    """Each fact's truth in every scenario, read by Silas. Over a block of m
    scenarios weighted 2^i (exact in a float), a fact's probability is
    Σ 2^i / (2^m − 1) over the scenarios where it holds: its numerator is the
    fact's truth, scenario by scenario (a world drawn twice merges two
    weights, and both bits stand). Returns each fact Silas reads within the
    budget with its truth mask over the scenario indices, and how many it
    refused."""
    n = len(a.worlds)
    blocks = []
    for start in range(0, n, TRUTH_BLOCK):
        m = min(TRUTH_BLOCK, n - start)
        blocks.append((start, m, SilasBelief.from_worlds(f, a.worlds[start:start + m],
                                                         [float(2 ** i) for i in range(m)], authority)))
    out, refused = [], 0
    for c in facts:
        query, mask = SilasQuery(c.text), 0
        try:
            for start, m, belief in blocks:
                bits = query.summarize(belief, BUDGET).event_probability.value * (2 ** m - 1)
                if bits.denominator != 1:
                    raise AssertionError(f"Silas's truth read is not a mask: {c.text}")
                mask |= int(bits) << start
        except SilasError:
            refused += 1
            continue
        out.append((c, mask))
    return out, refused


class _Column:
    """A column's scenarios and weights: exact probabilities of truth masks."""

    def __init__(self, a: Analysis, side: Sequence[int]):
        groups: dict[Fraction, int] = {}
        for r in side:
            w = Fraction(a.weights[r])
            groups[w] = groups.get(w, 0) | 1 << r
        self.groups = tuple(groups.items())
        self.total = sum(w * m.bit_count() for w, m in self.groups)

    def p(self, mask: int) -> Fraction:
        return sum((w * (mask & m).bit_count() for w, m in self.groups), Fraction(0)) / self.total


def _screen(t: np.ndarray, w: np.ndarray, n1: int):
    """In floats, every single fact's separation on the columns [:n1] and
    [n1:] of the truth matrix `t`, and every pair's under `and` and `or`,
    kept (upper triangle) only where the pair separates more than either of
    its facts; the rest are zero."""
    t1, t2 = t[:, :n1], t[:, n1:]
    w1, w2 = w[:n1] / w[:n1].sum(), w[n1:] / w[n1:].sum()
    p1, p2 = t1 @ w1, t2 @ w2
    single = np.abs(p1 - p2)
    j1, j2 = (t1 * w1) @ t1.T, (t2 * w2) @ t2.T
    both = np.abs(j1 - j2)
    either = np.abs((p1[:, None] + p1[None, :] - j1) - (p2[:, None] + p2[None, :] - j2))
    keep = np.triu(np.ones(both.shape, dtype=bool), 1)
    floor = np.maximum(single[:, None], single[None, :]) + 1e-9
    return single, np.where(keep & (both > floor), both, 0.0), np.where(keep & (either > floor), either, 0.0)


def _band(values: np.ndarray, top: int) -> np.ndarray:
    """The indices whose value is within float error of the `top`-th largest
    nonzero value or above it: the exact ranking is settled among these."""
    nonzero = values[values > 0]
    if not nonzero.size:
        return np.zeros((0, values.ndim), dtype=int) if values.ndim > 1 else np.zeros(0, dtype=int)
    kth = np.sort(nonzero)[-min(top, nonzero.size)]
    return np.argwhere(values >= kth - 1e-9) if values.ndim > 1 else np.flatnonzero(values >= kth - 1e-9)


def _explain(f, a: Analysis, one: Sequence[int], two: Sequence[int], authority, sit: Situation, own: int,
             tiles: tuple[int, int], top: int):
    """The declared space on both columns: every single fact's truth in each
    scenario read by Silas, the pairs screened in floats and settled exactly,
    the `top` single facts and two-fact explanations (ties: the simpler, then
    the text), the chance level from `RELABELINGS` relabelings, and whether a
    reason is found. Returns (found, facts, pairs, chance, searched)."""
    declared = space(sit, own, tiles)
    if not one or not two:
        return NO_REASON, (), (), (0.0, 0.0), (len(declared), 0, 0, 0)
    truths, refused = _truths(f, a, declared, authority)
    cols = [*one, *two]
    pooled = sum(1 << r for r in cols)
    distinct: dict[int, tuple[Candidate, int]] = {}
    named: dict[int, list[str]] = {}
    for c, mask in sorted(truths, key=lambda x: (x[0].atoms, x[0].text)):
        if mask & pooled not in (0, pooled):    # a fact constant on both columns separates nothing
            distinct.setdefault(mask & pooled, (c, mask))   # of facts alike on both columns, the simplest
            named.setdefault(mask & pooled, []).append(c.gloss)

    def alike(c: Candidate, mask: int) -> tuple[str, ...]:
        key = mask & pooled
        return tuple([g for g in named.get(key, ()) if g != c.gloss]
                     + [f"not: {g}" for g in named.get(pooled ^ key, ())])

    singles = list(distinct.values())
    c1, c2 = _Column(a, one), _Column(a, two)
    if not singles:
        return NO_REASON, (), (), (0.0, 0.0), (len(declared), 0, refused, 0)
    t = np.array([[(m >> r) & 1 for r in cols] for _, m in singles], dtype=np.float64)
    w = np.array([a.weights[r] for r in cols], dtype=np.float64)
    single, both, either = _screen(t, w, len(one))
    rng = np.random.default_rng(CHANCE_SEED)
    chance = [0.0, 0.0]
    for _ in range(RELABELINGS):
        perm = rng.permutation(len(cols))
        s, b, e = _screen(t[:, perm], w[perm], len(one))
        chance = [max(chance[0], float(s.max())), max(chance[1], float(b.max()), float(e.max()))]

    def sep(mask: int) -> Fraction:
        return abs(c1.p(mask) - c2.p(mask))

    def best(found: list[tuple[Fraction, Candidate, int]], taken: set[int]) -> list[tuple[Candidate, int]]:
        """The `top` best, exactly, a fact and its complement (which separates
        as much) shown once."""
        out = []
        for _, c, m in sorted(found, key=lambda x: (-x[0], x[1].atoms, x[1].text)):
            key = m & pooled
            if key in taken or pooled ^ key in taken:
                continue
            taken.add(key)
            out.append((c, m))
            if len(out) == top:
                break
        return out

    shown: set[int] = set()
    facts = tuple(dataclasses.replace(_report(f, a, one, two, authority, c, m, c1, c2), alike=alike(c, m))
                  for c, m in best([(sep(singles[i][1]), *singles[i]) for i in _band(single, 2 * top)], shown))
    paired = []
    for op, matrix, combine in (("and", both, int.__and__), ("or", either, int.__or__)):
        for i, j in _band(np.maximum(both, either), 4 * top):
            if matrix[i, j] <= 0:
                continue
            (ci, mi), (cj, mj) = singles[i], singles[j]
            mask = combine(mi, mj)
            s = sep(mask)
            if s > max(sep(mi), sep(mj)):
                paired.append((s, two_fact(ci, cj, op), mask))
    # A pair that reads, on both columns, as a single fact of the space (or its complement) is that fact.
    pairs = tuple(_report(f, a, one, two, authority, c, m, c1, c2) for c, m in best(paired, set(distinct)))
    one_fact = bool(facts) and facts[0].separation >= SEPARATION and facts[0].separation > chance[0]
    two_facts = bool(pairs) and pairs[0].separation >= SEPARATION and pairs[0].separation > chance[1]
    if two_facts and (not one_fact or pairs[0].separation > facts[0].separation):
        found = TWO_FACTS
    else:
        found = ONE_FACT if one_fact else NO_REASON
    kept = int(np.count_nonzero(both) + np.count_nonzero(either))
    return found, facts, pairs, (chance[0], chance[1]), (len(declared), len(singles), refused, kept)


def _report(f, a: Analysis, one: Sequence[int], two: Sequence[int], authority, c: Candidate, mask: int,
            c1: _Column, c2: _Column) -> Fact:
    """One fact, read by Silas on each column from its own text — which must
    agree, exactly, with what its truth mask says — with its exact prior
    over the fiber where Silas resolves it."""
    query = SilasQuery(c.text)
    p1 = query.summarize(_side(f, a, one, authority), BUDGET).event_probability.value
    p2 = query.summarize(_side(f, a, two, authority), BUDGET).event_probability.value
    if (p1, p2) != (c1.p(mask), c2.p(mask)):
        raise AssertionError(f"Silas reads {c.gloss} as {p1}, {p2}; its truth mask as {c1.p(mask)}, {c2.p(mask)}")
    summary = query.summarize(_side(f, a, one, authority, unit=True), BUDGET)
    prior = prior_authority = None
    try:
        exact = query.summarize(SilasBelief.uniform_resolving(f, c.reads, PRIOR_CAP), BUDGET).event_probability
        prior, prior_authority = exact.value, exact.authority.label
    except SilasError:
        pass
    return Fact(
        candidate=c, on_a=p1, on_b=p2,
        count_a=int(summary.event_probability.value * len(one)),
        count_b=int(query.summarize(_side(f, a, two, authority, unit=True), BUDGET).event_probability.value * len(two)),
        overall=query.summarize(SilasBelief.from_worlds(f, a.worlds, a.weights, authority), BUDGET)
        .event_probability.value,
        prior=prior, authority=summary.authority.label, prior_authority=prior_authority,
    )


def _holds(f, query: SilasQuery, world, authority) -> bool:
    belief = SilasBelief.from_worlds(f, [world], [1.0], authority)
    return query.summarize(belief, BUDGET).event_probability.value == 1


def _path(steps: Sequence[Step], r: int, viewer: int, root: int) -> tuple[tuple[int, int], ...]:
    """Scenario r's path through the plan after `root`: the plays reaching
    its last viewer cell, and the tile played there."""
    mine = [s for s in steps if r in s.cell]
    if not mine:
        return ((viewer, root),)
    return mine[-1].plays + ((viewer, mine[-1].tile),)


def _example(f, a: Analysis, side: Sequence[int], best: Fact | None, on_a: bool, authority,
             tiles: tuple[int, int], viewer: int) -> Example | None:
    """The heaviest world of `side` (lowest index among equals) in which the
    best fact reads as that side mostly reads it — or the heaviest, when
    there is no fact or no such world."""
    if not side:
        return None
    order = sorted(side, key=lambda r: (-a.weights[r], r))
    pick, holds = order[0], None
    if best is not None:
        want = (best.on_a > best.on_b) == on_a
        query = SilasQuery(best.candidate.text)
        for r in order:
            if _holds(f, query, a.worlds[r], authority) == want:
                pick, holds = r, want
                break
        else:
            holds = not want
    lines = a.line or {}
    return Example(
        index=pick, world=tuple(a.worlds[pick]), weight=a.weights[pick],
        paths={t: _path(lines.get(t, ()), pick, viewer, t) for t in tiles},
        margins={t: a.outcomes[t][pick] for t in tiles}, fact_holds=holds,
    )


# -- rendering --------------------------------------------------------------------------


def next_moves(r: Reason, root: int, depth: int = 1) -> list[tuple[int, Fraction, int]]:
    """The plan's `depth`-th viewer decision after `root`: (tile, share of the
    belief's mass that reaches it playing that tile, cells), heaviest first."""
    total = sum(Fraction(w) for w in r.weights)
    out: dict[int, tuple[Fraction, int]] = {}
    for s in r.lines.get(root, ()):
        if sum(seat == r.viewer for seat, _ in s.plays) == depth:
            m, c = out.get(s.tile, (Fraction(0), 0))
            out[s.tile] = (m + sum(Fraction(r.weights[i]) for i in s.cell) / total, c + 1)
    return sorted(((t, m, c) for t, (m, c) in out.items()), key=lambda x: (-x[1], x[0]))


def render(r: Reason) -> str:
    """The report as markdown."""
    z, o = tile_name(r.tile), tile_name(r.other)
    role = "bidder's side" if r.viewer % 2 == r.bidder % 2 else "defenders"
    goal = "make" if r.viewer % 2 == r.bidder % 2 else "set"
    lab = r.authority
    lines = [
        f"# Reason report: {z} (Zeb) vs {o}",
        "",
        f"- Seat S{r.viewer} to play ({role}); declaration {r.decl_id}, S{r.bidder} bid {r.bid}.",
        f"- Holding: {' '.join(tile_name(t) for t in sorted(tiles_of(r.own), reverse=True))}.",
        f"- Plays so far: {_plays(r.plays) if r.plays else '(none)'}.",
        f"- Plan: {describe(r.config)}; {r.n_worlds} scenarios.",
        "",
        "## Verdict",
        "",
    ]
    if r.verdict == INDIFFERENT:
        lines.append(f"**Indifferent — no reason.** {z} and {o} tie exactly under the plan "
                     f"(P({z} beats {o}, ties halved) = {_num(r.beats)}, {lab}). A tie is not a reason (#120).")
    elif r.verdict == INDISTINGUISHABLE:
        lines.append(f"**Indistinguishable — no reason.** The price, {r.price:+.4f} pmake for the mover, lies inside "
                     f"the noise bar ±{float(r.bar):.4f}; Silas compares P({z} beats {o}, ties halved) = "
                     f"{_num(r.beats)} ({lab}) with 1/2 at band {_num(r.bar / 2)}: {r.compare}.")
    else:
        better = z if r.compare == "better" else o
        lines.append(f"**{better} is better, outside the noise bar.** Price {r.price:+.4f} pmake for the mover "
                     f"({z} − {o}); bar ±{float(r.bar):.4f}. Silas: P({z} beats {o}, ties halved) = {_num(r.beats)} "
                     f"({lab}) vs 1/2 at band {_num(r.bar / 2)}: **{r.compare}**.")
    lines += ["", "## Noise bar", "", f"| plan | q {z} | q {o} | price (mover) |", "|---|---|---|---|",
              f"| base: {describe(r.config)} | {r.q[r.tile]:.4f} | {r.q[r.other]:.4f} | {r.price:+.4f} |"]
    for arm in r.arms:
        lines.append(f"| {arm.label}: {describe(arm.config)} | {arm.q[r.tile]:.4f} | {arm.q[r.other]:.4f} "
                     f"| {arm.price:+.4f} |")
    lines += ["", f"Bar = the furthest re-plan from the base price, rounded up to 1e-6: {float(r.bar):.6f} "
              "(each q is its own plan's, sampled)."]
    lines += ["", "## The line", ""]
    for t in (r.tile, r.other):
        lines.append(f"- After **{tile_name(t)}** (q {r.q[t]:.4f}):")
        for depth, which in ((1, "next"), (2, "second")):
            nxt = next_moves(r, t, depth)
            reached = sum((m for _, m, _ in nxt), Fraction(0))
            moves = ", ".join(f"{tile_name(u)} ({float(m):.0%}, {c} cell{'s' if c > 1 else ''})" for u, m, c in nxt)
            lines.append(f"  - its {which} decision plays {moves or '(none)'}; "
                         f"{float(1 - reached):.0%} of the mass is decided before it.")
    lines += ["", "## Where the tiles part", "",
              f"- A, {z} {goal}s and {o} does not: {len(r.side_a)} of {r.n_worlds} scenarios, "
              f"weight share {_num(r.share_a)} ({lab}).",
              f"- B, {o} {goal}s and {z} does not: {len(r.side_b)} of {r.n_worlds} scenarios, "
              f"weight share {_num(r.share_b)} ({lab}).",
              f"- The rest agree. Price = share A − share B = {float(r.share_a - r.share_b):+.4f}."]
    if r.verdict != REASON:
        return "\n".join(lines) + "\n"
    one, two = r.columns
    rest = "" if "the rest" not in r.columns else " (one side is empty: the rest are the scenarios where the tiles agree)"
    declared, distinct, refused, kept = r.searched
    lines += ["", "## What must be true (Silas)", "",
              f"Searched: the declared space of `scenario.reason.space`, {declared} single facts (holdings, a holding "
              "with another of a suit, voids, by seat and by 'an opponent'; the open trick's seats still to play; "
              "count at risk; the book prelude's nouns as holdings), of which "
              f"{distinct} read differently on these scenarios{f' ({refused} refused by the budget)' if refused else ''}"
              f"; and the conjunction and disjunction of every two of those, kept where the pair separates more "
              f"than either fact ({kept}). Ranked by |P(fact | {one}) − P(fact | {two})|, then the simpler{rest}. "
              f"Coverage: the share of {one} and of {two} the fact explains, read toward the column where it holds "
              "more.", "",
              f"Threshold, set before looking: a reason separates the columns by at least {_num(SEPARATION)} and by "
              f"more than the space's best on {RELABELINGS} random relabelings of these scenarios into columns of the "
              f"same sizes (chance: {r.chance[0]:.3f} for one fact, {r.chance[1]:.3f} for two); two facts are the "
              "reason only when they separate strictly more than a single fact that passes.", ""]
    best = r.facts[0].separation if r.facts else Fraction(0)
    pair = r.pairs[0].separation if r.pairs else Fraction(0)
    if r.found == ONE_FACT:
        lines.append(f"**Reason found: one fact**, separation {float(best):.3f}: {r.facts[0].candidate.gloss}.")
    elif r.found == TWO_FACTS:
        lines.append(f"**Reason found: two facts**, separation {float(pair):.3f} (the best single fact, "
                     f"{float(best):.3f}, falls short or separates less): {r.pairs[0].candidate.gloss}.")
    else:
        lines.append(f"**No reason found.** No fact of the declared space reaches the threshold: the best single "
                     f"fact separates {float(best):.3f} (chance {r.chance[0]:.3f}), the best two {float(pair):.3f} "
                     f"(chance {r.chance[1]:.3f}). The best candidates anyway:")
    for title, facts in (("One fact", r.facts), ("Two facts", r.pairs)):
        lines += ["", f"### {title}", "",
                  f"| # | fact | P(· \\| {one}) | P(· \\| {two}) | coverage {one} / {two} | worlds | P(·) in the sample "
                  "| exact prior over the fiber |", "|---|---|---|---|---|---|---|---|"]
        for i, fact in enumerate(facts, 1):
            prior = "refused" if fact.prior is None else f"{_num(fact.prior)} ({fact.prior_authority})"
            cover = " / ".join(f"{float(x):.0%}" for x in fact.coverage)
            lines.append(f"| {i} | {fact.candidate.gloss} | {_num(fact.on_a)} | {_num(fact.on_b)} | {cover} | "
                         f"{fact.count_a} / {fact.count_b} | {_num(fact.overall)} | {prior} |")
        if not facts:
            lines.append("| — | (none) | | | | | | |")
    lines += ["", f"Side probabilities and the sample column are {r.facts[0].authority if r.facts else lab}.", ""]
    for mark, facts in (("", r.facts), ("pair ", r.pairs)):
        for i, fact in enumerate(facts, 1):
            lines.append(f"- {mark}{i}: `{fact.candidate.text}`")
    named = [(i, fact.alike) for i, fact in enumerate(r.facts, 1) if fact.alike]
    if named:
        lines += ["", "The same split, named otherwise (other facts of the space that read alike on both columns):", ""]
        lines += [f"- {i}: {'; '.join(alike[:8])}{f' (and {len(alike) - 8} more)' if len(alike) > 8 else ''}"
                  for i, alike in named]
    which = "the two-fact explanation" if r.found == TWO_FACTS else "fact 1"
    lines += ["", "## Counterexample worlds", ""]
    for name, ex in (("A", r.examples[0]), ("B", r.examples[1])):
        if ex is None:
            lines.append(f"- Side {name} is empty.")
            continue
        fact = "" if ex.fact_holds is None else f" — {which} {'holds' if ex.fact_holds else 'fails'} here"
        lines.append(f"- **Side {name}**, scenario {ex.index} (weight {ex.weight:g}){fact}:")
        lines.append(f"  - hands: {_hands(ex.world, r.viewer)}")
        for t in (r.tile, r.other):
            made = "made" if ex.margins[t] >= 2 * r.bid - HAND_TOTAL else "set"
            lines.append(f"  - after {tile_name(t)}: {_plays(ex.paths[t])} … margin {ex.margins[t]:+.0f} ({made})")
    return "\n".join(lines) + "\n"


# -- the command line -------------------------------------------------------------------


def _main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -u -m scenario.reason", description=__doc__.split("\n\n")[0])
    ap.add_argument("spec", help="the plan, as a walt spec (arena/walt_play.py), e.g. k0,n512,F=net.npz,f=16")
    ap.add_argument("--decl", type=int, required=True)
    ap.add_argument("--bidder", type=int, default=0)
    ap.add_argument("--bid", type=int, required=True)
    ap.add_argument("--hands", required=True, help="the deal: four comma-separated tile masks, absolute seats")
    ap.add_argument("--plays", default="", help="seat:tile pairs, comma-separated, in play order")
    ap.add_argument("--other", type=int, required=True, help="the named tile's id")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="write the markdown here (default: stdout)")
    a = ap.parse_args(argv)
    from arena.walt_play import parse_walt_spec
    from scenario.atlas import _state
    from scenario.record import empty_record
    walt = parse_walt_spec(a.spec).walt
    config = dataclasses.replace(walt.config, seed=a.seed)
    rules = rules_for(a.decl)
    plays = tuple((int(s), int(t)) for s, t in (p.split(":") for p in a.plays.split(",") if p))
    rec = empty_record(a.bidder)
    for _, t in plays:
        rec = rec.advance(t, rules)
    hands = tuple(tuple(tiles_of(int(m))) for m in a.hands.split(","))
    state = _state(hands, a.bidder, a.bid, a.decl, plays, rec)
    text = render(reason(state, a.bid, a.other, config, belief=walt.belief))
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text)
    else:
        print(text)


if __name__ == "__main__":
    _main()
