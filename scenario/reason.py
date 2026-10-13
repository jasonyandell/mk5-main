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
  over the hidden hands (`candidates`), each evaluated by Silas on the
  decision's belief — Zeb's own scenarios and weights, labeled
  `sampled(n, seed)` — restricted to each side. The facts whose masses on
  the two sides differ most are the reason. Each comes with its exact
  probability under the uniform over the fiber, where Silas can resolve it.
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

from scenario_core import SilasAuthority, SilasBelief, SilasError, SilasQuery, silas_compare

from forge.zeb.types import ZebGameState

from scenario.planner import HAND_TOTAL, Analysis, Belief, Config, Rung, Step, Walt, situation_from_state
from scenario.record import remaining_hand
from scenario.rules import ALL_TILES, bit, rules_for, tiles_of
from scenario.silas import frame

BUDGET = 10_000_000          # Silas work units per summary
PRIOR_CAP = 200_000          # points of an exact prior's resolving belief
BAR_GRAIN = 10 ** 6          # the bar is rounded up to a multiple of 1/BAR_GRAIN
STRONGER = Rung(8, 4)        # the rung added below a one-rung plan (the arena's m8, g4)

INDIFFERENT = "indifferent"            # an exact tie: no reason (#120)
INDISTINGUISHABLE = "indistinguishable"  # the price lies inside the noise bar: no reason
REASON = "reason"


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


def candidates(decl_id: int, viewer: int, own: int, played: int, sizes: Sequence[int]) -> list[Candidate]:
    """The declared search space, at a decision of seat `viewer` holding `own`:

    1. `holds`: seat X holds t — X a hidden seat, t a hidden tile — and an
       opponent (either) holds t;
    2. `holds with`: X holds t and another tile of context q — q a context
       t follows, with another hidden tile in it — and the same for an
       opponent (one opponent holding both);
    3. `void`: X holds no tile of context q, and an opponent is void in q —
       q a context with a hidden tile in it.

    Nothing else is generated. Seats are named absolutely (`S1`); the
    opponent forms name the viewer's opponents by role (`opponent me foe`)."""
    rules = rules_for(decl_id)
    hidden = ALL_TILES & ~played & ~own
    seats = [s for s in range(4) if s != viewer and sizes[s] > 0]
    contexts = [k for k in range(8) if rules.follow_bits[k] & hidden]
    who = [(f"S{s}", f"S{s} ({'partner' if s == (viewer + 2) % 4 else 'opponent'})", "") for s in seats]
    who.append(("foe", "an opponent", "(viewer me) (opponent me foe) "))
    roles = {"foe": "(chair me) (chair foe)"}
    out: list[Candidate] = []

    def query(holder: str, extra_roles: str, body: str) -> str:
        r = " ".join(x for x in (roles.get(holder, ""), extra_roles) if x)
        return f"(query (roles{' ' + r if r else ''}) (out) (pattern {body}))"

    for holder, said, lead in who:
        base = 2 if holder == "foe" else 0
        for t in tiles_of(hidden):
            name = tile_name(t)
            out.append(Candidate(query(holder, "", f"{lead}(holds {holder} {name})"),
                                 f"{said} holds {name}", bit(t), base + 1))
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
                out.append(Candidate(query(holder, "(domino other)", body),
                                     f"{said} holds {name} and another of {_SUIT[k]}",
                                     bit(t) | (rules.follow_bits[k] & hidden), base + 4))
    for holder, said, lead in who:
        base = 2 if holder == "foe" else 0
        for k in contexts:
            out.append(Candidate(query(holder, "", f"{lead}(void {holder} {context_name(k)})"),
                                 f"{said} is void in {_SUIT[k]}", rules.follow_bits[k] & hidden, base + 1))
    return out


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

    @property
    def separation(self) -> Fraction:
        return abs(self.on_a - self.on_b)


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
    facts: tuple[Fact, ...]   # the top separating facts, best first (empty without a reason)
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

    facts: tuple[Fact, ...] = ()
    examples: tuple[Example | None, Example | None] = (None, None)
    columns = ("A", "B")
    if verdict == REASON:
        one, two = side_a, side_b
        if not side_a or not side_b:   # one tile dominates: set its side against the worlds where they agree
            agree = tuple(r for r in range(n) if zs[r] == os_[r])
            one, two, columns = (side_a, agree, ("A", "the rest")) if side_a else (agree, side_b, ("the rest", "B"))
        facts = _separate(f, a, one, two, sampled, sit, own, top)
        best = facts[0] if facts else None
        examples = (_example(f, a, side_a, best, True, sampled, (tile, other), viewer),
                    _example(f, a, side_b, best, False, sampled, (tile, other), viewer))
    return Reason(
        decl_id=sit.decl_id, viewer=viewer, bidder=sit.bidder, bid=bid, plays=sit.plays, own=own,
        config=config, tile=tile, other=other, q=dict(a.q), price=price, authority=sampled.label,
        n_worlds=n, side_a=side_a, side_b=side_b, share_a=share_a, share_b=share_b, columns=columns,
        facts=facts, examples=examples, lines=a.line or {}, weights=tuple(a.weights), arms=tuple(arms),
        bar=band, beats=left.value - right.value + Fraction(1, 2), verdict=verdict, compare=compared,
    )


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


def _separate(f, a: Analysis, one: Sequence[int], two: Sequence[int], authority, sit, own: int,
              top: int) -> tuple[Fact, ...]:
    """Every candidate on both sides, by Silas; the `top` whose probabilities
    on the two sides differ most (ties: the simpler query, then its text)."""
    if not one or not two:
        return ()
    rec = sit.record
    space = candidates(sit.decl_id, sit.viewer, own, rec.played, [rec.hand_size(s) for s in range(4)])
    b1, b2 = _side(f, a, one, authority), _side(f, a, two, authority)
    scored = []
    for c in space:
        query = SilasQuery(c.text)
        p1 = query.summarize(b1, BUDGET).event_probability.value
        p2 = query.summarize(b2, BUDGET).event_probability.value
        if p1 != p2:
            scored.append((-abs(p1 - p2), c.atoms, c.text, c, query, p1, p2))
    scored.sort(key=lambda x: x[:3])
    u1, u2 = _side(f, a, one, authority, unit=True), _side(f, a, two, authority, unit=True)
    whole = SilasBelief.from_worlds(f, a.worlds, a.weights, authority)
    out = []
    for *_, c, query, p1, p2 in scored[:top]:
        summary = query.summarize(u1, BUDGET)
        prior = prior_authority = None
        try:
            exact = query.summarize(SilasBelief.uniform_resolving(f, c.reads, PRIOR_CAP), BUDGET).event_probability
            prior, prior_authority = exact.value, exact.authority.label
        except SilasError:
            pass
        out.append(Fact(
            candidate=c, on_a=p1, on_b=p2,
            count_a=int(summary.event_probability.value * len(one)),
            count_b=int(query.summarize(u2, BUDGET).event_probability.value * len(two)),
            overall=query.summarize(whole, BUDGET).event_probability.value,
            prior=prior, authority=summary.authority.label, prior_authority=prior_authority,
        ))
    return tuple(out)


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
    lines += ["", "## What must be true (Silas)", "",
              "Searched: the declared space of `scenario.reason.candidates` (holdings, a holding with another "
              f"of a suit, voids; by seat and by 'an opponent'). Ranked by |P(fact | {one}) − P(fact | {two})|"
              f"{rest}.", "",
              f"| # | fact | P(· \\| {one}) | P(· \\| {two}) | worlds | P(·) in the sample | exact prior over the fiber |",
              "|---|---|---|---|---|---|---|"]
    for i, fact in enumerate(r.facts, 1):
        prior = "refused" if fact.prior is None else f"{_num(fact.prior)} ({fact.prior_authority})"
        lines.append(f"| {i} | {fact.candidate.gloss} | {_num(fact.on_a)} | {_num(fact.on_b)} | "
                     f"{fact.count_a} / {fact.count_b} | {_num(fact.overall)} | {prior} |")
    lines += ["", f"Side probabilities and the sample column are {r.facts[0].authority if r.facts else lab}.", ""]
    for i, fact in enumerate(r.facts, 1):
        lines.append(f"{i}. `{fact.candidate.text}`")
    lines += ["", "## Counterexample worlds", ""]
    for name, ex in (("A", r.examples[0]), ("B", r.examples[1])):
        if ex is None:
            lines.append(f"- Side {name} is empty.")
            continue
        fact = "" if ex.fact_holds is None else f" — fact 1 {'holds' if ex.fact_holds else 'fails'} here"
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
