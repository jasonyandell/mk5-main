# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""The book player: Winning 42's rules as a Silas policy stack, in the arena.

The eighteen book-faithful policies of `silas/examples/book/` ([[silas]]
§ The book, rule by rule) are asked at every decision of the book player's
seats, in a stated priority order. The first policy whose rule fires decides,
and the tile is drawn from the set that rule allows; where none fires, a
fallback plays. Every policy is asked at every decision, so the log records
each one that fired, not only the one that decided.

The priority order is read off the book's own structure: who is to play
(bidder, his partner, a setter) and whether he leads or follows. The policies'
guards already name the seat and the phase, so most pairs never fire
together; the order decides only where two do.

Leading, per seat:

- the bidder: `first-lead` (trick one), then `crisis` (a trump he cannot draw:
  the book's override of the plan), then `dead-double` (a dead double cashed
  at once while a setter holds trump), then `bidders-plan`;
- his partner: `partner-leads`;
- a setter: `trump-set` (a high trump at once, holding the trumps), then
  `setter-leads`.

Following, any seat, in this order: the contract's exceptions first
(`defend-the-84`, `trump-set`); then the count on the trick
(`partner-plays-the-top-trump`, `donate-when-safe`, `protect-the-count`,
`pounce`: `follow-low` names these as deciding before it); then taking the
trick (`take-the-trick`, `helper-stretch`); then the throws (`dead-double`,
which the book throws first, `shed-for-a-void`); then the follows with
nothing at stake (`low-from-a-pair`, `follow-with-the-off`, `follow-low`).

Two rules say only what the tile must not be: the pounce's
`no-count-on-a-lost-trick` and the donation's
`no-count-on-a-trick-that-can-be-lost` allow every tile but count. Which
no-count tile goes is the later rules' business (the throws, and
`shed-for-a-void`'s tile he need not keep, are written for exactly that), so
where one of them decides, its set is narrowed by each later firing policy of
the phase whose set meets it, in order, and the tile is drawn from what is
left (#146: before, the least no-count tile went, outside every co-firing
throw).

Variants (spec `book:<fallback>,<ties>[,w<eps>][/<walt options>]`):

  fallback  lowest  no rule fires: the lowest legal tile (Silas's own fallback)
            zeb     no rule fires: Zeb's tile
  ties      least   a rule allows a set: its least tile (Silas's own decision)
            zeb     a rule allows a set: Zeb's best tile inside it, by Zeb's
                    q for the mover's side (the bidding side's make
                    probability, or one less it for a setter), ties to the
                    lowest id (Zeb constrained by the book)
            union   every policy of the phase that fires may decide: Zeb's
                    best tile inside the union of their sets (the most
                    generous priority order: whichever firing rule allows the
                    tile Zeb prefers)
  w<eps>    Zeb decides, and the book breaks Zeb's near-ties: where the
            deciding rule allows a tile within eps of the best by Zeb's q
            for the mover's side, the best such tile; else Zeb's tile
            (`book:zeb,zeb,w0`: Zeb's exact ties; needs fallback and ties zeb)

  read=<likelihood>[+calls]
            the partner reads (#150): Zeb's worlds (the fallback's, the
            tie-break's) are weighted by the likelihood that the partner
            played by this stack (`arena/book_read.py`, `consistent` or
            `uniform`) and, with `+calls`, bid by `bid-only-enough`; the
            opponents stay uniform. Needs Zeb's uniform draw.

`book:lowest,least` is the pure Silas stack and plans nothing. Every other
variant asks Zeb (`walt:` options after the slash, as `arena.walt_play`
reads them) at every decision of its seats, so that the log holds Zeb's tile
and q beside the book's wherever a rule fired.

`SilasBidder` is a bidding policy of `silas/examples/book/bidding/`
(`risk-budget`, `bid-only-enough`, ...) as an arena `BidPolicy`.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from forge.zeb.types import ZebGameState

from arena.auction import PASS, BidContext, bidding_order, contract_points
from scenario.planner import situation_from_state
from scenario.rules import mask_of
from scenario.silas import frame

BOOK = Path(__file__).resolve().parents[1] / "silas" / "examples" / "book"

LEAD_ORDER = (
    "first-lead", "crisis", "dead-double", "bidders-plan",
    "partner-leads",
    "trump-set", "setter-leads",
)
FOLLOW_ORDER = (
    "defend-the-84", "trump-set",
    "partner-plays-the-top-trump", "donate-when-safe", "protect-the-count", "pounce",
    "take-the-trick", "helper-stretch",
    "dead-double", "shed-for-a-void",
    "low-from-a-pair", "follow-with-the-off", "follow-low",
)
POLICIES = tuple(sorted(set(LEAD_ORDER) | set(FOLLOW_ORDER)))
# The rules that only forbid count: the later firing policies choose within them.
NO_COUNT = frozenset({
    ("pounce", "no-count-on-a-lost-trick"),
    ("donate-when-safe", "no-count-on-a-trick-that-can-be-lost"),
})
FALLBACKS = ("lowest", "zeb")
# The work cap for one policy at one decision. A cap only stops runaway work:
# a policy that finishes under it allows the same tiles under any larger one.
# A policy that reads another seat's plays (crisis-read) runs that seat's
# policy on every hand he could hold, far past the binding's default 10^7.
BUDGET = 10**12
TIES = ("least", "zeb", "union")


@lru_cache(maxsize=None)
def _text(path: Path) -> str:
    return path.read_text()


@lru_cache(maxsize=None)
def compiled(name: str, decl: int):
    """The book policy `name` compiled under Zeb's declaration `decl`."""
    from scenario_core import SilasPolicy
    return SilasPolicy(_text(BOOK / f"{name}.silas"), decl, readers=str(BOOK))


@dataclass(frozen=True)
class Verdict:
    """The stack at one decision: every policy that fired, as (rule, allowed
    tiles), and the one that decides (`policy` None: no rule fired)."""

    fired: dict[str, tuple[str, tuple[int, ...]]]
    policy: str | None
    rule: str | None
    allowed: tuple[int, ...]

    def phase(self, leading: bool) -> list[str]:
        """The firing policies that may decide in this phase, in its order."""
        return [name for name in (LEAD_ORDER if leading else FOLLOW_ORDER) if name in self.fired]


def ask(state: ZebGameState, bid: int, swap: dict[str, str] | None = None) -> Verdict:
    """Every book policy at the decision of `state`'s seat to play, under the
    contract `bid` as the table says it (a marks bid stays a marks bid).
    `swap` compiles a stack name from another book file (`{"crisis":
    "crisis-read"}`): the stack's names and order are unchanged."""
    sit = situation_from_state(state, bid)
    seat = sit.viewer
    own = mask_of(t for t in state.hands[seat] if t not in state.played)
    f = frame(sit, own)
    fired = {}
    for name in POLICIES:
        layer, tiles = compiled((swap or {}).get(name, name), state.decl_id).allowed(f, budget=BUDGET)
        if layer != "fallback":
            fired[name] = (layer, tuple(sorted(tiles)))
    order = LEAD_ORDER if not state.current_trick else FOLLOW_ORDER
    return Verdict(fired, *decide(fired, order))


def decide(fired: dict[str, tuple[str, tuple[int, ...]]], order: Sequence[str]):
    """The deciding policy of the phase `order`, its rule and its set: the first
    that fired, its set narrowed by the later ones where it only forbids count
    (`NO_COUNT`). (None, None, ()) where none fired."""
    firing = [name for name in order if name in fired]
    if not firing:
        return None, None, ()
    policy = firing[0]
    rule, allowed = fired[policy]
    if (policy, rule) in NO_COUNT:
        left = set(allowed)
        for later in firing[1:]:
            within = left & set(fired[later][1])
            if within:
                left = within
        allowed = tuple(sorted(left))
    return policy, rule, allowed


class BookPlay:
    """The book player as an arena `PlayPolicy` (see the module docstring)."""

    def __init__(self, fallback: str = "zeb", ties: str = "least", zeb=None, log: bool = False,
                 within: float | None = None, read: str | None = None, shadow: bool = False,
                 swap: dict[str, str] | None = None):
        if fallback not in FALLBACKS or ties not in TIES:
            raise ValueError(f"book: fallback is one of {FALLBACKS}, ties one of {TIES}")
        if zeb is None and (fallback == "zeb" or ties != "least"):
            raise ValueError("book: a Zeb fallback or tie-break needs a Zeb (walt options)")
        if within is not None and (within < 0 or fallback != "zeb" or ties != "zeb"):
            raise ValueError("book: w<eps> is Zeb deciding with the book at its near-ties: eps >= 0, zeb,zeb")
        self.fallback, self.ties, self.zeb, self.within = fallback, ties, zeb, within
        # swap: a stack name compiled from another book file, e.g. {"crisis": "crisis-read"}.
        self.swap = dict(swap or {})
        for name, file in self.swap.items():
            if name not in POLICIES or not (BOOK / f"{file}.silas").exists():
                raise ValueError(f"book: swap names a stack policy and a book file, not {name}={file}")
        self.log: list[dict] | None = [] if log else None
        self.partner = None
        if read is not None:
            from arena.book_read import BookBelief
            if zeb is None or zeb.walt.belief is not None or zeb.walt.config.draw != "uniform":
                raise ValueError("book: the partner read weighs plain Zeb's uniform draw, nothing else")
            likelihood, _, calls = read.partition("+")
            if calls not in ("", "calls"):
                raise ValueError(f"book: read=<likelihood>[+calls], not {read!r}")
            self.partner = BookBelief(likelihood, seats="partner", calls=bool(calls))
            zeb.walt.belief = self.partner
        # shadow: also plan each decision unread, on the same sample, and log
        # whether the read moved the book player's tile (`moved`).
        self.moved: list[dict] | None = [] if shadow else None
        self._plain = None
        if shadow:
            if self.partner is None:
                raise ValueError("book: a shadow compares the read with the unread: it needs read=")
            from scenario.compiled import CompiledWalt
            self._plain = CompiledWalt(zeb.walt.config)

    def choose(
        self,
        states: Sequence[ZebGameState],
        bid_values: Sequence[int],
        marks: Sequence[tuple[int, int]] | None = None,
        marks_to_win: int = 7,
    ) -> list[int]:
        verdicts = [ask(s, int(b), self.swap) for s, b in zip(states, bid_values)]
        points = [contract_points(int(b)) for b in bid_values]
        if self.partner is not None:
            self.partner.note(states, bid_values)
        zeb = (self.zeb.walt.decide_batch(states, points)
               if self.zeb is not None else [(None, {})] * len(states))
        plain = self._plain.decide_batch(states, points) if self._plain is not None else None
        slots = []
        for k, (state, bid, v, (z_tile, q)) in enumerate(zip(states, bid_values, verdicts, zeb)):
            seat = (state.trick_leader + len(state.current_trick)) % 4
            legal = sorted(state.hands[seat][i] for i in _legal_slots(state))
            policy, rule, allowed, tile = self._pick(state, v, legal, z_tile, q)
            if plain is not None and len(legal) > 1:
                p_tile = self._pick(state, v, legal, *plain[k])[3]
                self.moved.append(dict(trick=len(state.play_history) // 4, policy=policy,
                                       zeb=z_tile, plain_zeb=plain[k][0], tile=tile, plain_tile=p_tile))
            if self.log is not None:
                self.log.append(dict(
                    hands=state.hands, plays=state.play_history, decl=state.decl_id, bidder=state.bidder,
                    bid=int(bid), seat=seat, legal=legal, zeb=z_tile,
                    q={int(t): float(x) for t, x in q.items()},
                    fired={k: [r, list(a)] for k, (r, a) in v.fired.items()},
                    policy=policy, rule=rule, allowed=list(allowed), tile=tile,
                ))
            slots.append(list(state.hands[seat]).index(tile))
        return slots

    def _pick(self, state: ZebGameState, v: Verdict, legal: list[int], z_tile, q: dict):
        """The book player's (policy, rule, allowed, tile) at one decision, given Zeb's tile and q."""
        seat = (state.trick_leader + len(state.current_trick)) % 4
        # Zeb's q is the bidding side's make probability: a setter's best is its least.
        sign = 1.0 if seat % 2 == state.bidder % 2 else -1.0
        best = lambda tiles: max(tiles, key=lambda t: (sign * q[t], -t))  # noqa: E731
        policy, rule, allowed = v.policy, v.rule, v.allowed
        if self.ties == "union" and policy is not None:
            allowed = tuple(sorted({t for name in v.phase(not state.current_trick) for t in v.fired[name][1]}))
        if self.within is not None and policy is not None:
            allowed = tuple(t for t in allowed if sign * q[t] >= sign * q[z_tile] - self.within)
        if policy is None:
            tile = legal[0] if self.fallback == "lowest" else z_tile
        elif not allowed:
            tile = z_tile  # Zeb decides: no tile of the book's is near its best
        elif self.ties == "least":
            tile = allowed[0]
        else:
            tile = best(allowed)
        if self.ties == "union" and policy is not None:
            # The decision is the first policy of the phase whose set holds the tile.
            policy = next(n for n in v.phase(not state.current_trick) if tile in v.fired[n][1])
            rule = v.fired[policy][0]
        return policy, rule, allowed, tile

    def __repr__(self) -> str:
        zeb = "" if self.zeb is None else f", zeb={self.zeb!r}"
        within = "" if self.within is None else f", within={self.within:g}"
        read = ("" if self.partner is None else
                f", read=partner:{self.partner.likelihood}" + ("+calls" if self.partner.calls else ""))
        swap = "".join(f", {n}={f}" for n, f in sorted(self.swap.items()))
        return f"BookPlay(fallback={self.fallback}, ties={self.ties}{within}{read}{swap}{zeb})"


def _legal_slots(state: ZebGameState) -> tuple[int, ...]:
    from forge.zeb.game import legal_actions
    return legal_actions(state)


def parse_book_spec(arg: str, log: bool = False) -> BookPlay:
    """`<fallback>,<ties>[,w<eps>][,read=<likelihood>[+calls]][/<walt options>]`, e.g.
    `zeb,least/k0,n512,F=net.npz,f=16`, `zeb,zeb,w0/k0,n512,F=net.npz,f=16` or
    `zeb,least,read=consistent+calls/k0,n512,F=net.npz,f=16`."""
    head, _, walt = arg.partition("/")
    fallback, _, rest = (head or "zeb,least").partition(",")
    ties, *rest = rest.split(",")
    within, read = None, None
    for t in rest:
        if t.startswith("read=") and t[5:]:
            read = t[5:]
        elif t.startswith("w") and t[1:]:
            within = float(t[1:])
        else:
            raise ValueError(f"book: unknown option {t!r}")
    zeb = None
    if walt:
        from arena.walt_play import parse_walt_spec
        zeb = parse_walt_spec(walt)
    return BookPlay(fallback, ties or "least", zeb, log, within, read)


# ------------------------------------------------------------------ bidding

class SilasBidder:
    """A bidding policy of `silas/examples/book/bidding/` as an arena
    `BidPolicy`: its call at each turn, read off the auction so far and the
    match score (in marks, or in points in the book's game), and its
    declaration (a pip trump; the arena's menu, the higher pip first). The
    arena's forced bid replaces the shaker's pass itself, as Silas's forced
    fallback would.

    The arena asks for the declaration with the hand and the bid alone, so
    the declaration is read off the shortest auction the winner wins with that
    bid (he opens it, the others pass), as Silas frames a hand known by its
    contract (`Auction::won_by`)."""

    def __init__(self, name: str):
        from scenario_core import SilasPolicy
        self.name = name
        self.policy = SilasPolicy((BOOK / "bidding" / f"{name}.silas").read_text(), 6)

    def bid(self, ctx: BidContext, rng: random.Random) -> int:
        order = bidding_order(ctx.dealer)
        calls = [ctx.bids[s] for s in order if ctx.bids[s] != -1]
        if ctx.points_to_win is None:
            match = dict(score=ctx.marks, scoring="marks", target=ctx.marks_to_win)
        else:
            match = dict(score=ctx.points, scoring="points", target=ctx.points_to_win)
        _, call = self.policy.bid(ctx.dealer, calls, ctx.seat, mask_of(ctx.hand), **match)
        return PASS if call == 0 else call

    def declare(self, hand: tuple[int, ...], bid: int, rng: random.Random) -> int:
        _, decl = self.policy.declare(3, [bid, PASS, PASS, PASS], 0, mask_of(hand))
        return decl

    def __repr__(self) -> str:
        return f"SilasBidder({self.name})"

