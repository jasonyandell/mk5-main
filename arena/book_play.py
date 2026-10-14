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

Variants (spec `book:<fallback>,<ties>[/<walt options>]`):

  fallback  lowest  no rule fires: the lowest legal tile (Silas's own fallback)
            zeb     no rule fires: Zeb's tile
  ties      least   a rule allows a set: its least tile (Silas's own decision)
            zeb     a rule allows a set: Zeb's best tile inside it, by Zeb's
                    q, ties to the lowest id (Zeb constrained by the book)

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
FALLBACKS = ("lowest", "zeb")
TIES = ("least", "zeb")


@lru_cache(maxsize=None)
def _text(path: Path) -> str:
    return path.read_text()


@lru_cache(maxsize=None)
def compiled(name: str, decl: int):
    """The book policy `name` compiled under Zeb's declaration `decl`."""
    from scenario_core import SilasPolicy
    return SilasPolicy(_text(BOOK / f"{name}.silas"), decl)


@dataclass(frozen=True)
class Verdict:
    """The stack at one decision: every policy that fired, as (rule, allowed
    tiles), and the one that decides (`policy` None: no rule fired)."""

    fired: dict[str, tuple[str, tuple[int, ...]]]
    policy: str | None
    rule: str | None
    allowed: tuple[int, ...]


def ask(state: ZebGameState, bid: int) -> Verdict:
    """Every book policy at the decision of `state`'s seat to play, under the
    contract `bid` as the table says it (a marks bid stays a marks bid)."""
    sit = situation_from_state(state, bid)
    seat = sit.viewer
    own = mask_of(t for t in state.hands[seat] if t not in state.played)
    f = frame(sit, own)
    fired = {}
    for name in POLICIES:
        layer, tiles = compiled(name, state.decl_id).allowed(f)
        if layer != "fallback":
            fired[name] = (layer, tuple(sorted(tiles)))
    order = LEAD_ORDER if not state.current_trick else FOLLOW_ORDER
    for name in order:
        if name in fired:
            rule, allowed = fired[name]
            return Verdict(fired, name, rule, allowed)
    return Verdict(fired, None, None, ())


class BookPlay:
    """The book player as an arena `PlayPolicy` (see the module docstring)."""

    def __init__(self, fallback: str = "zeb", ties: str = "least", zeb=None, log: bool = False):
        if fallback not in FALLBACKS or ties not in TIES:
            raise ValueError(f"book: fallback is one of {FALLBACKS}, ties one of {TIES}")
        if zeb is None and (fallback == "zeb" or ties == "zeb"):
            raise ValueError("book: a Zeb fallback or tie-break needs a Zeb (walt options)")
        self.fallback, self.ties, self.zeb = fallback, ties, zeb
        self.log: list[dict] | None = [] if log else None

    def choose(
        self,
        states: Sequence[ZebGameState],
        bid_values: Sequence[int],
        marks: Sequence[tuple[int, int]] | None = None,
        marks_to_win: int = 7,
    ) -> list[int]:
        verdicts = [ask(s, int(b)) for s, b in zip(states, bid_values)]
        zeb = (self.zeb.walt.decide_batch(states, [contract_points(int(b)) for b in bid_values])
               if self.zeb is not None else [(None, {})] * len(states))
        slots = []
        for state, bid, v, (z_tile, q) in zip(states, bid_values, verdicts, zeb):
            seat = (state.trick_leader + len(state.current_trick)) % 4
            legal = sorted(state.hands[seat][i] for i in _legal_slots(state))
            if v.policy is None:
                tile = legal[0] if self.fallback == "lowest" else z_tile
            elif self.ties == "zeb":
                tile = max(v.allowed, key=lambda t: (q.get(t, float("-inf")), -t))
            else:
                tile = v.allowed[0]
            if self.log is not None:
                self.log.append(dict(
                    hands=state.hands, plays=state.play_history, decl=state.decl_id, bidder=state.bidder,
                    bid=int(bid), seat=seat, legal=legal, zeb=z_tile,
                    q={int(t): float(x) for t, x in q.items()},
                    fired={k: [r, list(a)] for k, (r, a) in v.fired.items()},
                    policy=v.policy, rule=v.rule, allowed=list(v.allowed), tile=tile,
                ))
            slots.append(list(state.hands[seat]).index(tile))
        return slots

    def __repr__(self) -> str:
        zeb = "" if self.zeb is None else f", zeb={self.zeb!r}"
        return f"BookPlay(fallback={self.fallback}, ties={self.ties}{zeb})"


def _legal_slots(state: ZebGameState) -> tuple[int, ...]:
    from forge.zeb.game import legal_actions
    return legal_actions(state)


def parse_book_spec(arg: str, log: bool = False) -> BookPlay:
    """`<fallback>,<ties>[/<walt options>]`, e.g. `zeb,least/k0,n512,F=net.npz,f=16`."""
    head, _, walt = arg.partition("/")
    fallback, _, ties = (head or "zeb,least").partition(",")
    zeb = None
    if walt:
        from arena.walt_play import parse_walt_spec
        zeb = parse_walt_spec(walt)
    return BookPlay(fallback, ties or "least", zeb, log)


# ------------------------------------------------------------------ bidding

class SilasBidder:
    """A bidding policy of `silas/examples/book/bidding/` as an arena
    `BidPolicy`: its call at each turn, read off the auction so far and the
    match score, and its declaration (a pip trump; the arena's menu, the
    higher pip first). The arena's forced bid replaces the shaker's pass
    itself, as Silas's forced fallback would.

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
        _, call = self.policy.bid(ctx.dealer, calls, ctx.seat, mask_of(ctx.hand),
                                  score=ctx.marks, marks_to_win=ctx.marks_to_win)
        return PASS if call == 0 else call

    def declare(self, hand: tuple[int, ...], bid: int, rng: random.Random) -> int:
        _, decl = self.policy.declare(3, [bid, PASS, PASS, PASS], 0, mask_of(hand))
        return decl

    def __repr__(self) -> str:
        return f"SilasBidder({self.name})"

