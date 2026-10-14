"""Zeb's bidder in the arena: the package's `forecast` + `chooseBid`, and
the match-aware option (`arena/match_equity.py`), default off.

The forecast is `@jasonyandell/zeb`'s, computed by the same planner: for
each pip declaration, one plan from the bidder's chair as seat 0 before the
opening lead under a 30 contract (`walt:k0,n<worlds>,F=<net>,f=16`, seed
`seed`), its chosen lead's tails read at every target 30..42 (pmake(t) =
the belief's share in which the bidding side banks at least t). Only the
bidder's seven tiles enter: the other seats are filled with a fixed
arrangement of the 21 unseen tiles the planner never reads (it samples them).

`chooseBid` (zeb/index.js) is the hand-only rule: the highest target, at
least the lowest legal bid, some declaration makes with probability at least
`cutoff` (0.8), leaning on the declaration most likely to make it; else pass.
Plunge's two auction rules ride along: never overcall the partner; the
arena forces the shaker to 30 after its reshakes. It never bids 84.
The partner rule is a convention, not a law, so the mark-bid option is
priced over a partner's point bid too.

`met` adds one option at the bidder's turn, a mark bid (84, or one mark over
an 84): its make condition is all 42 points, so its pmake is the forecast's
at 42. It is taken when its match equity beats the hand-only call's,
E[V | 84] > E[V | c0]; c0's value counts the field's overcalls and what
passing leaves (`Odds`). With `marks`, the same option priced by the
expected mark margin instead (the score-blind yardstick).

Two one-mark repairs for the turn when an opponent holds the high bid
(#160, wiki/experiments/match-aware-bidding-2026-10-14.md § Follow-ups),
both default off:

  * `cal`: the forecast there is replaced by a calibrated one, logit p̃ =
    a + b·logit p (`Shrink`, fitted on the prior study's over-an-opponent
    contracts). The hand-only forecast never sees the opponent's bid, and
    over one it runs 3–8 points hot. A monotone map: the declaration it
    leans on is unchanged.
  * `over`: when the 0.8 rule passes, overcall with the call whose chance
    of taking the mark beats passing's (`Odds.pass_ours`, 17–20% over an
    opponent). Every call from pass to 42 moves one mark, so the one with
    the larger π is the better call at every score (the one-mark theorem):
    a score-blind change.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from forge.zeb.types import BidState, GamePhase, ZebGameState

from .auction import ONE_MARK, PASS, TWO_MARKS, BidContext
from .match_equity import HandLaw, Linear, MatchEquity, Odds, mark_bid_value, one_mark_value

PIP_DECLS = tuple(range(7))
TARGETS = tuple(range(30, 43))


def forecast(walt, hand: tuple[int, ...], decls=PIP_DECLS) -> list[tuple[int, tuple[float, ...]]]:
    """(decl, pmake at 30..42) per declaration for `hand` as seat 0, the
    package's `Player::forecast`."""
    hand = tuple(sorted(hand))
    unseen = [t for t in range(28) if t not in hand]
    hands = (hand, tuple(unseen[0:7]), tuple(unseen[7:14]), tuple(unseen[14:21]))
    states = [
        ZebGameState(
            hands=hands, dealer=3, phase=GamePhase.PLAYING,
            bid_state=BidState((30, 0, 0, 0), 0, 30), decl_id=d, bidder=0,
            played=frozenset(), play_history=(), current_trick=(), trick_leader=0,
            team_points=(0, 0),
        )
        for d in decls
    ]
    out = []
    for d, a in zip(decls, walt.analyze_batch(states, [30] * len(states))):
        total = sum(a.weights)
        margins = a.outcomes[a.tile]
        out.append((d, tuple(
            sum(w for m, w in zip(margins, a.weights) if m >= 2 * t - 42) / total for t in TARGETS)))
    return out


def choose_bid(fc, need: int, cutoff: float) -> tuple[int, int, float] | None:
    """zeb/index.js `chooseBid`: (bid, decl, pmake) or None to pass."""
    for bid in range(42, need - 1, -1):
        best = None
        for decl, pm in fc:
            p = pm[bid - 30]
            if p >= cutoff and (best is None or p > best[2]):
                best = (bid, decl, p)
        if best is not None:
            return best
    return None


def best_at(fc, points: int) -> tuple[int, float]:
    """(decl, pmake) of the declaration most likely to take `points` (first on a tie)."""
    decl, p = fc[0][0], -1.0
    for d, pm in fc:
        if pm[points - 30] > p:
            decl, p = d, pm[points - 30]
    return decl, p


# The deployed field's law (Zeb bids at n160 and plays, both sides), fitted on the FIT block
# of wiki/experiments/match-aware-bidding-2026-10-14.md (384 games, 4,251 hands; HF
# zeb/match-aware-bidding/, out/model.json) and frozen before the primary was planned.
# win_flip[w][f]: w = the first shaker's side took the mark, f = the next first shaker
# sits on the other side. A cell under 20 turns holds its holder's pooled value.
ZEB_LAW = HandLaw(((0.1682, 0.3312), (0.1701, 0.3305)))
ZEB_ODDS = Odds(
    keep={("none", 0, "low"): 1.0, ("none", 0, "high"): 1.0, ("none", 1, "low"): 0.9212,
          ("none", 1, "high"): 0.9913, ("none", 2, "low"): 0.938, ("none", 2, "high"): 0.9891,
          ("none", 3, "low"): 0.8818, ("none", 3, "high"): 0.9871, ("opp", 0, "low"): 1.0,
          ("opp", 0, "high"): 1.0, ("opp", 1, "low"): 0.9901, ("opp", 1, "high"): 1.0,
          ("opp", 2, "low"): 0.9901, ("opp", 2, "high"): 1.0, ("opp", 3, "low"): 0.9901,
          ("opp", 3, "high"): 0.9901},
    over_set=0.1727,
    pass_ours={("none", 0): 0.5049, ("none", 1): 0.4285, ("none", 2): 0.5053, ("none", 3): 0.4445,
               ("partner", 0): 0.8254, ("partner", 1): 0.7963, ("partner", 2): 0.81,
               ("partner", 3): 0.81, ("opp", 0): 0.1688, ("opp", 1): 0.1746, ("opp", 2): 0.2037,
               ("opp", 3): 0.1799},
)


@dataclass(frozen=True)
class Shrink:
    """A calibration of the forecast: logit p̃ = a + b·logit p (p clipped to
    [EPS, 1 − EPS]). Strictly increasing for b > 0; (0, 1) is the identity."""

    a: float
    b: float
    EPS = 1e-4

    def __call__(self, p: float) -> float:
        p = min(max(p, self.EPS), 1.0 - self.EPS)
        return 1.0 / (1.0 + math.exp(-(self.a + self.b * math.log(p / (1.0 - p)))))

    def forecast(self, fc):
        return [(d, tuple(self(p) for p in pm)) for d, pm in fc]


def overcall(fc, need: int, odds: Odds, key: tuple[str, int]) -> tuple[int, int, float] | None:
    """(bid, decl, pmake) of the one-mark call need..42 with the largest π
    (lowest bid on a tie), when that π beats passing's; else None."""
    best, pi = None, odds.pass_ours[key]
    for bid in range(need, ONE_MARK + 1):
        decl, p = best_at(fc, bid)
        if odds.take(bid, p, key) > pi:
            best, pi = (bid, decl, p), odds.take(bid, p, key)
    return best


# The forecast over an opponent's standing bid, calibrated (`cal`): fitted by maximum
# likelihood on every over-an-opponent contract that stood in the prior study's BASE and P
# blocks (HF zeb/match-aware-bidding/; zeb/match-aware-bidding-2/fit_cal.py), frozen
# before the follow-up's plan ran (out/cal.json).
ZEB_SHRINK = Shrink(-0.2483, 0.926)  # 2,600 contracts; se (0.049, 0.035); p 0.2 -> 0.178, 0.8 -> 0.738


def price(v, odds: Odds, us: int, them: int, nd: int, key: tuple[str, int], c0: int, p_c0: float,
          mark_call: int, p42: float) -> tuple[float, float, int]:
    """(E[V | c0], E[V | mark_call], the call with the larger): the
    hand-only call against the mark bid, by the yardstick v (a match equity,
    or `Linear`); a tie keeps c0."""
    e_c0 = one_mark_value(v, us, them, nd, odds.take(c0, p_c0, key))
    e_mark = mark_bid_value(v, us, them, nd, mark_call // ONE_MARK, p42)
    return e_c0, e_mark, (mark_call if e_mark > e_c0 else c0)


@dataclass
class Decision:
    """One bidder turn, for the record: the score, the hand-only call, the
    mark-bid option and their values under the bidder's yardstick."""

    seat: int
    marks: tuple[int, int]  # (us, them)
    dealer: int
    nd: int  # 1 when our side shakes first next hand
    key: tuple[str, int]
    c0: int
    p_c0: float
    mark_call: int
    p42: float
    e_c0: float
    e_mark: float
    call: int
    tag: object = None  # the caller's label for this turn (`ZebBidder.tag`), for joining to the hand


class ZebBidder:
    """Spec `zeb[:n<worlds>][,c<cutoff>][,s<seed>][,met|marks][,over][,cal]`."""

    def __init__(self, walt, cutoff: float = 0.8, mode: str = "hand",
                 law: HandLaw = ZEB_LAW, odds: Odds = ZEB_ODDS, label: str = "zeb",
                 over: bool = False, cal: bool = False, shrink: Shrink | None = None):
        if mode not in ("hand", "met", "marks"):
            raise ValueError(f"zeb bidder mode is hand, met or marks, not {mode!r}")
        self.walt = walt
        self.cutoff = cutoff
        self.mode = mode
        self.over = over
        self.cal = cal
        self.shrink = ZEB_SHRINK if shrink is None else shrink
        self.law = law
        self.odds = odds
        self.label = label
        self._fc: dict[tuple[int, ...], list] = {}
        self.decisions: list[Decision] = []
        self._yard = {"met": lambda target: MatchEquity(law, target), "marks": lambda target: Linear()}
        self._v: dict[int, object] = {}
        self.tag = None  # a callable the harness may set: its value labels each recorded Decision

    def forecast(self, hand: tuple[int, ...]):
        key = tuple(sorted(hand))
        if key not in self._fc:
            self._fc[key] = forecast(self.walt, key)
        return self._fc[key]

    def bid(self, ctx: BidContext, rng: random.Random) -> int:
        partner = ctx.high_seat != -1 and ctx.high_seat % 2 == ctx.team
        if partner and self.mode == "hand" and self.tag is None:
            return PASS  # never overcall the partner (Plunge's rule); nothing to price
        return self.decide(self.forecast(ctx.hand), ctx)

    def decide(self, fc, ctx: BidContext) -> int:
        """The call at this turn from the hand's forecast `fc` (`bid` without the forecast)."""
        partner = ctx.high_seat != -1 and ctx.high_seat % 2 == ctx.team
        holder = "none" if ctx.high_seat == -1 else "partner" if partner else "opp"
        after = sum(1 for b in ctx.bids if b == -1) - 1
        key = (holder, after)
        if holder == "opp" and self.cal:
            fc = self.shrink.forecast(fc)
        need = min(ctx.legal)
        hand_only = None if partner or need > ONE_MARK else choose_bid(fc, need, self.cutoff)
        if hand_only is None and self.over and holder == "opp" and need <= ONE_MARK:
            hand_only = overcall(fc, need, self.odds, key)
        c0, p_c0 = (PASS, 0.0) if hand_only is None else (hand_only[0], hand_only[2])
        if ctx.points_to_win is not None:
            return c0  # the match equity here is the marks game's
        mark_call = max(ctx.legal) if max(ctx.legal) >= TWO_MARKS else None
        if mark_call is None:
            return c0
        # The hand-only bidder prices the option too (by the match equity),
        # for the record; only `met` and `marks` act on it.
        yard = "marks" if self.mode == "marks" else "met"
        v = self._v.get((yard, ctx.marks_to_win))
        if v is None:
            v = self._v[(yard, ctx.marks_to_win)] = self._yard[yard](ctx.marks_to_win)
        us, them = ctx.marks[ctx.team], ctx.marks[1 - ctx.team]
        nd = int((ctx.dealer + 1) % 2 == ctx.team)  # we shake first next hand
        _, p42 = best_at(fc, 42)
        e_c0, e_mark, priced = price(v, self.odds, us, them, nd, key, c0, p_c0, mark_call, p42)
        self.decisions.append(Decision(ctx.seat, (us, them), ctx.dealer, nd, key, c0, p_c0, mark_call,
                                       p42, e_c0, e_mark, priced, self.tag() if self.tag else None))
        return c0 if self.mode == "hand" else priced

    def declare(self, hand: tuple[int, ...], bid: int, rng: random.Random) -> int:
        return best_at(self.forecast(hand), min(bid, ONE_MARK))[0]

    def __repr__(self) -> str:
        return f"ZebBidder(cutoff={self.cutoff}, mode={self.mode}, over={self.over}, cal={self.cal})"


SHIPPED_NET = "zeb/net/margin_d250k_h128x64.fnet"


def parse_zeb_bidder(arg: str) -> ZebBidder:
    """`zeb[:n<worlds>][,c<cutoff>][,s<seed>][,met|marks|hand][,over][,cal]`:
    the forecast at `worlds` (default 512, the graded setting) over the
    shipped net."""
    from pathlib import Path

    from scenario.compiled import CompiledWalt
    from scenario.field import FieldNet
    from scenario.planner import Config, Rung

    worlds, cutoff, seed, mode, flags = 512, 0.8, 0, "hand", {"over": False, "cal": False}
    for t in (p for p in arg.split(",") if p):
        if t in ("hand", "met", "marks"):
            mode = t
        elif t in flags:
            flags[t] = True
        elif t[0] in "ncs" and t[1:]:
            if t[0] == "n":
                worlds = int(t[1:])
            elif t[0] == "c":
                cutoff = float(t[1:])
            else:
                seed = int(t[1:])
        else:
            raise ValueError(f"unknown zeb bidder option {t!r}")
    net = FieldNet.from_bytes((Path(__file__).parent.parent / SHIPPED_NET).read_bytes())
    walt = CompiledWalt(Config((Rung(worlds, None, 16),), "make", seed, field=net))
    return ZebBidder(walt, cutoff, mode, label=f"zeb:{arg}", **flags)
