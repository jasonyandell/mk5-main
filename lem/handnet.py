# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""A field net written by hand: every weight chosen, none learned.

Zeb's field net (`scenario.field.FieldNet`) is an integer MLP over the
mover's information, read in the canonical frame where trump is pip 6. This
module writes one from rules instead of training it. Every neuron is a named
rule in the game's words, and its weights are small integers that compile the
rule exactly:

- **Layer 1 reads the table.** Each unit is a threshold over the input bits:
  "all of these, none of those", which is 0 or 1 with weights of ±1 and a
  bias. A count ("a trump is still out") is a pair of units whose difference
  is the 0/1 step, so layer 2 can use it as one literal.
- **Layer 2 states the rules.** Each unit is a conjunction of layer-1
  literals ("I am following, an opponent holds the trick, and this tile takes
  it"), again exactly 0 or 1.
- **The output ranks the tiles.** A rule unit votes for the tiles it is about
  with its priority; the static bias breaks ties toward the tile cheapest to
  give up. The net plays the highest-scoring legal tile, ties to the lowest id.

Shifts are zero, so the arithmetic is exact and every unit's value is the
truth of its rule on that row (`lem/tests/test_handnet.py` checks that each
boolean unit takes only 0 and 1 on real rows).

Rules come from the book (*Winning 42*, `silas/lib/book.silas` names them);
the unit papers (wiki/experiments/lem-unit-*.md) showed which ones a trained
net builds. Nothing here reads a hidden hand: the inputs are the field net's
own, the mover's information only.

    python -u -m lem.handnet OUT_DIR       # writes hand.npz, hand.fnet, neurons.md
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from forge.oracle.declarations import PIP_TRUMP_IDS
from forge.oracle.tables import DOMINO_COUNT_POINTS, DOMINO_HIGH, DOMINO_LOW
from scenario.field import CANON, N_FEATURES_OF, OFFSET, FieldNet
from scenario.rules import N_TILES, rules_for

LAYOUT = "field-v2"
FRAME = rules_for(6)                       # the canonical frame is sixes trump
TRUMP = 7                                  # the called suit's index in the rule tables
SEATS = ("left", "partner", "right")       # the trick, void and winning blocks' order
LEADER = {1: "right", 2: "partner", 3: "left"}   # who led, by how many tiles are on the trick


def name(t: int) -> str:
    return f"{DOMINO_HIGH[t]}-{DOMINO_LOW[t]}"


def is_trump(t: int) -> bool:
    return FRAME.led_suit[t] == TRUMP


def follows(t: int, suit: int) -> bool:
    return bool((FRAME.follow_bits[suit] >> t) & 1)


def suits_of(t: int) -> list[int]:
    """The suits a canonical tile follows: trump, or its one or two pips."""
    return [s for s in range(8) if follows(t, s)]


def can_take_in(t: int, s: int) -> bool:
    """Whether t, following suit s, can ever take a trick led in s: some tile
    that leads s ranks below it (an x-0 never beats the 0-0, which alone leads blanks)."""
    beaters = set(above(t, s))
    return any(FRAME.led_suit[u] == s and u != t and u not in beaters for u in range(N_TILES)) or FRAME.led_suit[t] == s


def above(t: int, suit: int) -> list[int]:
    """Tiles that beat t within `suit` (t following it)."""
    r = FRAME.rank[suit]
    return [u for u in range(N_TILES) if follows(u, suit) and r[u] > r[t]]


TRUMPS = [t for t in range(N_TILES) if is_trump(t)]
OFFS = [t for t in range(N_TILES) if not is_trump(t)]


# -- the inputs, by name

def f_hand(t): return OFFSET["hand"] + t
def f_gone(t): return OFFSET["gone"] + t
def f_trick(seat, t): return OFFSET["trick"] + SEATS.index(seat) * N_TILES + t
def f_void(seat, t): return OFFSET["void"] + SEATS.index(seat) * N_TILES + t
def f_ten(t): return OFFSET["count"] + t
def f_five(t): return OFFSET["count"] + N_TILES + t
def f_open(n): return OFFSET["open"] + n
def f_winning(seat): return OFFSET["winning"] + SEATS.index(seat)
F_SIDE = OFFSET["side"]


@dataclass
class Unit:
    name: str
    rule: str                      # the rule in the game's words
    why: str                       # where it comes from
    terms: dict                    # input index (layer 1) or layer-1 unit name (layer 2) -> weight
    bias: int
    out: dict = field(default_factory=dict)     # layer 2 only: canonical tile -> vote
    kind: str = "bool"             # "bool": exactly 0/1; "step": half of a step pair


class Builder:
    def __init__(self) -> None:
        self.l1: dict[str, Unit] = {}
        self.l2: dict[str, Unit] = {}
        self.b3 = np.zeros(N_TILES, np.int64)
        self.b3_why = ""

    def one(self, name, rule, why, terms, bias, kind="bool") -> str:
        assert name not in self.l1, name
        self.l1[name] = Unit(name, rule, why, dict(terms), bias, kind=kind)
        return name

    def all_none(self, name, rule, why, req=(), forbid=(), groups=()) -> str:
        """1 when every required input is on (each `groups` entry is a set of
        mutually exclusive inputs, one of which must be on) and no forbidden one is."""
        terms: dict[int, int] = {}
        for i in req:
            terms[i] = terms.get(i, 0) + 1
        for g in groups:
            for i in g:
                terms[i] = terms.get(i, 0) + 1
        for i in forbid:
            terms[i] = terms.get(i, 0) - 2
        return self.one(name, rule, why, terms, 1 - len(req) - len(groups))

    def step(self, name, rule, why, inputs, at_least: int) -> tuple[str, str]:
        """A step pair over a count of inputs: (u, v) with u − v = [count >= at_least]."""
        terms = {i: 1 for i in inputs}
        u = self.one(name + "+", rule, why, terms, 1 - at_least, kind="step")
        v = self.one(name + "−", rule + " (the step's upper half)", why, terms, -at_least, kind="step")
        return u, v

    def rule(self, name, rule, why, lits, out, nots=()) -> str:
        """A layer-2 conjunction: every literal true, every `nots` literal false.
        A literal is a layer-1 unit name, a step pair (u, v), or a list of mutually
        exclusive literals (their sum is the OR)."""
        terms: dict[str, int] = {}

        def add(lit, w):
            if isinstance(lit, tuple):
                terms[lit[0]] = terms.get(lit[0], 0) + w
                terms[lit[1]] = terms.get(lit[1], 0) - w
            elif isinstance(lit, list):
                for x in lit:
                    add(x, w)
            else:
                terms[lit] = terms.get(lit, 0) + w

        for lit in lits:
            add(lit, 1)
        for lit in nots:
            add(lit, -2)
        assert name not in self.l2, name
        self.l2[name] = Unit(name, rule, why, terms, 1 - len(lits), out=dict(out))
        return name

    def compile(self) -> FieldNet:
        n = N_FEATURES_OF[LAYOUT]
        l1 = list(self.l1)
        l2 = list(self.l2)
        w1 = np.zeros((n, len(l1)), np.int64)
        b1 = np.zeros(len(l1), np.int64)
        for j, k in enumerate(l1):
            u = self.l1[k]
            for i, w in u.terms.items():
                w1[i, j] += w
            b1[j] = u.bias
        w2 = np.zeros((len(l2), len(l1)), np.int64)
        b2 = np.zeros(len(l2), np.int64)
        w3 = np.zeros((N_TILES, len(l2)), np.int64)
        for j, k in enumerate(l2):
            u = self.l2[k]
            for nm, w in u.terms.items():
                w2[j, l1.index(nm)] += w
            b2[j] = u.bias
            for t, v in u.out.items():
                w3[t, j] += v
        return FieldNet(w1, b1, w2, b2, w3, self.b3.copy(), 0, 0, LAYOUT)


# -- the declaration, told by where the count sits in the canonical frame

def count_slots(d: int) -> tuple[list[int], list[int]]:
    canon = CANON[d]
    tens = sorted(canon[t] for t in range(N_TILES) if DOMINO_COUNT_POINTS[t] == 10)
    fives = sorted(canon[t] for t in range(N_TILES) if DOMINO_COUNT_POINTS[t] == 5)
    return tens, fives


PIP_NAMES = ("blanks", "ones", "twos", "threes", "fours", "fives", "sixes")

# The output's scale: a rule's vote is its priority, and a higher rule outvotes
# every lower one. Chosen by hand; `lem.handnet` lists every variant tried on the
# dev split (training deals, never the graded rows) in the experiment page.
P = dict(
    take=4000,          # an opponent holds the trick and this tile takes it
    secure=3200,        # partner's lead can lose its suit: take it over with trump
    pull=2800,          # on lead with trump out: draw trump (bidders with two, anyone with three)
    boss=300,           # ... with the boss trump first
    walk_double=2600,   # on lead: a double, which cannot lose its suit
    walk=2400,          # ... or another tile that cannot lose its suit
    count=60,           # per point of count, given or kept
    take_count_trump=0,   # per point, a trump that takes with count: none (it can be overtrumped)
    feed_third=20,      # per point of count, to partner's lead that cannot lose its suit
    low=4,              # per step of strength: give up the weakest tile
)


def strength(t: int) -> int:
    """How much a tile is worth keeping, 0 (a blank off) up: trumps above
    every off, the higher trump higher; among offs, doubles above singles,
    then by pips."""
    if is_trump(t):
        return 40 + FRAME.rank[TRUMP][t] % 64
    hi, lo = DOMINO_HIGH[t], DOMINO_LOW[t]
    return (14 if hi == lo else 0) + 2 * hi + lo // 2


def build(P: dict = P) -> Builder:
    B = Builder()

    # ---- layer 1: the table as it stands
    lead = B.all_none("lead", "I lead the trick", "position", req=[f_open(0)])
    follow = B.all_none("follow", "I play to a trick already led", "position",
                        groups=[[f_open(1), f_open(2), f_open(3)]])
    last = B.all_none("last", "I play last to the trick", "position", req=[f_open(3)])
    pwin = B.all_none("partner-winning", "partner holds the trick so far", "book: partner-winning",
                      req=[f_winning("partner")])
    owin = B.all_none("opponent-winning", "an opponent holds the trick so far", "book: ¬partner-winning",
                      groups=[[f_winning("left"), f_winning("right")]])
    side = B.all_none("bidding-side", "I am the bidder or his partner", "book: bidder, bidders-partner",
                      req=[F_SIDE])

    # one gate per count layout: twos and threes put the count in the same canonical
    # slots, so the frame cannot tell them apart and needs no second gate
    layouts: dict[tuple, list[int]] = {}
    for d in PIP_TRUMP_IDS:
        tens, fives = count_slots(d)
        layouts.setdefault((tuple(tens), tuple(fives)), []).append(d)
    decl = {}
    for (tens, fives), ds in layouts.items():
        nm = "/".join(PIP_NAMES[d] for d in ds)
        decl[ds[0]] = B.all_none(f"decl-{nm}", f"{nm} are trump (told by where the count sits)",
                                 "the count block", req=[f_ten(t) for t in tens] + [f_five(t) for t in fives])
    DECLS = list(decl)

    led = {}
    for s in list(range(6)) + [TRUMP]:
        sn = "trump" if s == TRUMP else str(s)
        for n, seat in LEADER.items():
            led[s, n] = B.all_none(f"led-{sn}@{n}", f"the trick was led in {sn}s by {seat} ({n} on it)",
                                   "the led suit", req=[f_open(n)],
                                   groups=[[f_trick(seat, u) for u in range(N_TILES) if FRAME.led_suit[u] == s]])

    # "fewer than 7 of the trumps are gone or mine": u = max(7 − Σ, 0), v = max(6 − Σ, 0), u − v = [Σ <= 6]
    terms = {f_gone(t): -1 for t in TRUMPS} | {f_hand(t): -1 for t in TRUMPS}
    trump_out = (B.one("trump-out+", "a trump outside my hand is still live", "book: trump-out", terms, 7, "step"),
                 B.one("trump-out−", "a trump outside my hand is still live (the step's upper half)",
                       "book: trump-out", terms, 6, "step"))
    two_trumps = B.step("trumps-2+", "I hold two trumps or more", "book: exactly-two-trumps",
                        [f_hand(t) for t in TRUMPS], 2)
    three_trumps = B.step("trumps-3+", "I hold three trumps or more", "book: exactly-three-trumps",
                          [f_hand(t) for t in TRUMPS], 3)

    # per tile: can it lose its suit when led; does it take the open trick
    top = {}
    for t in range(N_TILES):
        s = FRAME.led_suit[t]
        hi = above(t, s)
        terms = {}
        for u in hi:
            terms[f_gone(u)] = 1
            terms[f_hand(u)] = 1
        word = "boss trump" if s == TRUMP else "walker"
        top[t] = B.one(f"{word}:{name(t)}", f"no live tile outside my hand beats the {name(t)} when it leads",
                       f"book: {'boss-trump / sure-trump' if s == TRUMP else 'walker'}", terms, 1 - len(hi))

    takes = {}          # (t, suit) -> layer-1 unit: nothing on the trick beats t, read under that suit
    for t in range(N_TILES):
        for s in suits_of(t):
            if not can_take_in(t, s):
                continue
            blockers = set(above(t, s))
            if s != TRUMP:
                blockers |= set(TRUMPS)
            sn = "trump" if s == TRUMP else str(s)
            takes[t, s] = B.all_none(f"tops:{name(t)}/{sn}", f"nothing on the trick beats the {name(t)} in {sn}s",
                                     "book: can-take-it", forbid=[f_trick(seat, u) for seat in SEATS for u in sorted(blockers)])

    # count on the open trick, one step pair per count layout: S counts the trick's
    # tiles in that layout's count slots, K its five count bits (5 only under it);
    # u = max(S + 3K − 15, 0), v = max(S + 3K − 16, 0), u − v = [S >= 1] under the
    # layout and 0 under any other (S <= 3, K <= 4 there)
    count_on = []
    for d in DECLS:
        tens, fives = count_slots(d)
        dn = B.l1[decl[d]].name[5:]
        terms = {f_trick(seat, u): 1 for seat in SEATS for u in tens + fives}
        terms |= {f_ten(u): 3 for u in tens} | {f_five(u): 3 for u in fives}
        count_on.append((B.one(f"count-on-trick:{dn}+", f"count is on the trick ({dn} trump)",
                               "book: count-on-trick", terms, -15, "step"),
                         B.one(f"count-on-trick:{dn}−", f"count is on the trick ({dn} trump; the step's upper half)",
                               "book: count-on-trick", terms, -16, "step")))
    ten1, five1 = {}, {}
    for t in range(N_TILES):
        if any(t in count_slots(d)[0] for d in DECLS):
            ten1[t] = B.all_none(f"ten:{name(t)}", f"the {name(t)} is a ten under this trump", "book: count-tile",
                                 req=[f_ten(t)])
        if any(t in count_slots(d)[1] for d in DECLS):
            five1[t] = B.all_none(f"five:{name(t)}", f"the {name(t)} is a five under this trump", "book: count-tile",
                                  req=[f_five(t)])
    # partner led a tile no live tile outside my hand beats in its suit
    ptop = []
    for u in range(N_TILES):
        hi = above(u, FRAME.led_suit[u])
        terms = {f_trick("partner", u): 1} | {f_gone(w): 1 for w in hi} | {f_hand(w): 1 for w in hi}
        ptop.append(B.one(f"partner-led-top:{name(u)}", f"partner led the {name(u)} and nothing live outside my hand beats it in its suit",
                          "book: winner-cannot-be-beaten (as far as the suit goes)", terms, -len(hi)))
    third = B.all_none("third", "I play third: partner led", "position", req=[f_open(2)])

    # ---- layer 2: the rules
    for t in range(N_TILES):
        for s_ in suits_of(t):
            if (s_ == TRUMP and not is_trump(t)) or not can_take_in(t, s_):
                continue
            sn = "trump" if s_ == TRUMP else str(s_)
            if is_trump(t):
                # a trump takes a trick led in trump or, I being void, in any suit;
                # last to a countless trick it buys one point for a trump, so it ranks lower
                tk = B.rule(f"take:{name(t)}", f"an opponent holds the trick and the {name(t)} takes it",
                            "book: can-take-it", [follow, owin, takes[t, s_]], {t: P["take"]})
                B.rule(f"secure:{name(t)}", f"partner led a tile that can lose its suit and the {name(t)} takes the trick",
                       "book: protect partner's trick", [third, pwin, takes[t, s_]], {t: P["secure"]}, nots=[ptop])
            else:
                led_s = [led[s_, n] for n in (1, 2, 3)]
                B.rule(f"take:{name(t)}/{sn}", f"an opponent holds a trick led in {sn}s and the {name(t)} takes it",
                       "book: can-take-it", [led_s, owin, takes[t, s_]], {t: P["take"]})
            # taking with count banks it: undo keep-count's vote against the tile and add its count
            when = [follow, owin, takes[t, s_]] if is_trump(t) else [[led[s_, n] for n in (1, 2, 3)], owin, takes[t, s_]]
            for pts, unit in ((10, ten1.get(t)), (5, five1.get(t))):
                if unit:
                    B.rule(f"take-{'ten' if pts == 10 else 'five'}:{name(t)}" + ("" if is_trump(t) else f"/{sn}"),
                           f"... and the {name(t)} is a {'ten' if pts == 10 else 'five'}: take with the count",
                           "book: count on your own trick", when + [unit],
                           {t: pts * (P["count"] + (P["take_count_trump"] if is_trump(t) else P["count"]))})

    for d in DECLS:
        tens, fives = count_slots(d)
        give = {t: 10 * P["count"] for t in tens} | {t: 5 * P["count"] for t in fives}
        keep = {t: -v for t, v in give.items()}
        dn = B.l1[decl[d]].name[5:]
        B.rule(f"feed:{dn}", "partner holds the trick and I play last: give it count",
               "book: safe-trick (count to a safe trick)", [pwin, last, decl[d]], give)
        B.rule(f"feed-third:{dn}", "partner led a tile that cannot lose its suit: give it count",
               "book: safe-trick", [third, pwin, ptop, decl[d]],
               {t: v * P["feed_third"] // P["count"] for t, v in give.items()})
        B.rule(f"keep-count:{dn}", "an opponent holds the trick: keep my count off it",
               "book: don't give count to the setters' trick", [owin, decl[d]], keep)
        B.rule(f"lead-no-count:{dn}", "on lead: don't lead count that can lose",
               "book: an off calls for count", [lead, decl[d]], keep)

    B.rule("pull", "on lead for the bidders, two trumps or more, trump out: draw trump",
           "book: draw trump", [lead, side, trump_out, two_trumps], {t: P["pull"] for t in TRUMPS})
    B.rule("pull-long", "on lead, three trumps or more, trump out: draw trump",
           "book: four-trumps-or-more (lead trump)", [lead, trump_out, three_trumps],
           {t: P["pull"] for t in TRUMPS}, nots=[side])
    for t in TRUMPS:
        B.rule(f"boss:{name(t)}", f"on lead with trump out: the boss trump {name(t)} before a lower one",
               "book: draw with the top trump", [lead, trump_out, top[t]], {t: P["boss"]})
    for t in OFFS:
        dbl = DOMINO_HIGH[t] == DOMINO_LOW[t]
        B.rule(f"walk:{name(t)}", f"on lead: the {name(t)} cannot lose its suit; lead it",
               "book: walker (lead your doubles)", [lead, top[t]], {t: P["walk_double"] if dbl else P["walk"]})

    # the static ranking: give up the weakest tile
    B.b3 = np.array([-P["low"] * strength(t) for t in range(N_TILES)], np.int64)
    B.b3_why = "otherwise play the weakest tile: offs before trumps, singles before doubles, low pips first"
    return B


def with_decoys(B: Builder) -> Builder:
    """The audit's copy: three layer-1 units nothing reads (no layer-2 weight, so
    play is unchanged), keyed before the audit runs, so the instrument meets a
    grouping it should call related, one it should not, and a near miss."""
    doubles = [t for t in range(N_TILES) if DOMINO_HIGH[t] == DOMINO_LOW[t]]
    tens6, fives6 = count_slots(5)               # fives trump: the 6-6 slot holds a ten
    B.one("decoy-offshoot", "partner holds the trick (2), or partner led it (1): one idea and a weaker related facet",
          "decoy: recursive generalization", {f_winning("partner"): 2, f_open(2): 1}, 0, kind="decoy")
    B.one("decoy-unrelated", "partner holds the trick, or fives are trump: two unrelated things",
          "decoy: arbitrary packing", {f_winning("partner"): 1, f_ten(tens6[1]): 1}, 0, kind="decoy")
    B.one("decoy-near-miss", "I lead and hold no double",
          "decoy: a near miss of `lead`", {f_open(0): 1} | {f_hand(t): -2 for t in doubles}, 0, kind="decoy")
    return B


def random_twin(B: Builder, seed: int) -> Builder:
    """A structure-matched null: every unit replaced by a conjunction of the same
    arity and signs over random inputs (layer 1) or random layer-1 units (layer 2),
    so it is as boolean as the hand net and as wide, and means nothing."""
    rng = np.random.default_rng(seed)
    R = Builder()
    n = N_FEATURES_OF[LAYOUT]
    for u in B.l1.values():
        picks = rng.choice(n, size=len(u.terms), replace=False)
        R.one(u.name, "random", "null", {int(i): w for i, w in zip(picks, u.terms.values())}, u.bias, kind=u.kind)
    l1 = list(R.l1)
    for u in B.l2.values():
        picks = rng.choice(len(l1), size=len(u.terms), replace=False)
        R.l2[u.name] = Unit(u.name, "random", "null", {l1[i]: w for i, w in zip(picks, u.terms.values())}, u.bias)
    return R


def table(B: Builder) -> str:
    lines = ["| layer | unit | rule | from | weights |", "|---|---|---|---|---|"]
    def fname(i: int) -> str:
        for k in sorted(OFFSET, key=lambda k: -OFFSET[k]):
            if i >= OFFSET[k]:
                j = i - OFFSET[k]
                if k in ("hand", "gone"):
                    return f"{k}[{name(j)}]"
                if k in ("trick", "void"):
                    return f"{k}.{SEATS[j // N_TILES]}[{name(j % N_TILES)}]"
                if k == "count":
                    return f"{'ten' if j < N_TILES else 'five'}[{name(j % N_TILES)}]"
                if k == "winning":
                    return f"winning.{SEATS[j]}"
                return f"{k}[{j}]"
        return str(i)

    for u in B.l1.values():
        terms = " ".join(f"{w:+d}·{fname(i)}" for i, w in sorted(u.terms.items()))
        if len(terms) > 160:
            terms = terms[:160] + " …"
        lines.append(f"| 1 | `{u.name}` | {u.rule} | {u.why} | {terms} {u.bias:+d} |")
    for u in B.l2.values():
        terms = " ".join(f"{w:+d}·{k}" for k, w in u.terms.items())
        votes = ", ".join(f"{name(t)} {v:+d}" for t, v in sorted(u.out.items()))
        lines.append(f"| 2 | `{u.name}` | {u.rule} | {u.why} | {terms} {u.bias:+d} → {votes} |")
    lines.append(f"| out | bias | {B.b3_why} | the book: play low | "
                 + ", ".join(f"{name(t)} {int(B.b3[t]):+d}" for t in range(N_TILES)) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> None:
    out = Path(argv[0])
    out.mkdir(parents=True, exist_ok=True)
    B = build()
    net = B.compile()
    net.save(out / "hand.npz")
    (out / "hand.fnet").write_bytes(net.to_bytes())
    (out / "neurons.md").write_text(table(B))
    meta = {"h1": len(B.l1), "h2": len(B.l2), "layout": LAYOUT,
            "l1": [u.name for u in B.l1.values()], "l2": [u.name for u in B.l2.values()],
            "kinds": {u.name: u.kind for u in B.l1.values()}}
    (out / "units.json").write_text(json.dumps(meta, indent=1))
    D = with_decoys(build())
    D.compile().save(out / "hand_audit.npz")
    (out / "units_audit.json").write_text(json.dumps({"l1": list(D.l1), "l2": list(D.l2)}, indent=1))
    random_twin(build(), 0).compile().save(out / "random_twin.npz")
    print(f"{out}: hand-written field net, {len(B.l1)} × {len(B.l2)} units")


if __name__ == "__main__":
    main(sys.argv[1:])
