# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""The arena's ledger for Silas (`silas/docs/cross-game.md` § 2.5, #130 slice 4).

A paired comparison of two players in one chair of a four-seat field, played
by the arena and written down as outcomes, never utilities: per deal, per
seating (the chair under test and the shaker), per arm (a, b and the chance
player), the hands as dealt, the auction, the contract, the points each side
took and the 28 plays. Silas reads it (`Ledger::from_json`,
`Paired::from_ledger`, `silas compare --ledger FILE`), replays every row under
its own rules, settles it by its own `Contract::settle` and values it in its
own units.

This is the fixed-state runner: every hand from one given score (E1b's
one-hand games at 0-0 are the case), never a game to the target, and every
seat its own player, the chair's replacing the field's. A hand passed out (no
forced bid) is recorded as passed out: the reshake is another deal. The deals
are `random.Random(seed)` shuffles of the 28 tiles, seven to each seat from
seat 0.

    python -u -m arena.ledger --a silas:risk-budget --b heuristic+lowest \\
        --field overbid:5 --n 200 --seed 1 --out ledger.json

A seat is `<bidder>+<play>` as `arena.cli` reads them (`lowest`: the lowest
legal tile), or `silas:<policy>` for one Silas policy that bids, declares and
plays (`SilasSeat`); `--field` is four seats separated by `;`, or
`overbid:<k>`. The chance
arm is the arena's `RandomBidder` and lowest-legal play. `overbidding_field`
is the named deviant field: `HeuristicBidder(caution=-k)` at every seat.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from forge.oracle.tables import DOMINOES
from forge.zeb.game import apply_action, current_player, is_terminal
from forge.zeb.types import BidState, GamePhase, ZebGameState

from arena.auction import PASS, BidPolicy, bidding_order, run_auction, score_hand, settle_points
from arena.bidders import HeuristicBidder, RandomBidder
from arena.play import LowestPlay, PlayPolicy

FORMAT = "silas-ledger-1"
# The arena's auction has no marks cap below the five that four calls reach.
MAX_MARKS = 5
DECLARATIONS = {**{p: f"P{p}" for p in range(7)}, 7: "DT", 9: "NT"}
# The arena's menu (SilasBidder, HeuristicBidder): a pip trump, the higher pip first.
OFFERED = ["P6", "P5", "P4", "P3", "P2", "P1", "P0"]


@dataclass(frozen=True)
class Seat:
    """One seat: how it bids, how it plays, and its name in the ledger."""

    bid: BidPolicy
    play: PlayPolicy
    name: str


def chance_seat() -> Seat:
    """The chance arm: the arena's random bidder and the lowest legal tile."""
    return Seat(RandomBidder(), LowestPlay(), "random+lowest")


def overbidding_field(k: int, play: PlayPolicy | None = None, play_name: str = "lowest") -> tuple[Seat, ...]:
    """The named overbidding field: `HeuristicBidder(caution=-k)` at every
    seat, bidding k points past its ceiling, playing `play` (lowest legal)."""
    bidder = HeuristicBidder(caution=-k)
    play = play or LowestPlay()
    return tuple(Seat(bidder, play, f"heuristic:3,{-k}+{play_name}") for _ in range(4))


def tile(t: int) -> str:
    high, low = DOMINOES[t]
    return f"{high}-{low}"


def deals(n: int, seed: int) -> list[tuple[tuple[int, ...], ...]]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        deck = list(range(28))
        rng.shuffle(deck)
        out.append(tuple(tuple(sorted(deck[7 * s:7 * s + 7])) for s in range(4)))
    return out


class _Hand:
    """One hand in play: its key, its seats, and the zeb state."""

    def __init__(self, deal: int, chair: int, shaker: int, arm: str, hands, seats: tuple[Seat, ...], auction):
        self.deal, self.chair, self.shaker, self.arm = deal, chair, shaker, arm
        self.hands, self.seats, self.auction = hands, seats, auction
        self.state = None if auction is None else ZebGameState(
            hands=hands, dealer=shaker, phase=GamePhase.PLAYING,
            bid_state=BidState(bids=auction.bids, high_bidder=auction.winner, high_bid=auction.high_bid),
            decl_id=auction.decl_id, bidder=auction.winner, played=frozenset(), play_history=(),
            current_trick=(), trick_leader=auction.winner, team_points=(0, 0),
        )

    def row(self) -> dict:
        order = bidding_order(self.shaker)
        bids = self.auction.bids if self.auction is not None else (PASS,) * 4
        row = dict(
            deal=self.deal, hands=[[tile(t) for t in h] for h in self.hands],
            shaker=self.shaker, chair=self.chair, arm=self.arm,
            calls=["pass" if bids[s] == PASS else str(bids[s]) for s in order],
            contract=None, outcome=None, plays=[],
        )
        if self.auction is not None:
            a, side = self.auction, self.auction.winner % 2
            row.update(
                contract=dict(declarer=a.winner, bid=a.high_bid, declaration=DECLARATIONS[a.decl_id]),
                outcome=dict(declaring=self.state.team_points[side], other=self.state.team_points[1 - side]),
                plays=[[int(s), tile(t)] for s, t in self.state.play_history],
            )
        return row


def run_ledger(
    *,
    field: tuple[Seat, Seat, Seat, Seat],
    a: Seat,
    b: Seat,
    n: int,
    seed: int,
    chance: Seat | None = None,
    seats: str | int = "rotate",
    scoring: str = "marks",
    target: int = 7,
    score: tuple[int, int] = (0, 0),
    forced: bool = False,
) -> dict:
    """The ledger of `a` against `b` in the chair (`seats`: "rotate" for
    every chair, or one seat) of `field`, at every shaker, on `n` deals from
    `seed`, every hand from `score` under `scoring` to `target`. The chance
    arm is played alongside. Play runs in lockstep: each tick, one batched
    call per play policy."""
    chance = chance or chance_seat()
    chairs = range(4) if seats == "rotate" else (int(seats),)
    points = scoring == "points"
    match = dict(
        marks=(0, 0) if points else score, marks_to_win=target,
        points=score if points else (0, 0), points_to_win=target if points else None,
    )
    live = []
    for d, hands in enumerate(deals(n, seed)):
        for chair in chairs:
            for shaker in range(4):
                for arm, player in (("a", a), ("b", b), ("chance", chance)):
                    sat = tuple(player if s == chair else field[s] for s in range(4))
                    # The same dice for every arm of a seating.
                    rng = random.Random(hash((seed, d, chair, shaker)))
                    auction = run_auction(hands, shaker, tuple(s.bid for s in sat), rng, force_shaker=forced, **match)
                    live.append(_Hand(d, chair, shaker, arm, hands, sat, auction))
    rows = [h.row() for h in live if h.state is None]
    playing = [h for h in live if h.state is not None]
    while playing:
        groups: dict[int, tuple[PlayPolicy, list[_Hand]]] = {}
        for h in playing:
            policy = h.seats[current_player(h.state)].play
            groups.setdefault(id(policy), (policy, []))[1].append(h)
        for policy, hs in groups.values():
            slots = policy.choose([h.state for h in hs], [h.auction.high_bid for h in hs],
                                  [match["marks"]] * len(hs), target)
            for h, slot in zip(hs, slots):
                h.state = apply_action(h.state, slot)
        rows.extend(h.row() for h in playing if is_terminal(h.state))
        playing = [h for h in playing if not is_terminal(h.state)]
    rows.sort(key=lambda r: (r["deal"], r["chair"], r["shaker"], ("a", "b", "chance").index(r["arm"])))
    return dict(
        ledger=FORMAT,
        field=", ".join(f"S{s} {seat.name}" for s, seat in enumerate(field)),
        players=dict(a=a.name, b=b.name, chance=chance.name),
        sample=dict(n=n, seed=seed),
        seats="rotate" if seats == "rotate" else int(seats),
        auction=dict(max_marks=MAX_MARKS, forced=forced),
        offered=OFFERED,
        scoring=dict(kind=scoring, target=target),
        score=list(score),
        rows=rows,
    )


def paired_mean(ledger: dict, unit: str | None = None) -> tuple[Fraction, tuple[int, int, int]]:
    """The arena's own reading of a ledger: the mean over deals of each deal's
    mean over its seatings of a's net transfer less b's, for the chair's side,
    settled by the arena's scoring (`score_hand` in marks, `settle_points` in
    points; a passed-out hand moves nothing), exactly; and the flips (deals a
    did better, b did, tied)."""
    unit = unit or ledger["scoring"]["kind"]
    per: dict[int, dict[str, Fraction]] = {}
    seatings: dict[int, int] = {}
    for r in ledger["rows"]:
        if r["arm"] == "chance":
            continue
        net = 0
        if r["contract"] is not None:
            side = r["contract"]["declarer"] % 2
            taken = [0, 0]
            taken[side], taken[1 - side] = r["outcome"]["declaring"], r["outcome"]["other"]
            bid = r["contract"]["bid"]
            gained = (score_hand(bid, side, tuple(taken)).marks if unit == "marks"
                      else settle_points(bid, side, tuple(taken)))
            team = r["chair"] % 2
            net = gained[team] - gained[1 - team]
        cell = per.setdefault(r["deal"], {"a": Fraction(0), "b": Fraction(0)})
        cell[r["arm"]] += net
        if r["arm"] == "a":
            seatings[r["deal"]] = seatings.get(r["deal"], 0) + 1
    diffs = [(per[d]["a"] - per[d]["b"]) / seatings[d] for d in sorted(per)]
    flips = (sum(x > 0 for x in diffs), sum(x < 0 for x in diffs), sum(x == 0 for x in diffs))
    return sum(diffs, Fraction(0)) / len(diffs), flips


def parse_seat(spec: str, seed: int = 0) -> Seat:
    """`silas:<policy>` (one Silas policy, a whole seat), or `<bidder>+<play>`
    as `arena.cli` reads them, with `lowest` the lowest legal tile."""
    if "+" not in spec and spec.startswith("silas:"):
        from arena.book_play import SilasSeat
        seat = SilasSeat(spec.removeprefix("silas:"))
        return Seat(seat, seat, spec)
    bid, _, play = spec.partition("+")
    from arena.cli import parse_bidder, parse_play
    bidder = parse_bidder(bid, device="cpu", gus_adapter=None)
    if play == "lowest":
        return Seat(bidder, LowestPlay(), spec)
    player = parse_play(play, model=None, n_samples=0, device="cpu", seed=seed)
    return Seat(bidder, player, f"{bid}+{play_name(play, seed)}")


def play_name(spec: str, seed: int) -> str:
    """A play policy's name in the field's identity: its spec with each field net
    (`F=<path>`) named by its content (`#` and the first 16 hex digits of its
    SHA-256), and the seed it plays at, so two ledgers name one field alike
    only when they seated the same nets at the same seed."""
    def named(token: str) -> str:
        if not token.startswith("F="):
            return token
        digest = hashlib.sha256(Path(token[2:]).read_bytes()).hexdigest()[:16]
        return f"{token}#{digest}"
    head, sep, options = spec.partition(":")
    spec = head + sep + ",".join(named(t) for t in options.split(",")) if sep else spec
    return f"{spec} seed={seed}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--field", required=True, help="four seats separated by ';', seat 0 first; or overbid:<k>")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--seats", default="rotate", help="rotate, or the chair 0-3")
    ap.add_argument("--scoring", choices=("marks", "points"), default="marks")
    ap.add_argument("--target", type=int, default=None, help="7 in marks, 250 in points by default")
    ap.add_argument("--score", default="0,0")
    ap.add_argument("--forced", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.field.startswith("overbid:"):
        field = overbidding_field(int(args.field.removeprefix("overbid:")))
    else:
        field = tuple(parse_seat(s.strip(), args.seed) for s in args.field.split(";") if s.strip())
    if len(field) != 4:
        raise SystemExit("--field names four seats")
    target = args.target or (7 if args.scoring == "marks" else 250)
    score = tuple(int(x) for x in args.score.split(","))
    ledger = run_ledger(field=field, a=parse_seat(args.a, args.seed), b=parse_seat(args.b, args.seed),
                        n=args.n, seed=args.seed, seats=args.seats, scoring=args.scoring, target=target,
                        score=score, forced=args.forced)
    Path(args.out).write_text(json.dumps(ledger) + "\n")
    mean, flips = paired_mean(ledger)
    print(f"{args.out}: {len(ledger['rows'])} hands; arena paired mean {mean} = {float(mean):+.4f} "
          f"{args.scoring} a deal; flips {flips}", flush=True)


if __name__ == "__main__":
    main()
