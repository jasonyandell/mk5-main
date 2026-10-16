# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""Seat a participant at the arena's table: one (BidPolicy, PlayPolicy) per game.

The referee is the arena (`arena/engine.py`): deal, auction, play, marks. A
JavaScript participant runs in `tournament/seat/host.mjs`, one child process per
team per game, asked one JSON line per decision; it is sent its own seat's
dealt tiles and the public record only, never another hand.
"""
from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path
from typing import Sequence

from arena.auction import BidContext
from forge.zeb.game import current_player

from arena import cost

from .artifacts import materialize
from .participants import Participant, deterministic_budget

HOST = Path(__file__).resolve().parent / "seat" / "host.mjs"
NATIVE_HOST = Path(__file__).resolve().parent / "seat" / "native-host.mjs"


class HostError(RuntimeError):
    pass


class NodeHost:
    """A seat process. Requests are answered in order, one line each."""

    def __init__(self, kind: str, artifacts: Path, config: dict, script: Path | None = None):
        self.cost = None  # set by close(): what the host process cost (#173)
        self.proc = subprocess.Popen(
            ["node", str(script or HOST), kind, str(artifacts), json.dumps(config)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def ask(self, request: dict) -> dict:
        self.proc.stdin.write(json.dumps(request) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise HostError(f"seat host exited (code {self.proc.poll()}) on {request.get('op')}")
        reply = json.loads(line)
        if "error" in reply:
            raise HostError(reply["error"])
        return reply

    def close(self) -> None:
        self.cost = close_and_cost(self.proc)


def close_and_cost(proc: subprocess.Popen) -> dict | None:
    """Close a seat process's stdin, reap it (killed after 10 s) and return what it cost:
    CPU seconds, its waited children included, and on macOS its own instructions."""
    if proc.returncode is not None:
        return None
    proc.stdin.close()
    spent = cost.wait(proc, 10)
    if spent is None:
        proc.kill()
        spent = cost.wait(proc)
    return spent


class HostSeat:
    """BidPolicy and PlayPolicy for one team in one game, backed by a NodeHost.

    `game` and `hand_no` name the hand for the participant's own seeding
    (Plunge seeds Walt by game id and hand number); the game driver sets
    `hand_no` before every deal.
    """

    def __init__(self, host: NodeHost, game: str):
        self.host = host
        self.game = game
        self.hand_no = 1
        self._seat_of = {}  # hand -> seat, from this auction's bid turns

    # BidPolicy
    def bid(self, ctx: BidContext, rng: random.Random) -> int:
        # run_auction's declare() is not told the seat; every seat bids exactly
        # once before it, so the seat is the one that held this hand.
        self._seat_of[tuple(ctx.hand)] = ctx.seat
        return int(self.host.ask({
            "op": "bid", "hand": list(ctx.hand), "seat": ctx.seat, "high_bid": ctx.high_bid,
            "high_seat": ctx.high_seat, "legal": list(ctx.legal), "game": self.game,
            "hand_no": self.hand_no})["bid"])

    def declare(self, hand: tuple[int, ...], bid: int, rng: random.Random) -> int:
        seat = self._seat_of[tuple(hand)]
        return int(self.host.ask({"op": "declare", "hand": list(hand), "seat": seat, "bid": bid,
                                  "game": self.game, "hand_no": self.hand_no})["decl"])

    # PlayPolicy
    def choose(self, states, bid_values: Sequence[int], marks=None, marks_to_win: int = 7) -> list[int]:
        slots = []
        for s, bid in zip(states, bid_values):
            seat = current_player(s)
            tile = int(self.host.ask({
                "op": "play", "hand": list(s.hands[seat]), "seat": seat, "decl": s.decl_id,
                "bid": int(bid), "bidder": s.bidder, "plays": [list(p) for p in s.play_history],
                "leader": s.trick_leader, "points": list(s.team_points),
                "game": self.game, "hand_no": self.hand_no})["tile"])
            slots.append(list(s.hands[seat]).index(tile))
        return slots

    def close(self) -> None:
        self.host.close()

    @property
    def cost(self) -> dict | None:
        return self.host.cost


class NativeZebSeat(HostSeat):
    """A configured Zeb: the release's JavaScript bidder over its native Rust player.

    The match-aware bidder reads the auction's context (dealer, bids, marks), so
    the bid request carries it; the host never sees another seat's hand.
    """

    def bid(self, ctx: BidContext, rng: random.Random) -> int:
        self._seat_of[tuple(ctx.hand)] = ctx.seat
        return int(self.host.ask({
            "op": "bid", "hand": list(ctx.hand), "seat": ctx.seat, "high_bid": ctx.high_bid,
            "high_seat": ctx.high_seat, "legal": list(ctx.legal), "game": self.game,
            "hand_no": self.hand_no, "dealer": ctx.dealer, "bids": list(ctx.bids),
            "marks": list(ctx.marks), "marks_to_win": ctx.marks_to_win,
            "points_to_win": ctx.points_to_win})["bid"])


class ArenaSeat:
    """An mk5 arena player (`<bidder>+<play>`), built in this process."""

    cost = None  # in-process: counted in the referee's own CPU

    def __init__(self, bidder, play):
        self.bidder, self.play = bidder, play
        self.hand_no = 1

    def bid(self, ctx, rng):
        return self.bidder.bid(ctx, rng)

    def declare(self, hand, bid, rng):
        return self.bidder.declare(hand, bid, rng)

    def choose(self, states, bid_values, marks=None, marks_to_win=7):
        return self.play.choose(states, bid_values, marks, marks_to_win)

    def close(self) -> None:
        pass


def seat(p: Participant, game: str, device: str | None = None, *, historical: bool = False):
    """A fresh seat for one team in one game.

    FAIR CLOCKS (#181): a participant whose config says `clock: real` thinks on wall time and is
    not seated, except as a `historical` research seat (a paired check against a withdrawn player).
    """
    if not historical and not deterministic_budget(p.to_record()):
        raise ValueError(f"{p.id} thinks on wall time; FAIR CLOCKS forbids seating it")
    if p.kind == "zeb-npm":
        return HostSeat(NodeHost("zeb", materialize(p), p.config), game)
    if p.kind == "zeb-native":
        from .zeb_native_seat import verify_binary
        art = materialize(p)
        return NativeZebSeat(NodeHost("zeb", art, {**p.config, "nativeBinary": str(verify_binary(art))},
                                      script=NATIVE_HOST), game)
    if p.kind == "plunge-walt":
        return HostSeat(NodeHost("walt", materialize(p), p.config), game)
    if p.kind == "demon":
        # STRATEGY-FUSION FENCE. The Demon: a fenced tournament opponent; measuring stick only.
        from .demon_fenced_seat import DemonSeat
        return DemonSeat(p, materialize(p), game)
    if p.kind == "arena":
        materialize(p)
        bidder_spec, _, play_spec = p.pin["spec"].partition("+")
        if play_spec.startswith("lens"):
            from .eq_fenced_seat import eq_seat  # STRATEGY-FUSION FENCE: see that module
            return eq_seat(p, bidder_spec, play_spec, device)
        from arena.cli import parse_bidder, parse_play
        dev = device or "cpu"
        return ArenaSeat(parse_bidder(bidder_spec, device=dev, gus_adapter=None),
                         parse_play(play_spec, model=None, n_samples=0, device=dev, seed=0))
    raise ValueError(f"unknown participant kind {p.kind!r}")
