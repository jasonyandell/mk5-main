# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""The fair Walt's calibration panel (#181): a fixed list of the requests a Walt seat is asked.

    python -u -m tournament.fair_walt_panel build panel.json

PRE-REGISTERED RULE (written before any timing). The panel is what four recording seats
(the arena's heuristic bidder, uniform-legal play) are asked in seeded duplicate-style games,
dealt from base seeds 181000, 181001, ...: every bid request and the bidder's declare are
Walt's auction survey; every play request is a Walt decision. From the log, in order of
appearance and with no other choice made:

  - BID    the first 24 bid requests whose hand is new and whose seat is not forced to pass;
  - OPEN   the first 8 plays where the seat is the bidder and no tile has been played
           (production plays them at 160 worlds, 20 s);
  - MID    the first 16 plays that are not an opening lead, taking every 7th play of the log
           so they spread over the hand (production plays them at 40 worlds with the partner
           review, 14 s and up to 64 samples in 500 ms).

`python -m tournament.fair_walt_panel build` rewrites the same file byte for byte.
"""
from __future__ import annotations

import json
import random
import sys

from arena.auction import BidContext
from forge.zeb.game import current_player, legal_actions

from .seats import HostSeat

BASE_SEEDS = range(181000, 181040)
N_BID, N_OPEN, N_MID, MID_EVERY = 24, 8, 16, 7


class RecordingSeat(HostSeat):
    """Asks nothing of anyone: logs the host protocol's requests and answers them itself."""

    def __init__(self, log: list, game: str, bidder, rng: random.Random):
        super().__init__(None, game)
        self.log, self.bidder, self.rng = log, bidder, rng

    def bid(self, ctx: BidContext, rng: random.Random) -> int:
        self._seat_of[tuple(ctx.hand)] = ctx.seat
        self.log.append({"op": "bid", "hand": list(ctx.hand), "seat": ctx.seat, "high_bid": ctx.high_bid,
                         "high_seat": ctx.high_seat, "legal": list(ctx.legal), "game": self.game,
                         "hand_no": self.hand_no})
        return self.bidder.bid(ctx, rng)

    def declare(self, hand, bid, rng):
        seat = self._seat_of[tuple(hand)]
        self.log.append({"op": "declare", "hand": list(hand), "seat": seat, "bid": bid,
                         "game": self.game, "hand_no": self.hand_no})
        return self.bidder.declare(hand, bid, rng)

    def choose(self, states, bid_values, marks=None, marks_to_win=7):
        slots = []
        for s, bid in zip(states, bid_values):
            seat = current_player(s)
            self.log.append({"op": "play", "hand": list(s.hands[seat]), "seat": seat, "decl": s.decl_id,
                             "bid": int(bid), "bidder": s.bidder, "plays": [list(p) for p in s.play_history],
                             "leader": s.trick_leader, "points": list(s.team_points),
                             "game": self.game, "hand_no": self.hand_no})
            slots.append(self.rng.choice(legal_actions(s)))
        return slots

    def close(self) -> None:
        pass


def record(seed: int) -> list[dict]:
    from arena.cli import parse_bidder
    from arena.engine import ArenaConfig, _LiveGame, _run_lockstep, _seat_policies

    log: list[dict] = []
    cfg = ArenaConfig(base_seed=seed, marks_to_win=7)
    mk = lambda i: RecordingSeat(log, f"panel/{seed}/{i}", parse_bidder("heuristic", device="cpu", gus_adapter=None),
                                 random.Random(seed * 2 + i))
    sa, sb = mk(0), mk(1)

    class Game(_LiveGame):
        def _deal_and_bid(self):
            for s in (sa, sb):
                s.hand_no = self.hand_idx + 1
            super()._deal_and_bid()

    _run_lockstep([Game(0, 0, cfg, _seat_policies(0, sa, sb))], sa, sb, cfg.marks_to_win)
    return log


def build() -> dict:
    bids, opens, plays = [], [], []
    seen = set()
    for seed in BASE_SEEDS:
        for r in record(seed):
            if r["op"] == "bid":
                forced = r["high_seat"] >= 0 and r["high_seat"] % 2 == r["seat"] % 2
                k = (r["game"], r["hand_no"], r["seat"])
                if not forced and k not in seen:
                    seen.add(k)
                    bids.append(r)
            elif r["op"] == "play":
                if r["seat"] == r["bidder"] and not r["plays"]:
                    opens.append(r)
                else:
                    plays.append(r)
        if len(bids) >= N_BID and len(opens) >= N_OPEN and len(plays) >= N_MID * MID_EVERY:
            break
    return {"schema": "texas42-fair-walt-panel/1",
            "bid": bids[:N_BID], "open": opens[:N_OPEN], "mid": plays[::MID_EVERY][:N_MID]}


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] != "build":
        print(__doc__)
        return 2
    with open(argv[2], "w") as f:
        json.dump(build(), f, sort_keys=True, indent=0)
        f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
