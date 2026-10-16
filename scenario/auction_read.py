# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""The auction read (#117): what a bid, a pass, or a bid that lost tells the
table, by itself.

The book reads the auction (S-0021-a, S-0027-b, S-0077-c; p.54: a bidder who
lost the bid "probably has a double or two"), and every arena seat has
ignored it: the hidden hands after the auction are a uniform draw over the
void-consistent deals ([[book-assumptions]] A4). #149 built the reader
(`arena.regulars.RegularsPlay`): each opposing seat's turn in the auction,
a bid, a pass or a bid since overcalled, is evidence on its dealt hand under
the heuristic bidder, joined to Zeb's soft play-reading as a likelihood
(`scenario.planner.Situation.auction`). There it read the auction with a
flat belief over the bidder's caution, and the read alone looked worth
0.05-0.19 marks/game, unregistered, inside seed noise
([[regulars-table-2026-10-14]]). This is the registered test of the read
by itself, and of the read by a *known* method, which is what the book's
A4 assumes: the table knows how its opponents bid.

PRE-REGISTRATION (committed with this file before any measured batch; the
timing pilot, `PILOT`, is excluded from every number).

*Players.* Every seat plays deployed Zeb 0.3.0's plan (`scenario.small_n.
player`) and bids by the arena's heuristic rule at caution 0. Team B is
plain deployed Zeb. Team A is:
- **K**, the read by a known method: Zeb reading each opposing seat's
  auction turn under the heuristic bidder at caution 0, the field's true
  method, trembling with probability 0.1 so no world vanishes
  (`RegularsPlay(player(), memory=False, grid=(0,))`);
- **U**, the read by an unknown method: the same with the caution flat on
  {-10, -5, 0, 5, 10}, #149's R0 reader;
- **Za**, the A/A: plain Zeb at planner seed 1, the seed scatter of a
  player that reads nothing more.
K and U differ from team B only in the read (same planner seed 0), so where
the auction carries no evidence their play is team B's tile for tile.

*Arms.* Marks: K, U, Za. The book's game (250 points): PK, PZa.

*Estimator.* As `scenario.small_n` (paired halves, game pairs, percentile
bootstrap over game pairs, 10,000 resamples, seed 42).

*Primary.* K, team A's margin against the field in marks per game: the
value of reading the auction by a known method. **The book predicts K > 0.**
The bar: K's interval excludes 0 above = the read pays in our arena;
excludes 0 below = it costs; else a null at this power, with its MDE.

*Registered secondaries.* U (the read with the method unknown); K - U
(the value of knowing the method); Za (the A/A; its interval should cover
0); K - Za (the read against the seed scatter); the book's game, PK - PZa
and PK - 1/2 on the share of games won.

*Size.* From the timing pilot (`PILOT` = seed 3290, 4 game pairs of K, U,
Za and PK, excluded): 20-23 s a batch, so a batch of 12 runs about 60 s
quiet. The per-pair SD of a plain-Zeb arm against a near-
copy in #172 was 0.9-1.3 marks, so 576 game pairs give a K MDE of about
2.8·1.2/√576 = 0.14 marks/game at 80% power. Marks: `SEEDS` = 3301-3348 x
12 game pairs, **576 per arm** (144 batches); the book's game:
`POINT_SEEDS` = 3401-3416 x 12, **192 per arm** (32 batches); 176 batches
in all.

*Batch-drop rule and fence.* As `scenario.small_n`; every batch through
`scripts/cpu-slot`, seed-major. The read is a likelihood over information
public when it is used, this hand's auction ([[strategy-fusion-fence]]).
Every player thinks on a fixed world budget, never a clock.

    python -u -m scenario.auction_read arena --arm K --seed S --games G --out DIR
    python -u -m scenario.auction_read queue --root DIR [--workers 2] [--pilot]
    python -u -m scenario.auction_read summarize --runs DIR... --out FILE.json
"""
from __future__ import annotations

from typing import Sequence

from scenario import book_batches
from scenario.small_n import _games, _meta, contrast, diff, pairs, player

MARK_ARMS = ("K", "U", "Za")
POINT_ARMS = ("PK", "PZa")
ARMS = MARK_ARMS + POINT_ARMS
PILOT = (3290,)               # the timing pilot, excluded
GAMES = 12                    # game pairs per batch
SEEDS = tuple(range(3301, 3349))
POINT_SEEDS = tuple(range(3401, 3417))
KNOWN = (0,)                  # the field's caution: the method the table knows


def seats(arm: str) -> tuple:
    """(bid_a, play_a, bid_b, play_b) for an arm."""
    from arena.bidders import HeuristicBidder
    from arena.regulars import RegularsPlay

    name = arm.removeprefix("P")
    if name == "K":
        play_a = RegularsPlay(player(), memory=False, grid=KNOWN)
    elif name == "U":
        play_a = RegularsPlay(player(), memory=False)
    elif name == "Za":
        play_a = player(seed=1)
    else:
        raise ValueError(f"no arm {arm}")
    return HeuristicBidder(), play_a, HeuristicBidder(), player()


def summarize(runs: Sequence[str]) -> dict:
    by: dict[str, list[str]] = {}
    for run in runs:
        by.setdefault(_meta(run)["arm"], []).append(run)
    marks = {a: pairs(rs, "marks") for a, rs in by.items() if not a.startswith("P")}
    won = {a: pairs(rs, "won") for a, rs in by.items() if a.startswith("P")}
    out: dict = {"arms": {}, "registered": {}}
    for arm, rs in sorted(by.items()):
        out["arms"][arm] = {
            "batches": [{x: _meta(r)[x] for x in ("seed", "wall_s", "load_avg")} for r in rs],
            "games": sum(len(_games(r)) for r in rs),
            "a_margin": contrast(marks if arm in marks else won, [(1.0, arm)]),
        }
    reg = out["registered"]
    reg["primary K, the read by a known method, marks/game"] = contrast(marks, [(1.0, "K")])
    reg["U, the read by an unknown method, marks/game"] = contrast(marks, [(1.0, "U")])
    reg["K - U, the value of knowing the method, marks/game"] = contrast(marks, diff("K", "U"))
    reg["Za, the A/A, marks/game"] = contrast(marks, [(1.0, "Za")])
    reg["K - Za, marks/game"] = contrast(marks, diff("K", "Za"))
    reg["250: PK - PZa on share of games won"] = contrast(won, diff("PK", "PZa"))
    pk = contrast(won, [(1.0, "PK")])    # a share, whose chance baseline is a half
    reg["250: PK - 1/2 on share of games won"] = {**pk, **{k: pk[k] - 0.5 for k in ("mean", "lo", "hi")}}
    return out


def jobs(pilot: bool = False) -> list[tuple[str, int]]:
    """Every registered batch, seed-major: (arm, base seed)."""
    if pilot:
        return [(a, s) for s in PILOT for a in ("K", "U", "Za", "PK")]
    marks = [(a, s) for s in SEEDS for a in MARK_ARMS]
    points = [(a, s) for s in POINT_SEEDS for a in POINT_ARMS]
    return marks + points


if __name__ == "__main__":
    book_batches.main(__doc__, "scenario.auction_read", ARMS, seats, jobs, summarize, GAMES)
