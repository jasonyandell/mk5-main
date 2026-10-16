# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""The auction read against a seed-varied field (#190): #117 again, with the
planner seed in the sampling.

#117 found the auction read by a known method worth K = +0.17 [+0.06, +0.29]
marks/game against plain deployed Zeb, the book's sign. But its A/A, plain
Zeb at planner seed 1 against the seed-0 field, read +0.12 [+0.02, +0.23],
so the read's margin over that scatter was +0.05 [-0.09, +0.19]
([[auction-read-2026-10-16]]). The field was one planner seed, and the
bootstrap over deals does not see the variance between seeds. Here every
Zeb seat is `arena.seed_varied.SeedVaried`: each hand is played at a
planner seed its deal picks from a pool of 16, the same in every arm and
both halves of a game pair. The seed scatter is now part of the sampling,
and the A/A draws its own seed per deal.

PRE-REGISTRATION (committed with this file before any measured batch; the
timing pilot, `PILOT`, is excluded from every number).

*Players.* Every seat bids by the arena's heuristic rule at caution 0 and
plays deployed Zeb 0.3.0's plan (`scenario.small_n.player`) seed-varied:
`SeedVaried(player(), pool=16, offset)`. Team B, the field, has offset 0.
Team A is:
- **K**, the read by a known method: `RegularsPlay(SeedVaried(player()),
  memory=False, grid=(0,))`, offset 0, so on every deal it plays the
  field's seed and differs from it only by the read (#117's K);
- **U**, the same with the caution flat on {-10, -5, 0, 5, 10} (#117's U);
- **Za**, the A/A: `SeedVaried(player(), offset=16)`, plain Zeb at an
  independent seed per deal. **Its interval should cover 0.**

*Arms.* Marks: K, U, Za. The book's game (250 points): PK, PZa.

*Estimator.* As `scenario.small_n`.

*Primary.* K in marks per game, read against the A/A. **The book predicts
K > 0.** The bar: K's interval excludes 0 above *and* Za's covers 0 = the
read pays in our arena; Za's interval excluding 0 = the A/A failed again,
and K is reported as in #117, not claimed; else a null at this power, with
its MDE.

*Registered secondaries.* U; K - U; K - Za; the book's game, PK - PZa and
PK - 1/2 on the share of games won. Each also reported beside #117's
estimate for the same contrast.

*Size.* #117's game pairs in smaller batches. The timing pilot (`PILOT` =
seed 3490, 4 game pairs of K, U, Za and PK, excluded) took 56-79 s a
batch, three times #117's, because a batch's decisions split across 16
seeds and batch less. So 6 game pairs a batch (about 85-120 s, inside the
300 s cap under load): `SEEDS` = 3501-3596 x 6, **576 per marks arm** (288
batches); `POINT_SEEDS` = 3601-3632 x 6, **192 per points arm** (64
batches); 352 batches. At #117's SDs (1.35-1.72 per pair) the MDE on K is
about 0.16 marks/game.

*Batch-drop rule and fence.* As `scenario.small_n`; every batch through
`scripts/cpu-slot`, seed-major. The read is a likelihood over this hand's
public auction, and the seed is keyed on the deal, which every seat holds
the same value of in every arm; no seat sees another's hand through it
([[strategy-fusion-fence]]). Every player thinks on a fixed world budget.

    python -u -m scenario.seed_varied arena --arm K --seed S --games G --out DIR
    python -u -m scenario.seed_varied queue --root DIR [--workers 2] [--pilot]
    python -u -m scenario.seed_varied summarize --runs DIR... --out FILE.json
"""
from __future__ import annotations

from scenario import auction_read, book_batches
from scenario.small_n import player

MARK_ARMS = auction_read.MARK_ARMS
POINT_ARMS = auction_read.POINT_ARMS
ARMS = MARK_ARMS + POINT_ARMS
PILOT = (3490,)               # the timing pilot, excluded
GAMES = 6                     # game pairs per batch (16 seeds batch less; see Size)
SEEDS = tuple(range(3501, 3597))
POINT_SEEDS = tuple(range(3601, 3633))
POOL = 16                     # planner seeds per Zeb seat


def zeb(offset: int = 0):
    """Deployed Zeb 0.3.0 at a planner seed per deal, from `POOL` seeds at `offset`."""
    from arena.seed_varied import SeedVaried

    return SeedVaried(player(), pool=POOL, offset=offset)


def seats(arm: str) -> tuple:
    """(bid_a, play_a, bid_b, play_b) for an arm."""
    from arena.bidders import HeuristicBidder
    from arena.regulars import RegularsPlay

    name = arm.removeprefix("P")
    if name == "K":
        play_a = RegularsPlay(zeb(), memory=False, grid=auction_read.KNOWN)
    elif name == "U":
        play_a = RegularsPlay(zeb(), memory=False)
    elif name == "Za":
        play_a = zeb(offset=POOL)
    else:
        raise ValueError(f"no arm {arm}")
    return HeuristicBidder(), play_a, HeuristicBidder(), zeb()


def jobs(pilot: bool = False) -> list[tuple[str, int]]:
    """Every registered batch, seed-major: (arm, base seed)."""
    if pilot:
        return [(a, s) for s in PILOT for a in ("K", "U", "Za", "PK")]
    marks = [(a, s) for s in SEEDS for a in MARK_ARMS]
    points = [(a, s) for s in POINT_SEEDS for a in POINT_ARMS]
    return marks + points


if __name__ == "__main__":
    book_batches.main(__doc__, "scenario.seed_varied", ARMS, seats, jobs, auction_read.summarize, GAMES)
