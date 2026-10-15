# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""The fallible tracker (#151): the book's play against opponents who lose
track of the voids.

Several of the book's lessons are written against opponents who sometimes
lose track ([[book-assumptions]] A3): the straight-off 84 that hopes a
defender misdiscards, the low-looking master late in a hand, inattention as
exploitable. Every arena opponent has been Zeb, whose worlds respect every
void shown: it never forgets. Here the field forgets.

The model (`Config.void_miss` = p, arena `V=<p>`, main 005f06ad): at each
decision, each hidden seat's whole void record (every suit it has shown out
of) is forgotten with probability p, by a coin keyed on that decision's
record, and the worlds are drawn from what is remembered; the record the plan
advances through stays true. It is a lapse at each decision, not a count lost
for the rest of the hand: a seat forgotten at one decision may be remembered
at the next (the sticky form is #168). Forgetting played tiles has no
consistent form in this planner and is not modeled.

PRE-REGISTRATION (committed with this file before any measured batch; the
timing pilot, `PILOT`, is excluded from every number).

*Players.* Team B, the field, is deployed Zeb 0.3.0 with `void_miss` = p,
p in {0, 0.1, 0.2, 0.4}; at p = 0 it is deployed Zeb byte for byte. The
issue's grid stopped at 0.2; a lapse at each decision forgets less than a
count lost for good, so the grid runs to 0.4. Both teams bid by the
heuristic rule at caution 0. Team A is **B** the head-to-head's book player,
block c (`BookPlay("zeb", "zeb", player())`, crisis-read in its stack since
5e747bf5), or **Z** plain deployed Zeb.

*Arms* (team A, then 100·p). Marks: B0, B10, Z10, B20, Z20, B40, Z40. Z0 is
deployed Zeb against itself and reads 0 exactly, so it is not run. The
book's game (250 points): PB0, PB40, PZ40.

*Estimator.* As `scenario.small_n`.

*Primary.* The book player's gap to plain Zeb against the same field,
G(p) = B_p - Z_p (G(0) = B0), and its change as the field forgets, ΔG(p) =
G(p) - G(0), in marks per game, at p = 0.1, 0.2, 0.4. **The book predicts ΔG
rises with p**: its play is worth more against opponents who lose track.
The read is ΔG(0.4); the crossover p* is the least p in the grid where G(p)'s
interval covers or lies above 0, reported as "beyond the grid" if none does.

*Registered secondaries.* Each arm's margin against its field, and the cost
of forgetting to the field itself, Z_p (the field's loss to deployed Zeb as
it forgets); the book's game, PB40 - PZ40 against PB0 on games won; ΔG(0.1)
and ΔG(0.2) as the trend.

*Size.* From the timing pilot (`PILOT` = seed 2890, 4 game pairs of B0,
B40, Z40 and PB40, excluded): 17-21 s a batch of 4 pairs at load 5-11, so
a batch of 12 runs about 60 s quiet and well inside the 300 s cap loaded.
Marks: `SEEDS` = 2901-2948 x 12 game pairs, **576 game pairs per arm**
(336 batches); the book's game: `POINT_SEEDS` = 3001-3016 x 12, **192 per
arm** (48 batches); 384 batches in all. The per-pair SD of a three-arm
contrast in #163 was 1.6-2.3 marks, so ΔG's MDE at 80% power is about
0.19-0.27 marks/game; the book's game reads to about 7 points of share.

*Batch-drop rule and fence.* As `scenario.small_n`; every batch through
`scripts/cpu-slot`, two at a time, seed-major. The fallible field is an
information-state planner over a looser fiber ([[strategy-fusion-fence]]).

    python -u -m scenario.fallible arena --arm B40 --seed S --games G --out DIR
    python -u -m scenario.fallible queue --root DIR [--workers 2] [--pilot]
    python -u -m scenario.fallible summarize --runs DIR... --out FILE.json
"""
from __future__ import annotations

from typing import Sequence

from scenario import book_batches
from scenario.small_n import FLOOR, NET, READING, _games, _meta, contrast, diff, pairs, player

PS = (0, 10, 20, 40)          # 100 * void_miss
MARK_ARMS = ("B0", "B10", "Z10", "B20", "Z20", "B40", "Z40")
POINT_ARMS = ("PB0", "PB40", "PZ40")
ARMS = MARK_ARMS + POINT_ARMS
PILOT = (2890,)               # the timing pilot, excluded
GAMES = 12                    # game pairs per batch
SEEDS = tuple(range(2901, 2949))
POINT_SEEDS = tuple(range(3001, 3017))


def field(p100: int):
    """Deployed Zeb 0.3.0 whose worlds forget each seat's voids with p = p100/100."""
    from arena.walt_play import WaltPlay
    from scenario.field import FieldNet

    net = FieldNet.from_bytes(NET.read_bytes())
    return WaltPlay(level=0, n_outer=160, floor=FLOOR, field=net, seed=0, draw="posterior",
                    reading=READING, tie="margin", void_miss=p100 / 100)


def seats(arm: str) -> tuple:
    """(bid_a, play_a, bid_b, play_b) for an arm."""
    from arena.bidders import HeuristicBidder
    from arena.book_play import BookPlay

    name = arm.removeprefix("P")
    play_a = BookPlay("zeb", "zeb", player()) if name[0] == "B" else player()
    return HeuristicBidder(), play_a, HeuristicBidder(), field(int(name[1:]))


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
    reg["G(0) = B0, marks/game"] = contrast(marks, [(1.0, "B0")])
    for p in PS[1:]:
        reg[f"G({p / 100:g}) = B{p} - Z{p}, marks/game"] = contrast(marks, diff(f"B{p}", f"Z{p}"))
    for p in PS[1:]:
        key = "primary" if p == 40 else "trend"
        reg[f"{key} dG({p / 100:g}) = G({p / 100:g}) - G(0), marks/game"] = contrast(
            marks, diff(f"B{p}", f"Z{p}") + [(-1.0, "B0")])
    for p in PS[1:]:
        reg[f"the field's cost of forgetting: Z{p}, marks/game"] = contrast(marks, [(1.0, f"Z{p}")])
    reg["250: (PB40 - PZ40) - PB0 on share of games won"] = contrast(won, diff("PB40", "PZ40") + [(-1.0, "PB0")])
    reg["250: PB40 - PZ40 on share of games won"] = contrast(won, diff("PB40", "PZ40"))
    return out


def jobs(pilot: bool = False) -> list[tuple[str, int]]:
    """Every registered batch, seed-major: (arm, base seed)."""
    if pilot:
        return [(a, s) for s in PILOT for a in ("B0", "B40", "Z40", "PB40")]
    marks = [(a, s) for s in SEEDS for a in MARK_ARMS]
    points = [(a, s) for s in POINT_SEEDS for a in POINT_ARMS]
    return marks + points


if __name__ == "__main__":
    book_batches.main(__doc__, "scenario.fallible", ARMS, seats, jobs, summarize, GAMES)
