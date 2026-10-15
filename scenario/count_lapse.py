# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""The field that loses the count (#172): the book's play against opponents
who forget which tiles have gone.

The book's fallible opponent loses the count, and its ploys lean on that: the
straight-off 84 that hopes for a misdiscard, the low-looking master late in a
hand ([[book-assumptions]] A3). #151 seated a field that forgets voids, and
the book player and Zeb collected its cost equally ([[fallible-field-2026-10-15]]).
A lost count is not a fiber of the planner, whose worlds share one true record
(#172); it is a seat's view. Here the field decides at a view of the game in
which some tiles it saw played are remembered as still out
(`arena/count_lapse.py`): a played tile t and a live tile u exchanged
everywhere, which keeps the view a legal game of the same trick leaders, while
the game and its record stay true.

*The form.* A tile another seat played in a completed trick is forgotten with
probability p by a coin keyed on the deal, the seat and the tile, so once lost
it stays lost for the hand (the sticky form; #151's lapse re-remembered at
every decision). Its stand-in u is the first tile in a keyed order that keeps
the view a legal history with the same leaders; where none does, t is
remembered. In random play about 60% of forgotten tiles find one: 70% of tiles
that lost their trick, 30-45% of those that won it, so a played master is lost
at about 0.4p.

PRE-REGISTRATION (committed with this file before any measured batch; the
timing pilot, `PILOT`, is excluded from every number).

*Players.* Team B, the field, is deployed Zeb 0.3.0 (`scenario.small_n.player`)
deciding at the view of a seat that loses the count with p in {0, 0.05, 0.1,
0.2}; at p = 0 it is deployed Zeb byte for byte. A sticky coin forgets more as
the hand goes on: by the sixth trick a seat has seen about 15 tiles played by
others, so the grid forgets about 0.75, 1.5 and 3 of them. Both teams bid by
the heuristic rule at caution 0. Team A is **B** the book player, block c
(`BookPlay("zeb", "zeb", player())`), or **Z** plain deployed Zeb.

*Arms* (team A, then 100·p). Marks: B0, B5, Z5, B10, Z10, B20, Z20. Z0 is
deployed Zeb against itself and reads 0 exactly, so it is not run. The book's
game (250 points): PB0, PB20, PZ20.

*Estimator.* As `scenario.small_n`.

*Primary.* G(p) = B_p - Z_p, the book player's gap to plain Zeb against the
same field, and ΔG(p) = G(p) - G(0), in marks per game. **The book predicts ΔG
rises with p**: its play is worth more against opponents who lose the count.
The read is ΔG(0.2); the crossover p* is the least p in the grid where G(p)'s
interval covers or lies above 0, reported as "beyond the grid" if none does.

*Registered secondaries.* The field's own cost of losing the count, Z_p;
ΔG(0.05) and ΔG(0.1) as the trend; the book's game, (PB20 - PZ20) - (PB0 -
1/2) and PB20 - PZ20 on games won; mechanism: the field's tiles forgotten and
stood in for per decision, from each batch's `b_lapse`.

*Size.* As #151, whose timing this wrapper adds well under a millisecond a
decision to: `SEEDS` = 3101-3148 x 12 game pairs, **576 game pairs per marks
arm** (336 batches); `POINT_SEEDS` = 3201-3216 x 12, **192 per points arm** (48
batches); 384 batches. MDE on ΔG about 0.16 marks/game at #151's SD. The timing
pilot (`PILOT` = seed 3090, 4 game pairs of B0, B20, Z20 and PB20, excluded):
19-21 s a batch of 4, as #151; at p = 0.2 the field had forgotten 1.7 tiles a
decision on average, 1.0 of them with a stand-in.

*Batch-drop rule and fence.* As `scenario.small_n`; every batch through
`scripts/cpu-slot`, two at a time, seed-major. The forgetting seat's planner
runs on one view shared by all its worlds ([[strategy-fusion-fence]]).

    python -u -m scenario.count_lapse arena --arm B20 --seed S --games G --out DIR
    python -u -m scenario.count_lapse queue --root DIR [--workers 2] [--pilot]
    python -u -m scenario.count_lapse summarize --runs DIR... --out FILE.json
"""
from __future__ import annotations

from typing import Sequence

from scenario import book_batches
from scenario.small_n import _games, _meta, contrast, diff, pairs, player

PS = (0, 5, 10, 20)           # 100 * p
MARK_ARMS = ("B0", "B5", "Z5", "B10", "Z10", "B20", "Z20")
POINT_ARMS = ("PB0", "PB20", "PZ20")
ARMS = MARK_ARMS + POINT_ARMS
PILOT = (3090,)               # the timing pilot, excluded
GAMES = 12                    # game pairs per batch
SEEDS = tuple(range(3101, 3149))
POINT_SEEDS = tuple(range(3201, 3217))


def field(p100: int):
    """Deployed Zeb 0.3.0 deciding at the view of a seat that loses the count with p = p100/100."""
    from arena.count_lapse import CountLapse

    return CountLapse(player(), p100 / 100)


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
        lapse = [_meta(r)["b_lapse"] for r in rs]
        decisions = sum(x["decisions"] for x in lapse)
        out["arms"][arm] = {
            "batches": [{x: _meta(r)[x] for x in ("seed", "wall_s", "load_avg")} for r in rs],
            "games": sum(len(_games(r)) for r in rs),
            "a_margin": contrast(marks if arm in marks else won, [(1.0, arm)]),
            "field_forgotten_per_decision": sum(x["forgotten"] for x in lapse) / max(decisions, 1),
            "field_stood_per_decision": sum(x["stood"] for x in lapse) / max(decisions, 1),
        }
    reg = out["registered"]
    reg["G(0) = B0, marks/game"] = contrast(marks, [(1.0, "B0")])
    for p in PS[1:]:
        reg[f"G({p / 100:g}) = B{p} - Z{p}, marks/game"] = contrast(marks, diff(f"B{p}", f"Z{p}"))
    for p in PS[1:]:
        key = "primary" if p == 20 else "trend"
        reg[f"{key} dG({p / 100:g}) = G({p / 100:g}) - G(0), marks/game"] = contrast(
            marks, diff(f"B{p}", f"Z{p}") + [(-1.0, "B0")])
    for p in PS[1:]:
        reg[f"the field's cost of losing the count: Z{p}, marks/game"] = contrast(marks, [(1.0, f"Z{p}")])
    gap = contrast(won, diff("PB20", "PZ20") + [(-1.0, "PB0")])    # PB0's chance baseline is a half
    reg["250: (PB20 - PZ20) - (PB0 - 1/2) on share of games won"] = {
        **gap, **{k: gap[k] + 0.5 for k in ("mean", "lo", "hi")}}
    reg["250: PB20 - PZ20 on share of games won"] = contrast(won, diff("PB20", "PZ20"))
    return out


def jobs(pilot: bool = False) -> list[tuple[str, int]]:
    """Every registered batch, seed-major: (arm, base seed)."""
    if pilot:
        return [(a, s) for s in PILOT for a in ("B0", "B20", "Z20", "PB20")]
    marks = [(a, s) for s in SEEDS for a in MARK_ARMS]
    points = [(a, s) for s in POINT_SEEDS for a in POINT_ARMS]
    return marks + points


if __name__ == "__main__":
    book_batches.main(__doc__, "scenario.count_lapse", ARMS, seats, jobs, summarize, GAMES)
