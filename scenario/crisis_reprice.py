# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""The crisis as read, priced in the book player's stack (#162's next step).

`silas/examples/book/crisis-read.silas` is the book's crisis as Hand 31 reads
it: its two-trump trigger asks that one setter certainly holds both trumps
out, read from the setters' plays under `protect-the-count`, where
`crisis.silas` fires on any two trumps out the bidder cannot draw
([[silas]]; `silas/GAPS.md` 1). It is the first book rule that acts on a
reading of another seat's plays. This study puts it in the book player's
stack in place of `crisis` and prices the change, before the stack adopts it.

PRE-REGISTRATION (committed with this file before any measured batch; the
timing pilot, `PILOT`, is excluded from every number).

*Players.* Team B is deployed Zeb 0.3.0 (`scenario.small_n.player()`); both
teams bid by the heuristic rule at caution 0. Team A is the head-to-head's
book player, block c (`BookPlay("zeb", "zeb", player())`: Zeb's best tile
inside the deciding rule's set, Zeb's tile where no rule fires;
[[book-head-to-head-2026-10-13]]):
- **C**: the stack as it stands (`crisis`);
- **R**: the same stack with `crisis` compiled from `crisis-read`
  (`swap={"crisis": "crisis-read"}`).
C and R differ only where `crisis` fires and `crisis-read` does not: two
trumps out the bidder cannot draw, and no setter's play that makes one
setter's holding both of them certain.

*Arms.* Marks: C, R. The book's game (250 points): PC, PR.

*Estimator.* As `scenario.small_n`: game pairs under the arena's paired
halves, per-pair contrasts, 95% percentile bootstrap (10,000 resamples, seed
42).

*Primary.* Q = R - C in marks per game. **The book's crisis is the read one;
the prediction is Q >= 0**, and the stack adopts `crisis-read` unless Q's
interval lies below 0. Because the two stacks differ only at a rare
decision, most game pairs are identical; Q is reported with the share of
game pairs whose outcome differs, and a null is read with its MDE.

*Registered secondaries.* PR - PC on games won and settled points in the
book's game; each arm's margin against deployed Zeb; the share of game pairs
that differ between C and R, by game.

*Size.* From the timing pilot (`PILOT` = seed 2390, 4 game pairs each of C,
R and PR, excluded): batches took 23-26 s once in a slot at load average
29-50, so 12 pairs a batch stay well under the 300 s cap. One of the 4 game
pairs differed between C and R. The per-pair SD of Q was 0.75 on 4 pairs, too
few to trust; the sizing takes SD 1.0, so 192 pairs give a half-width of
1.96*1.0/sqrt(192) = 0.14 marks/game and an MDE of 2.8*1.0/sqrt(192) = 0.20
(80% power, two-sided 5%). So:
- marks: `SEEDS` = 2401-2416, 12 pairs a batch: **192 game pairs per arm**,
  two arms (32 batches);
- 250 points: `POINT_SEEDS` = 2451-2458: 96 game pairs per arm, two arms
  (16 batches).
The book player's policies run under a work cap of 10^12 per decision
(`arena.book_play.BUDGET`); a reading exceeds the binding's default 10^7,
and a cap changes no tile a policy allows within it.

*Batch-drop rule and fence.* As `scenario.small_n`; every batch through
`scripts/cpu-slot`, two at a time, seed-major.

    python -u -m scenario.crisis_reprice arena --arm R --seed S --games G --out DIR
    python -u -m scenario.crisis_reprice queue --root DIR [--workers 2] [--pilot]
    python -u -m scenario.crisis_reprice summarize --runs DIR... --out FILE.json
"""
from __future__ import annotations

from typing import Sequence

from scenario import book_batches
from scenario.small_n import _games, _meta, contrast, diff, pairs, player

MARK_ARMS = ("C", "R")
POINT_ARMS = ("PC", "PR")
ARMS = MARK_ARMS + POINT_ARMS
# The stack adopted crisis-read under the registered rule (see the page); its
# slot is now named crisis-read, so C swaps it back to crisis and R plays the
# stack as it stands: each arm plays exactly as registered.
BACK = {"crisis-read": "crisis"}
PILOT = (2390,)               # the timing pilot, excluded
GAMES = 12                    # game pairs per batch
SEEDS = tuple(range(2401, 2417))         # marks: 16 batches per arm, 192 game pairs
POINT_SEEDS = tuple(range(2451, 2459))   # 250 points: 8 batches per arm, 96 game pairs


def seats(arm: str) -> tuple:
    """(bid_a, play_a, bid_b, play_b) for an arm."""
    from arena.bidders import HeuristicBidder
    from arena.book_play import BookPlay

    swap = BACK if arm.removeprefix("P") == "C" else None
    return HeuristicBidder(), BookPlay("zeb", "zeb", player(), swap=swap), HeuristicBidder(), player()


def differing(runs_a: Sequence[str], runs_b: Sequence[str]) -> dict:
    """The share of game pairs whose two games differ in outcome (marks or
    points) between two arms on the same seeds."""
    def outcomes(runs):
        out = {}
        for run in runs:
            seed = _meta(run)["seed"]
            for g in _games(run):
                out.setdefault((seed, g["game"]), []).append((g["a_team"], g["a_marks"], g["b_marks"],
                                                              g["a_points"], g["b_points"]))
        return {k: sorted(v) for k, v in out.items()}
    a, b = outcomes(runs_a), outcomes(runs_b)
    common = sorted(set(a) & set(b))
    n = sum(a[k] != b[k] for k in common)
    return {"game_pairs": len(common), "differ": n, "share": n / len(common) if common else None}


def summarize(runs: Sequence[str]) -> dict:
    by: dict[str, list[str]] = {}
    for run in runs:
        by.setdefault(_meta(run)["arm"], []).append(run)
    marks = {a: pairs(rs, "marks") for a, rs in by.items() if not a.startswith("P")}
    won = {a: pairs(rs, "won") for a, rs in by.items() if a.startswith("P")}
    pts = {a: pairs(rs, "points") for a, rs in by.items() if a.startswith("P")}
    out: dict = {"arms": {}, "registered": {}}
    for arm, rs in sorted(by.items()):
        out["arms"][arm] = {
            "batches": [{x: _meta(r)[x] for x in ("seed", "wall_s", "load_avg")} for r in rs],
            "games": sum(len(_games(r)) for r in rs),
            "a_margin": contrast(marks if arm in marks else won, [(1.0, arm)]),
        }
    reg = out["registered"]
    reg["primary Q = R - C, marks/game"] = contrast(marks, diff("R", "C"))
    reg["250: PR - PC on share of games won"] = contrast(won, diff("PR", "PC"))
    reg["250: PR - PC on settled points/game"] = contrast(pts, diff("PR", "PC"))
    if "C" in by and "R" in by:
        reg["game pairs that differ, marks"] = differing(by["C"], by["R"])
    if "PC" in by and "PR" in by:
        reg["game pairs that differ, 250"] = differing(by["PC"], by["PR"])
    return out


def jobs(pilot: bool = False) -> list[tuple[str, int]]:
    """Every registered batch, seed-major: (arm, base seed)."""
    if pilot:
        return [(a, s) for s in PILOT for a in ("C", "R", "PR")]
    marks = [(a, s) for s in SEEDS for a in MARK_ARMS]
    points = [(a, s) for s in POINT_SEEDS for a in POINT_ARMS]
    return marks + points


if __name__ == "__main__":
    book_batches.main(__doc__, "scenario.crisis_reprice", ARMS, seats, jobs, summarize, GAMES)
