# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""The book over a reader with no plan (#161): do Winning 42's rules raise a
player who plans over no worlds at all?

The book is a policy for a reader who cannot enumerate ([[book-assumptions]]
A7). #154 handed it to Zeb at 16, 64 and 160 worlds: as Zeb's near-tie break
it neither raised nor cost Zeb at any n, and as a filter its cost fell from
1.10 marks/game at 160 worlds to 0.21 [-0.02, 0.44] at 16
([[book-small-n-2026-10-14]]). Every #154 fallback still planned. This study
seats the reader #154 extrapolates to, one who plans over no worlds: where no
rule of the book fires he plays his lowest legal tile.

PRE-REGISTRATION (committed with this file before any measured batch; the
timing pilot, `PILOT`, is excluded from every number).

*Players.* Team B in every arm is deployed Zeb 0.3.0's plan
(`scenario.small_n.player()`), and both teams bid by the arena's heuristic
rule at caution 0. Team A:
- **L0**: the lowest legal tile, always (`LowestPlay`);
- **K0**: the book player over the same fallback (`BookPlay("lowest",
  "least")`, the head-to-head's variant (a)): the book's 18 policies in their
  stated order, the least tile of the deciding rule's set, the lowest legal
  tile where none fires. K0 and L0 differ only by the rules;
- **A0**: L0 against deployed Zeb at planner seed 1, the A/A (L0 is
  deterministic, so the A/A varies the opponent's sample);
- **Z16**, **F16**: #154's arms on these seeds: Zeb at 16 worlds alone, and
  inside the book's sets (`BookPlay("zeb", "zeb", player(16))`).

*Arms.* Marks (games to 7): L0, K0, A0, Z16, F16. The book's game (250
points): PL0, PK0.

*Pairing and the estimator.* As `scenario.small_n`: a game pair is (base
seed, game index) under the arena's paired halves, every contrast is taken
per game pair over the pairs its arms hold, and its 95% interval is the
percentile bootstrap over game pairs (10,000 resamples, seed 42).

*Primary.* D0 = K0 - L0 in marks per game: what the book's rules are worth to
a reader who plans over no worlds. **The book predicts D0 > 0.** Read against
the A/A (A0 - L0, whose interval should cover 0) and the bar: D0's interval
excludes 0 above = the rules raise the reader they were written for;
excludes 0 below = they cost even him; else a null with its minimum
detectable effect.

*Registered secondaries* (marks per game unless stated):
- the trend from 16 worlds to none: D0 - E16, with E16 = F16 - Z16 (the
  rules as a filter over a 16-world plan); the book predicts D0 - E16 > 0;
- each arm's margin against deployed Zeb, and the cost of planning nothing:
  L0 - Z16 and K0 - Z16 (how much of a 16-world plan the rules recover:
  D0 / (Z16 - L0), reported with the two intervals, not a ratio interval);
- the book's game: PK0 - PL0 on team A's share of games won and on settled
  points per game;
- mechanism, by arm: contracts per hand, share made, mean bid.

*Size.* From the timing pilot (`PILOT` = seed 2090, 4 game pairs each of
L0, K0, F16 and PK0, excluded): batches of 4 pairs took 9-16 s at load
average about 6 (two at once through cpu-slot); the machine has run at 40-80,
so 12 pairs a batch stay well under the 300 s cap. The pilot already puts D0
far above 0 (+2.9 over 4 pairs: L0 makes 18% of its contracts, K0 42%), so
the primary is expected to clear its bar; what this study can still learn is
in the trend D0 - E16 and the recovery K0 - Z16. The primary's per-pair SD
was 2.2 on 4 pairs; the trend is a difference of two paired differences, so
the sizing takes SD 2.7 for it. 288 pairs give a half-width of
1.96*2.7/sqrt(288) = 0.31 marks/game on the trend and a minimum detectable
D0 - E16 of 2.8*2.7/sqrt(288) = 0.45 (80% power, two-sided 5%); the
primary's MDE at SD 2.2 is 0.36. So:
- marks: `SEEDS` = 2101-2124, `GAMES` = 12 pairs a batch: **288 game pairs
  per arm**, five arms (120 batches);
- 250 points: `POINT_SEEDS` = 2201-2216, 12 pairs a batch: 192 game pairs
  per arm, two arms (32 batches).

*Batch-drop rule.* A batch that exits nonzero is dropped, not rerun, and
listed. No effect is computed until every registered batch has run. Every
batch runs under `scripts/cpu-slot`, two at a time, seed-major.

*Fence.* Team A plans nothing, or Zeb's plan over the mover's own cell; the
book's rules read the frame alone ([[strategy-fusion-fence]]).

    python -u -m scenario.no_plan arena --arm K0 --seed S --games G --out DIR
    python -u -m scenario.no_plan queue --root DIR [--workers 2] [--pilot]
    python -u -m scenario.no_plan summarize --runs DIR... --out FILE.json
"""
from __future__ import annotations

from typing import Sequence

from scenario import book_batches
from scenario.small_n import _games, _meta, contrast, diff, pairs, player

MARK_ARMS = ("L0", "K0", "A0", "Z16", "F16")
POINT_ARMS = ("PL0", "PK0")
ARMS = MARK_ARMS + POINT_ARMS
PILOT = (2090,)               # the timing pilot, excluded
GAMES = 12                    # game pairs per batch
SEEDS = tuple(range(2101, 2125))         # marks: 24 batches per arm, 288 game pairs
POINT_SEEDS = tuple(range(2201, 2217))   # 250 points: 16 batches per arm, 192 game pairs


class LowestPlay:
    """The lowest legal tile, by tile id: the book player's `lowest` fallback
    with no rule above it."""

    def choose(self, states, bid_values, marks=None, marks_to_win: int = 7) -> list[int]:
        from forge.zeb.game import legal_actions

        out = []
        for s in states:
            seat = (s.trick_leader + len(s.current_trick)) % 4
            out.append(min(legal_actions(s), key=lambda slot: s.hands[seat][slot]))
        return out

    def __repr__(self) -> str:
        return "LowestPlay()"


def seats(arm: str) -> tuple:
    """(bid_a, play_a, bid_b, play_b) for an arm."""
    from arena.bidders import HeuristicBidder
    from arena.book_play import BookPlay

    name = arm.removeprefix("P")
    play_a = {
        "L0": LowestPlay, "A0": LowestPlay,
        "K0": lambda: BookPlay("lowest", "least"),
        "Z16": lambda: player(16),
        "F16": lambda: BookPlay("zeb", "zeb", player(16)),
    }[name]()
    return HeuristicBidder(), play_a, HeuristicBidder(), player(seed=1 if name == "A0" else 0)


def summarize(runs: Sequence[str]) -> dict:
    by: dict[str, list[str]] = {}
    for run in runs:
        by.setdefault(_meta(run)["arm"], []).append(run)
    marks = {a: pairs(rs, "marks") for a, rs in by.items() if not a.startswith("P")}
    won = {a: pairs(rs, "won") for a, rs in by.items() if a.startswith("P")}
    pts = {a: pairs(rs, "points") for a, rs in by.items() if a.startswith("P")}
    out: dict = {"arms": {}, "registered": {}}
    for arm, rs in sorted(by.items()):
        games = [g for r in rs for g in _games(r)]
        hands = [h for g in games for h in g["hands"]]
        bids = {s: [h for h in hands if (h["bidder_team"] == h["a_team"]) == (s == "a") and not h["forced"]]
                for s in "ab"}
        out["arms"][arm] = {
            "batches": [{x: _meta(r)[x] for x in ("seed", "wall_s", "load_avg")} for r in rs],
            "games": len(games), "hands": len(hands),
            "contracts_per_hand": {s: len(v) / max(len(hands), 1) for s, v in bids.items()},
            "made": {s: sum(h["made"] for h in v) / len(v) if v else None for s, v in bids.items()},
            "mean_bid": {s: sum(h["bid"] for h in v) / len(v) if v else None for s, v in bids.items()},
            "a_margin": contrast(marks if arm in marks else won, [(1.0, arm)]),
        }
    reg = out["registered"]
    reg["primary D0 = K0 - L0, marks/game"] = contrast(marks, diff("K0", "L0"))
    reg["A/A = A0 - L0, marks/game"] = contrast(marks, diff("A0", "L0"))
    reg["E16 = F16 - Z16, marks/game"] = contrast(marks, diff("F16", "Z16"))
    reg["trend D0 - E16, marks/game"] = contrast(marks, diff("K0", "L0") + diff("Z16", "F16"))
    reg["planning nothing: L0 - Z16, marks/game"] = contrast(marks, diff("L0", "Z16"))
    reg["the rules' recovery: K0 - Z16, marks/game"] = contrast(marks, diff("K0", "Z16"))
    reg["250: PK0 - PL0 on share of games won"] = contrast(won, diff("PK0", "PL0"))
    reg["250: PK0 - PL0 on settled points/game"] = contrast(pts, diff("PK0", "PL0"))
    return out


def jobs(pilot: bool = False) -> list[tuple[str, int]]:
    """Every registered batch, seed-major: (arm, base seed)."""
    if pilot:
        return [(a, s) for s in PILOT for a in ("L0", "K0", "F16", "PK0")]
    marks = [(a, s) for s in SEEDS for a in MARK_ARMS]
    points = [(a, s) for s in POINT_SEEDS for a in POINT_ARMS]
    return marks + points


if __name__ == "__main__":
    book_batches.main(__doc__, "scenario.no_plan", ARMS, seats, jobs, summarize, GAMES)
