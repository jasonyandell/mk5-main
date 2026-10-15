# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""The book's own overbidders (#163): the book's bidding against the two
kinds of overbidder it names, at power.

#152 seated the book's bidding (`risk-budget`) against a field that overbids
by a fixed step: its cost shrank from 0.92 to 0.57 marks/game as the step
grew to 10 points, a null at MDE 0.52 ([[deviant-field-2026-10-14]]). That
field overbid by rote. The book names two overbidders of its own
([[book-assumptions]] A6; catalog rows in our words): one raises a weak hand
over an opponent's bid because he doubts he could set it (S-0155-a, the
set-bidding fallacy), the other habitually bids 35 holding offs in the
ten-count suits (S-0070-a). Its answer to both is restraint: pass and set
him (S-0080-c).

PRE-REGISTRATION (committed with this file before any measured batch; the
timing pilot, `PILOT`, is excluded from every number).

*Players.* Every seat plays Zeb 0.3.0's plan (`scenario.small_n.player()`).
Team B, the field, bids by one of:
- **S**: `silas/examples/populations/set-bid.silas`, bid-only-enough except
  that over an opponent's standing bid it raises while the least raise and the
  count exposed come to at most 48 (six points past the hand's worth);
- **T**: `silas/examples/populations/ten-count-35.silas`, 35 on a suit of
  three with an off that exposes a ten-count tile, else bid-only-enough;
- **H**: the arena's heuristic bidder at caution 0, the style-free field.
Team A bids by **H** (the heuristic at caution 0, the control), **R** (the
book's `risk-budget`), or plays as **B** (the head-to-head's book player,
block c, bidding H).

*Arms* (team A, then the field). Marks (games to 7): HS, RS, BS, HT, RT, BT,
RH. The book's game (250 points): PHS, PRS, PHT, PRT, PRH. HH is the same
player both sides and reads 0 exactly, so it is not run.

*Estimator.* As `scenario.small_n`.

*Primary.* For each overbidder f in {S, T}: the book's bidding's value
against it, relative to the style-free field, X_f = (R_f - H_f) - RH, in
marks per game. **The book predicts X_f > 0**: its restraint is worth more
against the overbidders it was written for. The bar: X_f's interval above 0
= the book's bidding pays more against its own overbidders; below 0 = less;
else a null with its MDE.

*Registered secondaries.* C_f = R_f - H_f (whether the book's bidding still
costs against each overbidder); the field against the control, H_f (the book
predicts overbidders set themselves: H_f > 0); the book player, B_f - H_f;
the book's game, PR_f - PH_f on games won and settled points, and PRH;
mechanism by arm: contracts per hand, share made, mean bid.

*Size.* From the timing pilot (`PILOT` = seed 2590, 4 game pairs each of
RS, HT, BT and PRT, excluded): 17-23 s a batch, so 12 pairs stay well under
the 300 s cap. In the pilot the ten-count field took 87% of the contracts at
a mean bid of 35 and made 39-43% of them. The primary combines three arms; #152's
single contrast came in at SD 2.4 per pair, so the sizing takes SD 3.3 for X_f.
576 game pairs give a half-width of 1.96*3.3/sqrt(576) = 0.27 marks/game and
an MDE of 2.8*3.3/sqrt(576) = 0.39 (80% power, two-sided 5%). So:
- marks: `SEEDS` = 2601-2648, 12 pairs a batch: **576 game pairs per arm**,
  seven arms (336 batches);
- 250 points: `POINT_SEEDS` = 2701-2716: 192 game pairs per arm, five arms
  (80 batches).

*Batch-drop rule and fence.* As `scenario.small_n`; every batch through
`scripts/cpu-slot`, two at a time, seed-major.

    python -u -m scenario.overbidders arena --arm RS --seed S --games G --out DIR
    python -u -m scenario.overbidders queue --root DIR [--workers 2] [--pilot]
    python -u -m scenario.overbidders summarize --runs DIR... --out FILE.json
"""
from __future__ import annotations

from typing import Sequence

from scenario import book_batches
from scenario.small_n import _games, _meta, contrast, diff, pairs, player

FIELDS = {"S": "populations/set-bid", "T": "populations/ten-count-35"}
MARK_ARMS = ("HS", "RS", "BS", "HT", "RT", "BT", "RH")
POINT_ARMS = ("PHS", "PRS", "PHT", "PRT", "PRH")
ARMS = MARK_ARMS + POINT_ARMS
PILOT = (2590,)               # the timing pilot, excluded
GAMES = 12                    # game pairs per batch
SEEDS = tuple(range(2601, 2649))         # marks: 48 batches per arm, 576 game pairs
POINT_SEEDS = tuple(range(2701, 2717))   # 250 points: 16 batches per arm, 192 game pairs


def bidder(code: str):
    from arena.bidders import HeuristicBidder
    from arena.book_play import SilasBidder

    if code == "H":
        return HeuristicBidder()
    if code == "R":
        return SilasBidder("risk-budget")
    return SilasBidder(FIELDS[code])


def seats(arm: str) -> tuple:
    """(bid_a, play_a, bid_b, play_b) for an arm."""
    from arena.book_play import BookPlay

    a, field = arm.removeprefix("P")
    play_a = BookPlay("zeb", "zeb", player()) if a == "B" else player()
    return bidder("H" if a == "B" else a), play_a, bidder(field), player()


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
    for f in FIELDS:
        reg[f"primary X{f} = (R{f} - H{f}) - RH, marks/game"] = contrast(marks, diff(f"R{f}", f"H{f}") + [(-1.0, "RH")])
    for f in FIELDS:
        reg[f"C{f} = R{f} - H{f}, marks/game"] = contrast(marks, diff(f"R{f}", f"H{f}"))
        reg[f"the field: H{f} against 0, marks/game"] = contrast(marks, [(1.0, f"H{f}")])
        reg[f"the book player: B{f} - H{f}, marks/game"] = contrast(marks, diff(f"B{f}", f"H{f}"))
    reg["RH (the book's bidding against the style-free field), marks/game"] = contrast(marks, [(1.0, "RH")])
    for f in FIELDS:
        reg[f"250: PR{f} - PH{f} on share of games won"] = contrast(won, diff(f"PR{f}", f"PH{f}"))
        reg[f"250: PR{f} - PH{f} on settled points/game"] = contrast(pts, diff(f"PR{f}", f"PH{f}"))
    reg["250: PRH on share of games won"] = contrast(won, [(1.0, "PRH")])
    return out


def jobs(pilot: bool = False) -> list[tuple[str, int]]:
    """Every registered batch, seed-major: (arm, base seed)."""
    if pilot:
        return [(a, s) for s in PILOT for a in ("RS", "HT", "BT", "PRT")]
    marks = [(a, s) for s in SEEDS for a in MARK_ARMS]
    points = [(a, s) for s in POINT_SEEDS for a in POINT_ARMS]
    return marks + points


if __name__ == "__main__":
    book_batches.main(__doc__, "scenario.overbidders", ARMS, seats, jobs, summarize, GAMES)
