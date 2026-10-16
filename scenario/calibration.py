# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""Calibrating the Book desk's estimator (#191): how often does an A/A miss?

Two A/A arms missed their 95% intervals in opposite directions: plain Zeb at
planner seed 1 against the seed-0 field read +0.12 [+0.02, +0.23] marks/game
(#117), and a seed-varied A/A read -0.14 [-0.25, -0.03] (#190). Neither a
per-seed effect nor batch correlation explains it, and the planner's choice
for a state does not depend on its batch (40 states, alone and in shuffled
batches: 0 differences). So the question is the estimator: the paired halves
and the percentile bootstrap over game pairs (`scenario.small_n.contrast`).

Two seeds are two deterministic policies, and one can truly be better than
the other by the luck of its seed: a nonzero difference between seed 1 and
seed 0 is not miscalibration. This design separates the two. It holds the
pair of policies fixed across replicates on disjoint deals (F), which pins
their true difference and tests whether the intervals cover it, and draws a
fresh pair of seeds per replicate (V), which measures seed-to-seed spread.

PRE-REGISTRATION (committed with this file before any measured batch).

*Players.* Every seat bids by the heuristic rule at caution 0 and plays
deployed Zeb 0.3.0's plan (`scenario.small_n.player`) at a fixed planner
seed. **F<r>** (r = 0..19): team A at seed 1, team B at seed 0, on deal
block r. **V<r>** (r = 0..19): team A at seed 100 + 2r, team B at seed
101 + 2r, on its own deal block. Marks (games to 7) only.

*Size.* Each replicate is 24 batches of 12 game pairs, **288 game pairs**,
on base seeds disjoint from every other replicate and every prior study:
F<r> on 5001 + 24r .. 5024 + 24r, V<r> on 6001 + 24r .. 6024 + 24r. 40
replicates, 960 batches. (288 rather than 576 a replicate buys twenty
replicates an arm at the same cost; the house estimator is the same.)

*Estimator.* Each replicate's mean and 95% interval exactly as the house
reports one: `scenario.small_n.contrast` (game pairs, percentile bootstrap,
10,000 resamples, seed 42). Its standard error is read as (hi - lo)/3.92.

*Registered reads.*
1. **Calibration (F).** R_F = SD of the twenty F means / RMS of their
   standard errors. A calibrated estimator gives R_F near 1; under
   normality its 95% range at 19 degrees of freedom is about [0.68, 1.31].
   **R_F above 1.31 = the intervals undercover.** Also: how many F
   intervals miss the pooled F mean (expect 1 of 20; 4 or more is p < 0.02).
2. **delta(1,0).** The pooled F mean over all 5,760 game pairs, with its
   bootstrap interval: the true difference between the seed-1 and seed-0
   policies, which #117's A/A estimated as +0.12.
3. **Seed spread (V).** R_V the same ratio for V, and the between-pair
   spread sigma_seed^2 = var(V means) - mean(SE^2), floored at 0, as
   marks/game. How many V intervals miss 0.
4. **Tails.** The per-pair distribution pooled over F: its kurtosis and
   the share of pairs at |margin| >= 6, as a check on the percentile
   bootstrap.

*Batch-drop rule.* As `scenario.small_n`. Every batch through
`scripts/cpu-slot`. Every player thinks on a fixed world budget.

    python -u -m scenario.calibration arena --arm F3 --seed S --games G --out DIR
    python -u -m scenario.calibration queue --root DIR [--workers 6]
    python -u -m scenario.calibration summarize --runs DIR... --out FILE.json
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from scenario import book_batches
from scenario.small_n import RESAMPLES, _meta, contrast, pairs, player

REPLICATES = 20
BATCHES = 24                  # batches a replicate
GAMES = 12                    # game pairs a batch
ARMS = tuple(f"{k}{r}" for k in "FV" for r in range(REPLICATES))


def block(arm: str) -> range:
    """The base seeds of an arm's replicate."""
    start = (5001 if arm[0] == "F" else 6001) + BATCHES * int(arm[1:])
    return range(start, start + BATCHES)


def seeds_of(arm: str) -> tuple[int, int]:
    """(team A's planner seed, team B's)."""
    r = int(arm[1:])
    return (1, 0) if arm[0] == "F" else (100 + 2 * r, 101 + 2 * r)


def seats(arm: str) -> tuple:
    from arena.bidders import HeuristicBidder

    if arm not in ARMS:
        raise ValueError(f"no arm {arm}")
    a, b = seeds_of(arm)
    return HeuristicBidder(), player(seed=a), HeuristicBidder(), player(seed=b)


def _ratio(rows: list[dict]) -> dict:
    means = np.array([r["mean"] for r in rows])
    ses = np.array([(r["hi"] - r["lo"]) / 3.92 for r in rows])
    rms = float(np.sqrt((ses ** 2).mean()))
    return {"replicates": len(rows), "sd_of_means": float(means.std(ddof=1)), "rms_se": rms,
            "ratio": float(means.std(ddof=1)) / rms,
            "excess_var": float(max(0.0, means.var(ddof=1) - (ses ** 2).mean()))}


def summarize(runs: Sequence[str]) -> dict:
    by: dict[str, list[str]] = {}
    for run in runs:
        by.setdefault(_meta(run)["arm"], []).append(run)
    marks = {a: pairs(rs, "marks") for a, rs in by.items()}
    reps = {a: {**contrast(marks, [(1.0, a)]), "seeds": seeds_of(a), "batches": len(by[a])} for a in sorted(by)}
    f = [reps[a] for a in reps if a[0] == "F"]
    v = [reps[a] for a in reps if a[0] == "V"]
    pooled = {k: x for a in marks if a[0] == "F" for k, x in marks[a].items()}
    delta = contrast({"F": pooled}, [(1.0, "F")])
    d = np.array(list(pooled.values()))
    out = {"replicates": reps, "registered": {}}
    reg = out["registered"]
    reg["1 calibration (F): SD of means / RMS SE"] = _ratio(f)
    reg["1 F intervals missing the pooled F mean"] = sum(not (r["lo"] <= delta["mean"] <= r["hi"]) for r in f)
    reg["2 delta(1,0), pooled F, marks/game"] = delta
    vr = _ratio(v)
    reg["3 seed spread (V): SD of means / RMS SE"] = vr
    reg["3 sigma_seed, marks/game"] = math.sqrt(vr["excess_var"])
    reg["3 V intervals missing 0"] = sum(not (r["lo"] <= 0 <= r["hi"]) for r in v)
    z = (d - d.mean()) / d.std()
    reg["4 tails (pooled F pairs)"] = {"pairs": len(d), "excess_kurtosis": float((z ** 4).mean() - 3),
                                      "share_at_6_or_more": float((np.abs(d) >= 6).mean()),
                                      "resamples": RESAMPLES}
    return out


def jobs(pilot: bool = False) -> list[tuple[str, int]]:
    """Every registered batch, interleaved so every replicate fills at the same pace."""
    if pilot:
        raise SystemExit("no pilot: the batch is #117's plain-Zeb batch, timed there")
    return [(a, block(a)[i]) for i in range(BATCHES) for a in ARMS]


if __name__ == "__main__":
    book_batches.main(__doc__, "scenario.calibration", ARMS, seats, jobs, summarize, GAMES)
