# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""Walt's bidder in the arena: Zeb's bidder (`arena/zeb_bid.py`) holding
Walt's planner.

The forecast is the honest plan's root splat. For each pip declaration
(decls 0-6), one plan from the bidder's chair as seat 0 before the opening
lead under a 30 contract, by the `CompiledWalt` of an arena walt play spec
(`arena.walt_play.parse_walt_spec`); its chosen lead's tails read at every
target 30..42 are pmake. The floor is what makes the splat a forecast: the
clairvoyant tail (f=1) runs twenty points hot, the floored plan is calibrated
(wiki/entities/scenario-player.md § "The floor calibrates the forecast").
Zeb's bidder is the same read over the net planner.

The call is `choose_bid`, the hand-only rule: the highest target some
declaration makes with probability at least the cutoff (0.8), declaring the
one most likely to make it; never over the partner, never 84.
"""
from __future__ import annotations

from arena.walt_play import parse_walt_spec
from arena.zeb_bid import ZebBidder

# Two rungs, 160 root worlds, inner 8, the honest floor 16, the random-play field, utility make.
DEFAULT_SPEC = "k1,n160,m8"


def parse_walt_bidder(arg: str) -> ZebBidder:
    """`waltbid[:<walt play options>][,c<cutoff>]`: `c<cutoff>` is the
    cutoff (default 0.8); every other token is a walt play option, over the
    default `k1,n160,m8` (`waltbid:k1,n160,m8,c0.8` is the default bidder).
    Refused: `u=points` (the forecast reads the make plan's margins at 30),
    `DTNT=` (the forecast covers pip trumps only) and `py` (the forecast
    plans the seven declarations as one batch, in the compiled core)."""
    cutoff, options = 0.8, []
    for t in (p for p in arg.split(",") if p):
        if t[0] == "c" and not t.startswith("c="):
            cutoff = float(t[1:])
        elif t.startswith("u=") and t != "u=make":
            raise ValueError(f"waltbid forecasts from the make plan's margins at 30: u=make, not {t!r}")
        elif t.startswith("DTNT="):
            raise ValueError(f"waltbid forecasts pip trumps only (decls 0-6), as Zeb's bidder does: no {t!r}")
        elif t == "py":
            raise ValueError("waltbid plans the seven declarations as one batch in the compiled core: no 'py'")
        else:
            options.append(t)
    play = parse_walt_spec(",".join([DEFAULT_SPEC, *options]))
    return ZebBidder(play.walt, cutoff, "hand", label=f"waltbid:{arg}")
