# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: MIT
"""The package's match-aware bidding (zeb/match.js, via index.js `chooseBid`)
against the arena's referee (`arena/zeb_bid.py` mode "met", `arena/match_equity.py`).

Run from the repository root after the package is built (the `zeb` suite of
scripts/check does: `npm test` first). Values must agree to 1e-12 and calls
must be identical.
"""
import json
import random
import subprocess
from pathlib import Path


from arena.auction import BidContext, legal_bids
from arena.match_equity import MatchEquity
from arena.zeb_bid import ZEB_LAW, ZEB_ODDS, ZebBidder, price

ZEB = Path(__file__).parent.parent
TOL = 1e-12

NODE = """
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
const dir = process.argv[1] + '/';
const { matchPrice } = await import(pathToFileURL(dir + 'match.js'));
const { chooseBid } = await import(pathToFileURL(dir + 'index.js'));
const { prices, bids } = JSON.parse(readFileSync(0, 'utf8'));
const out = {
  prices: prices.map((p) => matchPrice(p)),
  bids: bids.map((b) => chooseBid(b.forecast, b.options)),
};
process.stdout.write(JSON.stringify(out));
"""


def node(prices, bids):
    r = subprocess.run(["node", "--input-type=module", "-e", NODE, str(ZEB)], input=json.dumps(
        {"prices": prices, "bids": bids}), capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    return out["prices"], out["bids"]


def test_pricing_grid():
    """`price` over every score pair 0..7, both nd, every holder/after key, and a grid of
    c0, p_c0, p42 and mark_call, and at other targets."""
    keys = sorted(ZEB_ODDS.pass_ours)
    cases = []
    for target in (7, 5, 3):
        for us in range(8):
            for them in range(8):
                if target != 7 and (us > target or them > target):
                    continue
                for nd in (0, 1):
                    for key in keys:
                        for c0 in ((0,) if key[0] == "partner" else (0, 30, 34, 35, 42)):
                            for p_c0 in ((0.0,) if c0 == 0 else (0.45, 0.9)):
                                for mark_call in (84, 126):
                                    for p42 in (0.05, 0.3, 0.55, 0.9):
                                        cases.append((target, us, them, nd, key, c0, p_c0, mark_call, p42))
    assert len(cases) >= 2000
    got, _ = node([
        {"us": us, "them": them, "marksToWin": target, "weShakeNext": bool(nd), "holder": key[0],
         "after": key[1], "c0": c0, "pC0": p_c0, "markCall": mark_call, "p42": p42}
        for target, us, them, nd, key, c0, p_c0, mark_call, p42 in cases], [])
    v = {t: MatchEquity(ZEB_LAW, t) for t in (7, 5, 3)}
    worst = 0.0
    for (target, us, them, nd, key, c0, p_c0, mark_call, p42), g in zip(cases, got):
        e_c0, e_mark, call = price(v[target], ZEB_ODDS, us, them, nd, key, c0, p_c0, mark_call, p42)
        assert g["call"] == call, (target, us, them, nd, key, c0, p_c0, mark_call, p42)
        worst = max(worst, abs(g["eC0"] - e_c0), abs(g["eMark"] - e_mark))
    assert worst <= TOL, worst
    print(f"pricing parity: {len(cases)} contexts, max deviation {worst:.3g}")


def _forecast(rng):
    """A package forecast (rows of 13 non-increasing tails) and the arena's (decl, pmake) form."""
    rows = []
    for decl in range(7):
        tail = sorted((round(rng.random(), rng.choice((2, 4, 12))) for _ in range(13)), reverse=True)
        if rng.random() < 0.3:
            tail[12] = rng.choice((0.9, 0.5, 0.05))  # make 42 the quantity that decides
        rows.append((decl, tuple(tail)))
    if rng.random() < 0.25:  # a tied best at 42: the first listed keeps it
        rows[3] = (3, rows[1][1])
    return rows


def test_bidder_bid_is_chooseBid_with_match():
    """`ZebBidder(mode="met").bid` against `chooseBid(..., { match })` over random auctions
    and forecasts: the same bid, every time."""
    rng = random.Random(42)
    contexts, bids = [], []
    for _ in range(3000):
        fc = _forecast(rng)
        seat, dealer = rng.randrange(4), rng.randrange(4)
        high_bid = rng.choice((0, 0, 30, 31, 33, 35, 38, 41, 42, 84, 126))
        if high_bid == 0:
            after = rng.randrange(4)
            others = [s for s in range(4) if s != seat]
            waiting = rng.sample(others, after)
            high_seat = -1
            spoken = {s: 0 for s in others if s not in waiting}
        else:
            after = rng.randrange(3)
            others = [s for s in range(4) if s != seat]
            waiting = rng.sample(others, after)
            high_seat = rng.choice([s for s in others if s not in waiting])
            spoken = {s: 0 for s in others if s not in waiting}
            spoken[high_seat] = high_bid
        order = tuple(-1 if s == seat or s in waiting else spoken[s] for s in range(4))
        legal = legal_bids(high_bid)
        marks = (rng.randrange(7), rng.randrange(7))
        points_to_win = rng.choice((None, None, None, 250))
        ctx = BidContext(
            hand=(0, 1, 6, 8, 9, 15, 20), seat=seat, dealer=dealer, bids=order, high_bid=high_bid,
            high_seat=high_seat, legal=legal, marks=marks, marks_to_win=rng.choice((7, 7, 5)),
            points_to_win=points_to_win)
        bidder = ZebBidder(None, 0.8, "met")
        bidder._fc[tuple(sorted(ctx.hand))] = fc
        mine = bidder.bid(ctx, rng)
        team = ctx.team
        partner = high_seat != -1 and high_seat % 2 == team
        holder = "none" if high_seat == -1 else "partner" if partner else "opp"
        match = {"us": marks[team], "them": marks[1 - team], "marksToWin": ctx.marks_to_win,
                 "weShakeNext": (dealer + 1) % 2 == team, "holder": holder,
                 "passesAfter": sum(1 for b in order if b == -1) - 1, "maxBid": max(legal)}
        if points_to_win is not None:
            match["pointsToWin"] = True
        contexts.append(mine)
        bids.append({"forecast": [{"decl": d, "mean": 0, "pmake": list(pm)} for d, pm in fc],
                     "options": {"need": min(legal), "cutoff": 0.8, "match": match}})
    _, got = node([], bids)
    assert len(got) == len(contexts) >= 2000
    marks_bid = 0
    for mine, theirs in zip(contexts, got):
        assert (0 if theirs is None else theirs["bid"]) == mine
        marks_bid += mine >= 84
    assert 100 < marks_bid < len(contexts) - 100, marks_bid  # the grid exercises both sides of the choice
    print(f"bidder parity: {len(contexts)} auctions, {marks_bid} mark bids")


def test_the_package_is_built():
    assert (ZEB / "pkg" / "zeb.js").exists(), "run `npm test` in zeb/ first (it builds the wasm)"
