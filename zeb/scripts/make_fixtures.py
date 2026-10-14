"""Write the parity fixtures: positions planned by the Python referee
(`scenario.planner.Walt`) under exactly the configs the zeb package uses.

    python -u -m zeb.scripts.make_fixtures

Opening leads (the forecast's question) and mid-hand positions across the
seven pip declarations, doubles-trump and no-trump: bidders and defenders,
forced moves, twelve of the thirteen contracts (every one but 32), one seed
past 2^53, and a few pip positions asked with reading off (`reading:
false`). Play is random legal play by the engine (`forge.zeb.game`),
seeded, so the file is the same on every run. The net is the blob the
package embeds, read from zeb/net, and its sha256 is recorded so a stale
file is caught. It refuses to write a file in which neither of 0.3.0's
defaults acts: reading must move q at three positions or more, and the tie
rule the tile at one or more (asked of the compiled core, which the
fixtures themselves hold to the referee).

`tests/fixtures/parity.json` holds, per position, the request as the
package takes it (the seed as a decimal string, since JSON numbers stop at
2^53) and the referee's answer: the tile, whether it was forced, q, the
scenarios' weights, every legal tile's per-scenario final margins, the
tails read from them, and — at opening leads — the forecast. The tails and
means are summed in scenario order with Python's float `sum`, the
summation the core mirrors, so they are bit-exact targets too.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from forge.zeb.game import apply_action, current_player, is_terminal, legal_actions, new_game

from scenario.field import CANON, FieldNet
from scenario.planner import Config, Reading, Rung, Walt

ZEB = Path(__file__).resolve().parents[1]
NET = ZEB / "net" / "margin_d250k_h128x64.fnet"
OUT = ZEB / "tests" / "fixtures" / "parity.json"
TARGETS = range(30, 43)
FLOOR = 16
READING = Reading("uniform", 1 << 22, 0.28)   # zeb/src/lib.rs READING

# (declaration, plays before the position, worlds) per mid-hand case; each
# case's deal, bidder, contract and play come from its own seeded stream.
MIDHAND = [
    (0, 1, 64), (1, 3, 32), (2, 5, 48), (3, 7, 32), (4, 9, 64), (5, 11, 32), (6, 13, 48),
    (0, 15, 32), (1, 17, 24), (2, 19, 32), (3, 21, 16), (4, 23, 24), (5, 25, 16), (6, 26, 16),
    (6, 4, 160), (3, 2, 160), (5, 6, 64),
    (7, 2, 160), (7, 6, 32), (7, 12, 24), (7, 20, 16), (9, 3, 160), (9, 8, 32), (9, 14, 24), (9, 22, 16),
    (1, 17, 160),  # the fiber (124 deals) is enumerated, so the floor stands down: q moves (0.2.0)
]
# Mid-hand pip cases asked with reading off (`reading: false`), the 0.2.0 draw.
UNREAD = [(2, 5, 48), (6, 10, 64), (4, 18, 160)]
WORKERS = 6                  # positions planned at once, one process each
# (declaration, worlds) per opening lead, from seat 0 having bid 30.
OPENING = [(0, 160), (1, 160), (2, 160), (3, 160), (4, 512), (5, 160), (6, 160), (7, 160), (9, 160)]
BIG_SEED = (1 << 63) + 12345


def config(decl: int, worlds: int, seed: int, net: FieldNet, reading: bool = True, tie: str = "margin") -> Config:
    """The package's plan: the cheap player over the net under a pip trump,
    drawing its worlds from the soft uniform reading's posterior unless
    reading is off; a two-rung Walt (m8, a 4-play inner horizon) over
    uniform worlds elsewhere. Exact ties go to make, then margin (0.3.0);
    `tie="lowest"` is 0.2.0's rule, asked only to show the fixtures move."""
    if decl in CANON:
        if reading:
            return Config((Rung(worlds, None, FLOOR),), "make", seed, field=net, draw="posterior",
                          reading=READING, tie=tie)
        return Config((Rung(worlds, None, FLOOR),), "make", seed, field=net, tie=tie)
    return Config((Rung(8, 4), Rung(worlds, None, FLOOR)), "make", seed, tie=tie)


def tails(margins, weights) -> list[float]:
    total = sum(weights)
    return [sum(w for m, w in zip(margins, weights) if m >= 2 * b - 42) / total for b in TARGETS]


def position(game_seed: int, decl: int, depth: int, bid: int | None) -> tuple:
    """A state `depth` plays into a seeded deal under `decl`, by random legal
    play; the bidder, unless given, is the engine's."""
    rng = random.Random(game_seed)
    state = replace(new_game(game_seed), decl_id=decl)
    if bid is not None:
        state = replace(state, bidder=0, trick_leader=0)
    for _ in range(depth):
        state = apply_action(state, rng.choice(legal_actions(state)))
    assert not is_terminal(state)
    return state, bid if bid is not None else rng.randint(30, 42)


def case(job: tuple) -> tuple[dict, bool, bool]:
    """One position: the request, the referee's answer, and whether each
    0.3.0 default moved it — reading (q against the unread plan) and the tie
    rule (the tile against the lowest id's), those two asked of the core."""
    from scenario.compiled import CompiledWalt

    game_seed, decl, depth, given_bid, worlds, seed, reading = job
    net = FieldNet.from_bytes(NET.read_bytes())
    state, bid = position(game_seed, decl, depth, given_bid)
    seat = current_player(state)
    request = {
        "decl": state.decl_id, "bidder": state.bidder, "bid": bid, "seat": seat,
        "hand": sorted(state.hands[seat]), "plays": [list(p) for p in state.play_history],
        "worlds": worlds, "seed": str(seed),
    } | ({} if reading else {"reading": False})
    a = Walt(config(decl, worlds, seed, net, reading)).analyze(state, bid)
    forced = not a.worlds
    referee = {"tile": a.tile, "forced": forced}
    if forced:
        return {"request": request, "referee": referee}, False, False
    unread = CompiledWalt(config(decl, worlds, seed, net, False), threads=1).analyze(state, bid)
    lowest = CompiledWalt(config(decl, worlds, seed, net, reading, "lowest"), threads=1).analyze(state, bid)
    assert all(float(m).is_integer() for o in a.outcomes.values() for m in o)
    referee |= {
        "q": [[t, a.q[t]] for t in sorted(a.q)],
        "weights": a.weights,
        "outcomes": [[t, [int(m) for m in a.outcomes[t]]] for t in sorted(a.outcomes)],
        "pmake": [[t, tails(a.outcomes[t], a.weights)] for t in sorted(a.outcomes)],
    }
    if not state.play_history and state.bidder == seat == 0 and bid == 30:
        chosen = a.outcomes[a.tile]
        mean = sum(m * w for m, w in zip(chosen, a.weights)) / sum(a.weights)
        referee["forecast"] = {"decl": state.decl_id, "mean": mean, "pmake": tails(chosen, a.weights)}
    read_moved = reading and decl in CANON and unread.q != a.q
    return {"request": request, "referee": referee}, read_moved, lowest.tile != a.tile


def main() -> None:
    blob = NET.read_bytes()
    jobs = [(1000 + i, decl, 0, 30, worlds, 7 * i + 1, True) for i, (decl, worlds) in enumerate(OPENING)]
    jobs += [(2000 + i, decl, depth, None, worlds, BIG_SEED if i == 0 else 11 * i + 3, True)
             for i, (decl, depth, worlds) in enumerate(MIDHAND)]
    jobs += [(3000 + i, decl, depth, None, worlds, 13 * i + 5, False)
             for i, (decl, depth, worlds) in enumerate(UNREAD)]
    t0 = time.time()
    # The slowest positions (doubles and no-trump: the two-rung referee) start first.
    order = sorted(range(len(jobs)), key=lambda i: jobs[i][1] in CANON)
    with ProcessPoolExecutor(WORKERS) as pool:
        done = dict(zip(order, pool.map(case, [jobs[i] for i in order])))
    results = [done[i] for i in range(len(jobs))]
    cases = [c for c, _, _ in results]
    moved = {"reading": sum(r for _, r, _ in results), "tie": sum(t for _, _, t in results)}
    forced = sum(c["referee"]["forced"] for c in cases)
    assert forced >= 2, f"only {forced} forced positions; pick other depths"
    assert moved["reading"] >= 3 and moved["tie"] >= 1, f"the fixtures must show both defaults act: {moved}"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = {"net": hashlib.sha256(blob).hexdigest(), "positions": cases}
    OUT.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    print(f"{OUT}: {len(cases)} positions ({forced} forced; reading moved q at {moved['reading']}, "
          f"the tie rule moved the tile at {moved['tie']}), {OUT.stat().st_size:,} bytes, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
