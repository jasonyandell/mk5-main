"""Write the parity fixtures: positions planned by the Python referee
(`scenario.planner.Walt`) under exactly the configs the zeb package uses.

    python -u -m zeb.scripts.make_fixtures

Opening leads (the forecast's question) and mid-hand positions across the
seven pip declarations, doubles-trump and no-trump: bidders and defenders,
forced moves, twelve of the thirteen contracts (every one but 32), one seed
past 2^53. Play is
random legal play by the engine (`forge.zeb.game`), seeded, so the file is
the same on every run. The net is the blob the package embeds, read from
zeb/net, and its sha256 is recorded so a stale file is caught.

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
from dataclasses import replace
from pathlib import Path

from forge.zeb.game import apply_action, current_player, is_terminal, legal_actions, new_game

from scenario.field import CANON, FieldNet
from scenario.planner import Config, Rung, Walt

ZEB = Path(__file__).resolve().parents[1]
NET = ZEB / "net" / "margin_d250k_h128x64.fnet"
OUT = ZEB / "tests" / "fixtures" / "parity.json"
TARGETS = range(30, 43)
FLOOR = 16

# (declaration, plays before the position, worlds) per mid-hand case; each
# case's deal, bidder, contract and play come from its own seeded stream.
MIDHAND = [
    (0, 1, 64), (1, 3, 32), (2, 5, 48), (3, 7, 32), (4, 9, 64), (5, 11, 32), (6, 13, 48),
    (0, 15, 32), (1, 17, 24), (2, 19, 32), (3, 21, 16), (4, 23, 24), (5, 25, 16), (6, 26, 16),
    (6, 4, 160), (3, 2, 160), (5, 6, 64),
    (7, 2, 160), (7, 6, 32), (7, 12, 24), (7, 20, 16), (9, 3, 160), (9, 8, 32), (9, 14, 24), (9, 22, 16),
    (1, 17, 160),  # the fiber (124 deals) is enumerated, so the floor stands down: q moves (0.2.0)
]
# (declaration, worlds) per opening lead, from seat 0 having bid 30.
OPENING = [(0, 160), (1, 160), (2, 160), (3, 160), (4, 512), (5, 160), (6, 160), (7, 160), (9, 160)]
BIG_SEED = (1 << 63) + 12345


def config(decl: int, worlds: int, seed: int, net: FieldNet) -> Config:
    """The package's plan: the cheap player over the net under a pip trump,
    a two-rung Walt (m8, a 4-play inner horizon) elsewhere."""
    if decl in CANON:
        return Config((Rung(worlds, None, FLOOR),), "make", seed, field=net)
    return Config((Rung(8, 4), Rung(worlds, None, FLOOR)), "make", seed)


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


def case(state, bid: int, worlds: int, seed: int, net: FieldNet) -> dict:
    seat = current_player(state)
    request = {
        "decl": state.decl_id, "bidder": state.bidder, "bid": bid, "seat": seat,
        "hand": sorted(state.hands[seat]), "plays": [list(p) for p in state.play_history],
        "worlds": worlds, "seed": str(seed),
    }
    a = Walt(config(state.decl_id, worlds, seed, net)).analyze(state, bid)
    forced = not a.worlds
    referee = {"tile": a.tile, "forced": forced}
    if not forced:
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
    return {"request": request, "referee": referee}


def main() -> None:
    blob = NET.read_bytes()
    net = FieldNet.from_bytes(blob)
    cases = []
    t0 = time.time()
    for i, (decl, worlds) in enumerate(OPENING):
        state, bid = position(1000 + i, decl, 0, 30)
        cases.append(case(state, bid, worlds, 7 * i + 1, net))
        print(f"opening decl {decl} n{worlds}: {time.time() - t0:.0f}s", flush=True)
    for i, (decl, depth, worlds) in enumerate(MIDHAND):
        state, bid = position(2000 + i, decl, depth, None)
        seed = BIG_SEED if i == 0 else 11 * i + 3
        cases.append(case(state, bid, worlds, seed, net))
        print(f"mid-hand decl {decl} after {depth} plays n{worlds}: {time.time() - t0:.0f}s", flush=True)
    forced = sum(c["referee"]["forced"] for c in cases)
    assert forced >= 2, f"only {forced} forced positions; pick other depths"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = {"net": hashlib.sha256(blob).hexdigest(), "positions": cases}
    OUT.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    print(f"{OUT}: {len(cases)} positions ({forced} forced), {OUT.stat().st_size:,} bytes, "
          f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
