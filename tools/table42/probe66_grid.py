#!/usr/bin/env python3
"""Issue #66 follow-up: the full claimed-vs-measured grid for Jud's hand-3 hand.

Hand: 0-0 2-1 3-2 5-4 6-1 6-3 6-4. Conditioning as at the table (canonical
bids (0,0,0,30), seat 2, dealer 2). For every pip declaration: the head's
claimed P(pts >= thr) row (jud_net.pmake_table — the object it bid from)
vs the measured make-rate from N world-replays under jud self-play with
that declaration forced at a 31 contract.
"""
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from arena.jud_play import JudPlay  # noqa: E402
from champion.jud_net import load_jud_net  # noqa: E402
from forge.oracle.declarations import DECL_ID_TO_NAME, GAME_DECL_IDS  # noqa: E402
from forge.oracle.tables import DOMINOES  # noqa: E402
from forge.zeb.game import apply_action, is_terminal  # noqa: E402
from forge.zeb.types import BidState, GamePhase, ZebGameState  # noqa: E402

DOM_ID = {f"{h}-{l}": i for i, (h, l) in enumerate(DOMINOES)}
HAND = tuple(sorted(DOM_ID[t] for t in "0-0 2-1 3-2 5-4 6-1 6-3 6-4".split()))
REST = [i for i in range(28) if i not in HAND]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 500

net = load_jud_net(REPO / "champion" / "jud_net.pt", device="cpu")
play = JudPlay(net)

claimed = net.pmake_table(HAND, (0, 0, 0, 30), 2, 2)

def measure(decl: int) -> tuple[float, float]:
    rng = random.Random(868829779 + decl)
    states = []
    for _ in range(N):
        r = REST[:]
        rng.shuffle(r)
        states.append(ZebGameState(
            hands=(tuple(sorted(r[0:7])), tuple(sorted(r[7:14])), HAND,
                   tuple(sorted(r[14:21]))),
            dealer=2, phase=GamePhase.PLAYING,
            bid_state=BidState(bids=(0, 0, 31, 30), high_bidder=2, high_bid=31),
            decl_id=decl, bidder=2, played=frozenset(), play_history=(),
            current_trick=(), trick_leader=2, team_points=(0, 0)))
    while True:
        live = [i for i, s in enumerate(states) if not is_terminal(s)]
        if not live:
            break
        choices = play.choose([states[i] for i in live], [31] * len(live))
        for i, c in zip(live, choices):
            states[i] = apply_action(states[i], c)
    pts = [s.team_points[0] for s in states]
    return sum(1 for p in pts if p >= 31) / N, sum(pts) / N

print(f"hand: 0-0 2-1 3-2 5-4 6-1 6-3 6-4 · contract 31 · N={N}/decl · jud self-play")
print(f"{'decl':<8} {'claimed P(>=30)':>15} {'claimed P(>=31)':>15} {'MEASURED P(>=31)':>17} {'gap':>7} {'mean pts':>9}")
t0 = time.time()
for d in GAME_DECL_IDS:
    name = DECL_ID_TO_NAME[d]
    c30, c31 = claimed[d][30], claimed[d][31]
    m31, mean = measure(d)
    print(f"{name:<8} {c30:>15.3f} {c31:>15.3f} {m31:>17.3f} {c31-m31:>+7.3f} {mean:>9.1f}", flush=True)
print(f"\n{time.time()-t0:.0f}s total")
