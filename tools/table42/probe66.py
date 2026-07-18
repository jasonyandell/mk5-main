#!/usr/bin/env python3
"""Issue #66 probe: replay Jud's hand-3 auction seat.

Jud held 0-0 2-1 3-2 5-4 6-1 6-3 6-4, heard (Jed 30, Jason pass, Claude
pass), bid 31 and declared fours. Sample N worlds uniform over the 21
unseen tiles (the physics posterior u — deliberately eq-style, no g),
play each out under jud-v1 self-play (JudPlay all four seats), and report
the realized distribution of bidding-team points: the candlewax of the
hand. Registered predictions: jud 0.68, Jason 0.20, Claude ~0.4x.
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
from forge.zeb.game import is_terminal  # noqa: E402
from forge.zeb.game import apply_action  # noqa: E402
from forge.zeb.types import BidState, GamePhase, ZebGameState  # noqa: E402

DOM_ID = {f"{h}-{l}": i for i, (h, l) in enumerate(DOMINOES)}
JUD_HAND = tuple(sorted(DOM_ID[t] for t in "0-0 2-1 3-2 5-4 6-1 6-3 6-4".split()))
REST = [i for i in range(28) if i not in JUD_HAND]
FOURS = next(d for d in GAME_DECL_IDS if DECL_ID_TO_NAME[d] == "fours")

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
rng = random.Random(868829779)

net = load_jud_net(REPO / "champion" / "jud_net.pt", device="cpu")
play = JudPlay(net)

states = []
for _ in range(N):
    r = REST[:]
    rng.shuffle(r)
    hands = (tuple(sorted(r[0:7])), tuple(sorted(r[7:14])), JUD_HAND,
             tuple(sorted(r[14:21])))
    states.append(ZebGameState(
        hands=hands, dealer=2, phase=GamePhase.PLAYING,
        bid_state=BidState(bids=(0, 0, 31, 30), high_bidder=2, high_bid=31),
        decl_id=FOURS, bidder=2, played=frozenset(), play_history=(),
        current_trick=(), trick_leader=2, team_points=(0, 0)))

t0 = time.time()
step = 0
while True:
    live_idx = [i for i, s in enumerate(states) if not is_terminal(s)]
    if not live_idx:
        break
    subset = [states[i] for i in live_idx]
    choices = play.choose(subset, [31] * len(subset))
    for i, c in zip(live_idx, choices):
        states[i] = apply_action(states[i], c)
    step += 1
    if step % 7 == 0:
        print(f"  step {step}: {len(live_idx)} live, {time.time()-t0:.0f}s", flush=True)

pts = sorted(s.team_points[0] for s in states)
make31 = sum(1 for p in pts if p >= 31) / N
print(f"\nN={N} worlds, jud-v1 self-play, {time.time()-t0:.0f}s")
print(f"bidding-team points: mean {sum(pts)/N:.1f}, "
      f"min {pts[0]}, p25 {pts[N//4]}, median {pts[N//2]}, p75 {pts[3*N//4]}, max {pts[-1]}")
print(f"\nP(make) by threshold (the candlewax row):")
for thr in (30, 31, 32, 35, 42):
    print(f"  >= {thr}: {sum(1 for p in pts if p >= thr)/N:.3f}")
print(f"\nP(make 31) = {make31:.3f}   [claims: jud 0.68 · Jason 0.20 · Claude ~0.4x]")
hist = {}
for p in pts:
    hist[p // 5 * 5] = hist.get(p // 5 * 5, 0) + 1
print("\npoints histogram (5-pt buckets):")
for b in sorted(hist):
    print(f"  {b:2d}-{b+4:2d}: {'#' * max(1, hist[b]*60//N)} {hist[b]}")
