"""Price guard-premium cells (otis guard-premium probe, registered page).

For each collected cell (arm A: actor holds 3-2 + exactly one junk two among
slough candidates): build the twin (arm B: 3-2 swapped for a neutral tile
from another seat's unplayed hand, full history replay-legal), price every
candidate discard in both arms with BOTH pricers (V1 tied rollouts, V2 fate
head), and grade:

  retention margin(g; s) = max_{c in C\\{g}} price(c) - price(g)
  guard premium(cell)    = margin(junk2; A) - margin(junk2; B)

G3 mechanism channel (arm A only): P(my team captures the 3-2) on the
successor of keeping vs discarding the junk two, read from the fate head.

Usage:
    PYTHONPATH=$PWD python -u scratch/otis-night2/price_guard_cells.py \
        scratch/otis-night2/guard_cells.jsonl \
        --out scratch/otis-night2/guard_premiums.jsonl
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from arena.match import _bootstrap_ci_mean
from arena.slough_override import DEFAULT_STUDENT, TiedSloughPlay
from forge.eq.game_tensor import GameStateTensor
from forge.zeb.game import apply_action
from forge.zeb.types import BidState, GamePhase, ZebGameState
from otis.fates import (
    COUNT_TILE_IDS,
    DOMINO_COUNT_POINTS,
    DOMINO_HIGH,
    DOMINO_LOW,
    pips_to_domino_id,
)
from otis.model import TILE_PIPS
from otis.play_model import OtisPlayNet, featurize_play_state
from otis.tiedroll import load_tied_policy

TILE_32 = pips_to_domino_id("3-2")
IDX_32 = TILE_PIPS.index("3-2")
TRUMP_PIP_DECLS = set(range(7))  # decl 0-6 = pip trump; 7 doubles; 8 purged; 9 notrump


def pips(d: int) -> set[int]:
    return {int(DOMINO_HIGH[d]), int(DOMINO_LOW[d])}


def cell_state(cell: dict) -> ZebGameState:
    return ZebGameState(
        hands=tuple(tuple(h) for h in cell["hands"]),
        dealer=cell["dealer"],
        phase=GamePhase.PLAYING,
        bid_state=BidState(bids=tuple(cell["bids"]),
                           high_bidder=cell["high_bidder"],
                           high_bid=cell["high_bid"]),
        decl_id=cell["decl_id"],
        bidder=cell["bidder"],
        played=frozenset(d for _, d in cell["play_history"]),
        play_history=tuple((p, d) for p, d in cell["play_history"]),
        current_trick=tuple(cell["current_trick"]),
        trick_leader=cell["trick_leader"],
        team_points=tuple(cell["team_points"]),
    )


def led_suit_pip(cell: dict) -> int:
    from forge.oracle.tables import led_suit_for_lead_domino

    return int(led_suit_for_lead_domino(cell["current_trick"][0], cell["decl_id"]))


def history_legal(deal: list[list[int]], decl: int, bidder: int,
                  history: list[tuple[int, int]]) -> bool:
    """Replay the history on the twin deal, engine legality at every step."""
    state = GameStateTensor.from_deals([deal], [decl], device="cpu", bidders=[bidder])
    for (p, dom) in history:
        if int(state.current_player[0].item()) != p:
            return False
        slot = deal[p].index(dom)
        if not bool(state.legal_actions()[0, slot].item()):
            return False
        state = state.apply_actions(torch.tensor([slot], dtype=torch.long))
    return True


def find_twin(cell: dict) -> tuple[int, ...] | None:
    """Return twin hands tuple (3-2 swapped for a legal neutral tile), or None."""
    P = cell["actor"]
    decl = cell["decl_id"]
    played = {d for _, d in cell["play_history"]}
    forbidden = {2, 3, led_suit_pip(cell)}
    if decl in TRUMP_PIP_DECLS:
        forbidden.add(decl)
    for r in range(1, 4):
        seat = (P + r) % 4
        for t in cell["hands"][seat]:
            if t in played or DOMINO_COUNT_POINTS[t] > 0:
                continue
            if DOMINO_HIGH[t] == DOMINO_LOW[t]:  # double
                continue
            if pips(t) & forbidden:
                continue
            twin = [list(h) for h in cell["hands"]]
            twin[P][twin[P].index(TILE_32)] = t
            twin[seat][twin[seat].index(t)] = TILE_32
            if history_legal(twin, decl, cell["bidder"],
                             [tuple(x) for x in cell["play_history"]]):
                return tuple(tuple(h) for h in twin)
    return None


def retention_margin(prices: dict[int, float], g: int, comparators: list[int]) -> float:
    return max(prices[c] for c in comparators if c != g) - prices[g]


def fate_scores(net: OtisPlayNet, s: ZebGameState, candidates: list[int],
                P: int) -> tuple[dict[int, float], dict[int, float]]:
    """Per-candidate fate-ledger score and P(my team captures 3-2)."""
    from otis.model import TILE_VALUES

    feats = torch.stack([
        featurize_play_state(apply_action(s, s.hands[P].index(c)), perspective=P)
        for c in candidates
    ])
    with torch.no_grad():
        out = net(feats)
        fate_p = torch.softmax(out["fate"], dim=-1)          # [k, 5, 8]
        capture = fate_p[:, :, :4].sum(dim=-1)               # [k, 5]
        trick_p = torch.softmax(out["trick"], dim=-1)        # [k, 8]
        e_tricks = (trick_p * torch.arange(8, dtype=torch.float32)).sum(dim=-1)
    values = torch.tensor(TILE_VALUES, dtype=torch.float32)
    scores = (capture * values).sum(dim=-1) + e_tricks
    return (
        {c: float(scores[i]) for i, c in enumerate(candidates)},
        {c: float(capture[i, IDX_32]) for i, c in enumerate(candidates)},
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cells")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--m-worlds", type=int, default=50)
    args = ap.parse_args()

    cells = [json.loads(l) for l in open(args.cells)]
    print(f"{len(cells)} collected cells", flush=True)

    # Tied pricer: TiedSloughPlay's pricing verbatim, no arena harness.
    pricer = TiedSloughPlay.__new__(TiedSloughPlay)
    pricer._tied = load_tied_policy(
        str(Path(DEFAULT_STUDENT).resolve()
            if Path(DEFAULT_STUDENT).exists()
            else "/Users/jason/code/mk5-main/" + DEFAULT_STUDENT),
        args.device,
    )
    pricer._m = args.m_worlds
    pricer._world_sampler = None
    pricer.device = args.device

    net = OtisPlayNet.load("otis/models/otis_play_v0.pt", map_location="cpu")
    net.eval()

    out_path = Path(args.out)
    rows = []
    n_no_twin = 0
    t0 = time.time()
    for i, cell in enumerate(cells):
        twin_hands = find_twin(cell)
        if twin_hands is None:
            n_no_twin += 1
            continue
        sA = cell_state(cell)
        sB = dataclasses.replace(sA, hands=twin_hands)
        P, g = cell["actor"], cell["junk2"]
        cands = list(cell["candidates"])

        _, rA = pricer._price_and_choose(sA, cands, cell["default_dom"])
        _, rB = pricer._price_and_choose(sB, cands, cell["default_dom"])
        pA = {int(k): v for k, v in rA["prices"].items()}
        pB = {int(k): v for k, v in rB["prices"].items()}
        tied_premium = retention_margin(pA, g, cands) - retention_margin(pB, g, cands)

        fA, cap32A = fate_scores(net, sA, cands, P)
        fB, _ = fate_scores(net, sB, cands, P)
        fate_premium = retention_margin(fA, g, cands) - retention_margin(fB, g, cands)

        best_other = max((c for c in cands if c != g), key=lambda c: fA[c])
        g3_delta = cap32A[best_other] - cap32A[g]  # keep junk2 vs discard it

        row = {
            "i": i, "actor": P, "decl": cell["decl_id"], "junk2": g,
            "ply": len(cell["play_history"]),
            "n_cands": len(cands),
            "tied_premium": round(tied_premium, 3),
            "fate_premium": round(fate_premium, 4),
            "g3_delta_cap32": round(g3_delta, 4),
            "tied_margin_A": round(retention_margin(pA, g, cands), 3),
            "tied_margin_B": round(retention_margin(pB, g, cands), 3),
            "ess_A": rA["ess"], "ess_B": rB["ess"],
        }
        rows.append(row)
        with open(out_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        if (i + 1) % 25 == 0:
            print(f"  [{i + 1}/{len(cells)}] priced {len(rows)} "
                  f"(no-twin {n_no_twin}) t+{time.time() - t0:.0f}s", flush=True)

    n = len(rows)
    print(f"\n{n} priced cells ({n_no_twin} dropped: no legal twin) "
          f"in {time.time() - t0:.0f}s", flush=True)
    if n == 0:
        return 1
    for key, label in [("tied_premium", "G1 tied"), ("fate_premium", "G2 fate")]:
        vals = [r[key] for r in rows]
        mean = sum(vals) / n
        lo, hi = _bootstrap_ci_mean(vals)
        pos = sum(1 for v in vals if v > 0) / n
        print(f"{label}: mean {mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"positive {pos:.1%}  N={n}", flush=True)
    g3 = [r["g3_delta_cap32"] for r in rows]
    g3pos = sum(1 for v in g3 if v > 0) / n
    print(f"G3 mechanism: P(capture 3-2) higher when keeping junk2 in "
          f"{g3pos:.1%} of cells; mean delta {sum(g3) / n:+.4f}", flush=True)
    by_depth: dict[str, list[float]] = {}
    for r in rows:
        by_depth.setdefault(f"trick{r['ply'] // 4}", []).append(r["tied_premium"])
    for k in sorted(by_depth):
        v = by_depth[k]
        print(f"  {k}: n={len(v)} tied mean {sum(v) / len(v):+.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
