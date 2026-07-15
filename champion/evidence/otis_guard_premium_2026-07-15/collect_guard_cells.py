"""Collect on-policy matched-pair cells for the otis guard-premium probe.

Runs incumbent-vs-incumbent (margin:wp(r8)+lens:ev both sides) with team A's
play wrapped in a state-dumping shadow: at every triggered slough decision
(the registered W6 predicate, borrowed verbatim from TiedSloughPlay) that
additionally qualifies for the guard contrast —

  actor still holds the 3-2 (unplayed) AND exactly one junk two among the
  candidates AND >= 1 non-two junk candidate AND decl not in {twos, threes}

— the full ZebGameState is dumped as JSON. Play is never altered (pure
shadow). Twin construction + pricing happen in price_guard_cells.py.

Usage:
    PYTHONPATH=$PWD python -u scratch/otis-night2/collect_guard_cells.py \
        --n-games 512 --base-seed 11000000 --out scratch/otis-night2/guard_cells.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from arena.cli import parse_bidder, parse_play  # noqa: E402
from arena.engine import ArenaConfig  # noqa: E402
from arena.lens_play import LensPlay  # noqa: E402
from arena.match import run_match, summarize  # noqa: E402
from arena.slough_override import TiedSloughPlay  # noqa: E402
from forge.zeb.game import current_player as zeb_current_player  # noqa: E402
from otis.fates import DOMINO_HIGH, DOMINO_LOW, pips_to_domino_id  # noqa: E402

TILE_32 = pips_to_domino_id("3-2")


def has_two(d: int) -> bool:
    return DOMINO_HIGH[d] == 2 or DOMINO_LOW[d] == 2


class GuardCellDump(LensPlay):
    """LensPlay + qualifying-cell state dump. Never changes the action."""

    _trigger_candidates = TiedSloughPlay._trigger_candidates

    def __init__(self, model, out_path: str, **kw):
        super().__init__(model, **kw)
        self._out = Path(out_path)
        self.n_triggers = 0
        self.n_cells = 0

    def choose(self, states, bid_values, marks=None, marks_to_win=7):
        actions = super().choose(states, bid_values, marks, marks_to_win)
        for i, s in enumerate(states):
            candidates = self._trigger_candidates(s, actions[i])
            if candidates is None:
                continue
            self.n_triggers += 1
            P = zeb_current_player(s)
            if s.decl_id in (2, 3):
                continue
            if TILE_32 not in s.hands[P] or TILE_32 in s.played:
                continue
            junk_twos = [c for c in candidates if has_two(c)]
            if len(junk_twos) != 1 or len(candidates) - 1 < 1:
                continue
            self.n_cells += 1
            cell = {
                "hands": [list(h) for h in s.hands],
                "dealer": s.dealer,
                "decl_id": s.decl_id,
                "bidder": s.bidder,
                "bids": list(s.bid_state.bids),
                "high_bidder": s.bid_state.high_bidder,
                "high_bid": s.bid_state.high_bid,
                "play_history": [list(x) for x in s.play_history],
                "current_trick": list(s.current_trick),
                "trick_leader": s.trick_leader,
                "team_points": list(s.team_points),
                "actor": P,
                "candidates": candidates,
                "junk2": junk_twos[0],
                "default_dom": s.hands[P][actions[i]],
            }
            with open(self._out, "a") as f:
                f.write(json.dumps(cell) + "\n")
        return actions


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-games", type=int, default=512)
    ap.add_argument("--base-seed", type=int, default=11000000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    from forge.zeb.eval.loading import DEFAULT_ORACLE, load_oracle

    root = Path(__file__).resolve().parent.parent.parent
    model = load_oracle(str(root / DEFAULT_ORACLE), args.device)
    bid_spec = "margin:wp,model=champion/margin_net_r8.pt"
    bid_a = parse_bidder(bid_spec, device=args.device, model=model, gus_adapter=None)
    bid_b = parse_bidder(bid_spec, device=args.device, model=model, gus_adapter=None)
    play_a = GuardCellDump(model, args.out, utility="ev",
                           n_samples=10, device=args.device)
    play_b = parse_play("lens:ev", model=model, n_samples=10,
                        device=args.device, seed=args.base_seed + 1)

    cfg = ArenaConfig(marks_to_win=7, max_redeals=3, base_seed=args.base_seed)
    print(f"Collecting guard cells: {args.n_games} games, seed {args.base_seed}",
          flush=True)
    t0 = time.time()
    result = run_match(bid_a=bid_a, bid_b=bid_b, play_a=play_a, play_b=play_b,
                       n_games=args.n_games, cfg=cfg,
                       label_a="incumbent+dump", label_b="incumbent",
                       verbose=True, fast_batching=True)
    s = summarize(result)
    print(f"done in {time.time() - t0:.0f}s: {s['n_hands']} hands, "
          f"{play_a.n_triggers} triggers, {play_a.n_cells} qualifying cells "
          f"-> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
