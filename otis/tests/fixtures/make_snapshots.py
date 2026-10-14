"""Regenerate otis/tests/fixtures/selfplay_0.snapshots.json (five arena hands).

    PYTHONPATH=. python -u otis/tests/fixtures/make_snapshots.py   # from the repo root

The hands come from the arena's own CPU engine (heuristic bidders, seeded random
play), so the recorded points are the engine's, independent of the fate parser.
They are chosen so tests/test_snapshots.py has what it derives by hand:
  row 0  a sixes contract on which 6-4 leads a trick and wins it by trump power,
  row 2  a 42-0 sweep,
  rows 1, 3, 4  the next ordinary hands.
The output is deterministic (int-tuple hashing is not randomised).
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from arena.bidders import HeuristicBidder
from arena.engine import ArenaConfig, run_half
from arena.match import snapshot_rows
from arena.play import RandomPlay
from otis.fates import NeutralGame, parse_game_fates, pips_to_domino_id
from otis.snapshots import load_snapshot_hands

OUT = Path(__file__).resolve().parent / "selfplay_0.snapshots.json"


def _six_four_leads_and_wins(row: dict) -> bool:
    if row["decl_id"] != 6:  # sixes
        return False
    tid = pips_to_domino_id("6-4")
    i = next(k for k, (_, d) in enumerate(row["plays"]) if d == tid)
    if i % 4 != 0:
        return False
    holder = next(s for s, h in enumerate(row["hands"]) if tid in h)
    g = NeutralGame("f", row["hands"], row["decl_id"], row["bidder"], row["bid_value"],
                    [tuple(p) for p in row["plays"]])
    t = next(t for t in parse_game_fates(g).tiles if t.tile == "6-4")
    return t.won_by_trump and t.winner_seat == holder


def main() -> None:
    games = run_half(
        n_games=40, a_team=0, cfg=ArenaConfig(marks_to_win=2, base_seed=7),
        bid_a=HeuristicBidder(), bid_b=HeuristicBidder(),
        play_a=RandomPlay(seed=1), play_b=RandomPlay(seed=2),
    )
    rows = snapshot_rows(SimpleNamespace(games=games))
    sweep = next(r for r in rows if 0 in (r["bidder_team_pts"], r["opp_team_pts"]))
    sixes = next(r for r in rows if _six_four_leads_and_wins(r))
    rest = [r for r in rows if r is not sweep and r is not sixes][:3]
    chosen = [sixes, rest[0], sweep, rest[1], rest[2]]
    OUT.write_text(json.dumps({"snapshots": chosen, "metadata": {"base_seed": 7}}) + "\n")
    assert len(list(load_snapshot_hands(OUT))) == 5


if __name__ == "__main__":
    main()
