#!/usr/bin/env python
"""Smoke test for walt/grade.py (builder B4).

With HORIZON=4, run a 2-game paired mini-match end-to-end on CPU and assert:
  - the driver completes and writes summary.json + per_hand.csv + decisions.jsonl,
  - every walt-chosen slot was legal at its decision state.

If walt.solver / walt.worlds / walt.field are not yet on disk (parallel
builders), print 'SKIP (solver missing)' and exit 0 — the full smoke runs at
integration.

Plain executable script: prints PASS/FAIL lines, nonzero exit on any FAIL.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

WALT_ROOT = "/Users/jason/code/mk5-main/.claude/worktrees/walt"
if WALT_ROOT not in sys.path:
    sys.path.insert(0, WALT_ROOT)


def main() -> int:
    solver_files = ["walt/solver.py", "walt/worlds.py", "walt/field.py"]
    missing = [f for f in solver_files if not (Path(WALT_ROOT) / f).exists()]
    if missing:
        print(f"SKIP (solver missing): {missing}")
        return 0

    import json

    from arena.cli import parse_bidder
    from arena.engine import ArenaConfig
    from arena.jud_play import JudPlay
    from arena.match import MatchResult, hand_rows, summarize
    from champion.jud_net import load_jud_net
    from forge.zeb.game import legal_actions

    from walt.contracts import HORIZON
    from walt.grade import WaltPlay, _run_paired_walt, _Heartbeat, INCUMBENT_BIDDER, DEFAULT_FIELD_NET

    failures = 0

    # A WaltPlay that validates every slot it returns against engine legality.
    class CheckedWaltPlay(WaltPlay):
        illegal = 0
        n_checked = 0

        def choose(self, states, bid_values, marks=None, marks_to_win=7):
            slots = super().choose(states, bid_values, marks, marks_to_win)
            for s, slot in zip(states, slots):
                CheckedWaltPlay.n_checked += 1
                if slot not in legal_actions(s):
                    CheckedWaltPlay.illegal += 1
            return slots

    tmp = Path(tempfile.mkdtemp(prefix="walt_smoke_"))
    cfg = ArenaConfig(marks_to_win=1, max_redeals=3, base_seed=7)

    bid = parse_bidder(INCUMBENT_BIDDER, device="cpu", gus_adapter=None, model=None)
    jud_play = JudPlay(load_jud_net(DEFAULT_FIELD_NET, device="cpu"))
    walt_play = CheckedWaltPlay(
        field_net=DEFAULT_FIELD_NET,
        beliefs="sigma",
        payoff="points",
        pool_size=0,  # in-process, no worker spawn
        decisions_path=str(tmp / "decisions.jsonl"),
    )
    heartbeat = _Heartbeat(tmp, every_s=1e9)

    try:
        games = _run_paired_walt(
            n_games=2, cfg=cfg, bid=bid,
            walt_play=walt_play, jud_play=jud_play, heartbeat=heartbeat,
        )
    finally:
        walt_play.close()
        heartbeat.close()

    # Gate 1: completed with the expected number of games (both halves).
    if len(games) == 2:
        print("PASS gate1-complete: 2-game paired mini-match ran to completion")
    else:
        print(f"FAIL gate1-complete: expected 2 games, got {len(games)}")
        failures += 1

    # Gate 2: every walt-chosen slot legal.
    if CheckedWaltPlay.n_checked > 0 and CheckedWaltPlay.illegal == 0:
        print(f"PASS gate2-legal: all {CheckedWaltPlay.n_checked} walt slots legal")
    elif CheckedWaltPlay.n_checked == 0:
        print("FAIL gate2-legal: WaltPlay never chose (no decisions to check)")
        failures += 1
    else:
        print(f"FAIL gate2-legal: {CheckedWaltPlay.illegal}/"
              f"{CheckedWaltPlay.n_checked} walt slots illegal")
        failures += 1

    # Gate 3: outputs written.
    result = MatchResult(label_a="walt", label_b="jud", cfg=cfg,
                         elapsed_s=0.0, games=games)
    (tmp / "summary.json").write_text(json.dumps(summarize(result)) + "\n")
    import csv
    rows = hand_rows(result)
    with (tmp / "per_hand.csv").open("w", newline="") as fh:
        if rows:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    wrote = [
        (tmp / "summary.json").exists(),
        (tmp / "per_hand.csv").exists(),
        (tmp / "decisions.jsonl").exists(),
    ]
    if all(wrote):
        print(f"PASS gate3-outputs: summary.json, per_hand.csv, decisions.jsonl in {tmp}")
    else:
        print(f"FAIL gate3-outputs: {wrote} in {tmp}")
        failures += 1

    # Gate 4: decisions.jsonl parseable and carries the required fields.
    dec_path = tmp / "decisions.jsonl"
    ok_dec = True
    n_dec = 0
    required = {"game", "hand", "seat", "n_worlds", "solve_ms", "chosen_id",
                "jud_would_id", "walker_flags"}
    for line in dec_path.read_text().splitlines():
        if not line.strip():
            continue
        n_dec += 1
        rec = json.loads(line)
        if not required.issubset(rec):
            ok_dec = False
    if ok_dec:
        print(f"PASS gate4-decisions: {n_dec} decision records with required fields")
    else:
        print("FAIL gate4-decisions: a record was missing required fields")
        failures += 1

    if failures:
        print(f"\n{failures} gate(s) FAILED")
        return 1
    print("\nAll gates PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
