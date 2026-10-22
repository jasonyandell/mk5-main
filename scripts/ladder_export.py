#!/usr/bin/env python
# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Export the marks7 ladder for its standings page: ratings, pairs, history, cost.

    python -u scripts/ladder_export.py [out dir, default scratch/ladder/page]

Writes ratings.json, pairs.json, history.json and summary.json, the datasets of
The 42 Ladder (https://claude.ai/artifact/TJYRL2ac7afwMookxGPdhp).
"""
import json, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tournament.__main__ import open_store, DEFAULT_STORE  # noqa: E402
from tournament.worker import read_state, ratings, ladder_anchor  # noqa: E402
from tournament.participants import supports_ladder  # noqa: E402
from tournament.rating import paired_margin  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else "scratch/ladder/page")
out.mkdir(parents=True, exist_ok=True)
ladder = "marks7"
store = open_store(DEFAULT_STORE)
participants, records = read_state(store, ladder)
participants = {k: p for k, p in participants.items() if supports_ladder(p, ladder)}
anchor = ladder_anchor(ladder)


def family(pid):
    for prefix, name in (("zeb", "Zeb"), ("demon", "Demon"), ("walt", "Walt"), ("eq", "E[Q]")):
        if pid.startswith(prefix):
            return name
    return "other"


# think cost: instructions each seat retires per game (FAIR CLOCKS reports it; never a budget)
ins = defaultdict(list)
for r in records:
    for g in r["games"]:
        c = g.get("cost") or {}
        for side in ("a", "b"):
            if f"{side}_self_instructions" in c:
                ins[r[side]].append(c[f"{side}_self_instructions"])

f = ratings(participants, records, anchor)
rows = []
for i, r in enumerate(x for x in f.ratings if x.games > 0):
    a = r.player == anchor
    rows.append({"rank": i + 1, "player": r.player, "family": family(r.player),
                 "elo": round(r.elo, 1), "lo": round(r.elo if a else r.lo, 1), "hi": round(r.elo if a else r.hi, 1),
                 "games": r.games, "wins": r.wins, "win_pct": round(100 * r.wins / r.games, 1),
                 "ginstr_per_game": round(sum(ins[r.player]) / len(ins[r.player]) / 1e9, 2) if ins[r.player] else None,
                 "anchor": a})

pairs, wins = {}, {}
for rec in records:
    a, b = rec["a"], rec["b"]
    key, sign = ((a, b), 1) if a < b else ((b, a), -1)
    by = defaultdict(list)
    for g in rec["games"]:
        by[g["pair"]].append(g)
    for gs in by.values():
        pairs.setdefault(key, []).append(sign * sum(g["a_marks"] - g["b_marks"] for g in gs) / len(gs))
    w = wins.setdefault(key, [0, 0])
    for g in rec["games"]:
        w[0 if (g["a_won"] == (sign == 1)) else 1] += 1
prow = []
for (x, y), d in sorted(pairs.items()):
    m, lo, hi = paired_margin(d)
    ok = len(d) >= 5  # an interval from a handful of pairs estimates its own width badly
    prow.append({"pair": f"{x} vs {y}", "row": x, "col": y, "dup_pairs": len(d), "games": sum(wins[(x, y)]),
                 "row_wins": wins[(x, y)][0], "col_wins": wins[(x, y)][1], "marks_per_game": round(m, 3),
                 "lo": round(lo, 3) if ok else None, "hi": round(hi, 3) if ok else None})
    prow.append({"pair": f"{y} vs {x}", "row": y, "col": x, "dup_pairs": len(d), "games": sum(wins[(x, y)]),
                 "row_wins": wins[(x, y)][1], "col_wins": wins[(x, y)][0], "marks_per_game": round(-m, 3),
                 "lo": round(-hi, 3) if ok else None, "hi": round(-lo, 3) if ok else None})

days = sorted({r["claimed_utc"][:8] for r in records})
hist = []
for day in days:
    sub = [r for r in records if r["claimed_utc"][:8] <= day]
    fd = ratings(participants, sub, anchor)
    n_games = sum(len(r["games"]) for r in sub)
    for r in fd.ratings:
        if r.games > 0:
            a = r.player == anchor
            hist.append({"day": f"{day[:4]}-{day[4:6]}-{day[6:]}", "player": r.player, "family": family(r.player),
                         "elo": round(r.elo, 1), "lo": round(r.elo if a else r.lo, 1),
                         "hi": round(r.elo if a else r.hi, 1), "games": r.games,
                         "faceoffs_to_date": len(sub), "games_to_date": n_games})

summary = [{"ladder": ladder, "anchor": anchor, "faceoffs": len(records),
            "games": sum(len(r["games"]) for r in records), "players": len(rows),
            "first_day": days[0], "last_day": days[-1]}]
for name, data in (("ratings", rows), ("pairs", prow), ("history", hist), ("summary", summary)):
    (out / f"{name}.json").write_text(json.dumps(data))
    print(name, len(data), (out / f"{name}.json").stat().st_size)
