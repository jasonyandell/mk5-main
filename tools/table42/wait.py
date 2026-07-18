#!/usr/bin/env python3
"""
wait — block until it's a seat's turn (or new chat / game end), then print
the seat's view and exit. One process per turn keeps an agent teammate's
loop at one tool call per decision.

Usage: python3 -u wait.py --run run/<gid> --seat N [--timeout 1800]
Exit codes: 0 = your turn / game end / new chat (view printed), 2 = timeout.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def render(v: dict, new_chat: list[dict]) -> str:
    names = v["seatNames"]
    out = [
        f"=== hand {v['hand_idx'] + 1} · {v['phase']} · marks {v['marks'][0]}-{v['marks'][1]} "
        f"(to {v['marks_to_win']}) · hand pts {v['team_points'][0]}-{v['team_points'][1]}"
        + (f" · trump {v['decl']}" if v["decl"] else ""),
        f"teams: {names[0]}&{names[2]} vs {names[1]}&{names[3]} — you are {v['name']} (seat {v['seat']})",
    ]
    if v["bids"]:
        out.append("bids: " + ", ".join(f"{n} {b}" for n, b in v["bids"].items()) +
                   (f" — {v['bidder']} owns it at {v['high_bid']}" if v["bidder"] else ""))
    for i, t in enumerate(v["tricks"]):
        out.append(f"trick {i + 1}: " + ", ".join(f"{p[0]} {p[1]}" for p in t["plays"]) +
                   f" — {t['winner']} +{t['points']}")
    if v["current_trick"]:
        out.append("current trick: " + ", ".join(f"{p[0]} {p[1]}" for p in v["current_trick"]))
    out.append(f"your hand: {' '.join(v['hand']) or '(empty)'}")
    if v["turn"]:
        out.append(f"YOUR TURN (turn_id {v['turn_id']}). Legal moves: " + ", ".join(v["menu"]))
    else:
        out.append(f"waiting on {v['actor']}" if v["actor"] else "")
    for m in new_chat:
        out.append(f"CHAT {m['name']}: {m['text']}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seat", type=int, required=True)
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()
    rundir = Path(args.run)
    viewf = rundir / f"view-seat-{args.seat}.json"
    seenf = rundir / f".wait-seen-{args.seat}.json"
    seen = json.loads(seenf.read_text()) if seenf.exists() else {"chat_id": 0, "turn_id": -1}

    start = time.time()
    while time.time() - start < args.timeout:
        try:
            v = json.loads(viewf.read_text())
        except Exception:
            time.sleep(1)
            continue
        new_chat = [m for m in v["chat"] if m["id"] > seen["chat_id"] and m["seat"] != v["seat"]]
        wake = (v["phase"] == "game_end"
                or (v["turn"] and v["turn_id"] != seen["turn_id"])
                or new_chat)
        if wake:
            if v["chat"]:
                seen["chat_id"] = max(seen["chat_id"], max(m["id"] for m in v["chat"]))
            if v["turn"]:
                seen["turn_id"] = v["turn_id"]
            seenf.write_text(json.dumps(seen))
            if v["phase"] == "game_end":
                print("GAME ENDED")
            print(render(v, new_chat))
            return
        time.sleep(1)
    print("wait timeout — table quiet; host may need a restart")
    sys.exit(2)



if __name__ == "__main__":
    main()
