#!/usr/bin/env python3
"""
act — submit a move (with reasoning) or table talk to the host.

  python3 -u act.py move --run run/<gid> --seat N --turn-id X \
      --move "play 6-4" --reasoning "..."
  python3 -u act.py say --run run/<gid> --seat N --name Jeb --text "..."
"""
import argparse
import json
import time
from pathlib import Path

ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
mv = sub.add_parser("move")
for a in ("--run", "--move", "--reasoning"):
    mv.add_argument(a, required=True)
mv.add_argument("--seat", type=int, required=True)
mv.add_argument("--turn-id", type=int, required=True)
say = sub.add_parser("say")
for a in ("--run", "--name", "--text"):
    say.add_argument(a, required=True)
say.add_argument("--seat", type=int, required=True)
args = ap.parse_args()

run = Path(args.run)
if args.cmd == "move":
    (run / "moves" / f"seat-{args.seat}.json").write_text(json.dumps(
        {"turn_id": args.turn_id, "move": args.move, "reasoning": args.reasoning}))
    print(f"submitted turn {args.turn_id}: {args.move}")
else:
    f = run / "chat_in" / f"{args.seat}-{time.time_ns()}.json"
    f.write_text(json.dumps({"seat": args.seat, "name": args.name, "text": args.text}))
    print("said")
