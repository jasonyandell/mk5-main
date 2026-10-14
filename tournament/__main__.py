"""The tournament's front door.

    python -u -m tournament seed                      # append the roster's participants (idempotent)
    python -u -m tournament add-zeb 0.3.0             # pin a new npm release as a participant
    python -u -m tournament work --budget-s 240       # the worker loop (resumes any claim first)
    python -u -m tournament table                     # the derived rating table

`--store` picks where the records live: `hf:<repo id>` (the default,
hf:jasonyandell/texas-42-tournament) or `git:<path to a clone>`. The HF token is
huggingface_hub's own (HF_TOKEN, or its token file); see docs/SECRETS.md.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tournament.participants import ROSTER, zeb_npm  # noqa: E402
from tournament.store import GitStore, HfStore  # noqa: E402
from tournament.worker import log, seed_roster, table, work  # noqa: E402

DEFAULT_STORE = "hf:jasonyandell/texas-42-tournament"


def open_store(spec: str):
    kind, _, where = spec.partition(":")
    if kind == "hf":
        return HfStore(where, cache=ROOT / "scratch" / "tournament" / "hf" / where.replace("/", "__"))
    if kind == "git":
        return GitStore(where)
    raise SystemExit(f"--store must be hf:<repo> or git:<path>, not {spec!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=DEFAULT_STORE)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("seed", help="append the initial roster's participant records")
    z = sub.add_parser("add-zeb", help="pin @jasonyandell/zeb@<version> from npm as a participant")
    z.add_argument("version")
    z.add_argument("--worlds", type=int, default=160)
    w = sub.add_parser("work", help="run the worker loop")
    w.add_argument("--ladder", default="marks7")
    w.add_argument("--pairs", type=int, default=1, help="duplicate deal pairs per faceoff")
    w.add_argument("--budget-s", type=float, default=240.0, help="no new faceoff starts after this")
    w.add_argument("--max-faceoffs", type=int)
    w.add_argument("--only", help="comma-separated participant ids to schedule among")
    w.add_argument("--device", help="torch device for the fenced E[Q] seat (default: mps > cuda > cpu)")
    w.add_argument("--serial", action="store_true", help="play a faceoff's games one at a time")
    t = sub.add_parser("table", help="print the derived rating table")
    t.add_argument("--ladder", default="marks7")
    args = ap.parse_args()

    store = open_store(args.store)
    if args.cmd == "seed":
        seed_roster(store, ROSTER)
    elif args.cmd == "add-zeb":
        info = json.loads(subprocess.run(
            ["npm", "view", f"@jasonyandell/zeb@{args.version}", "dist", "--json"],
            check=True, capture_output=True, text=True).stdout)
        p = zeb_npm(args.version, info["integrity"], info["shasum"], worlds=args.worlds)
        seed_roster(store, [p])
    elif args.cmd == "work":
        only = set(args.only.split(",")) if args.only else None
        n = work(store, ladder=args.ladder, pairs=args.pairs, budget_s=args.budget_s,
                 max_faceoffs=args.max_faceoffs, only=only, parallel=not args.serial,
                 device=args.device)
        log(f"done: {n} faceoff(s) this run")
    elif args.cmd == "table":
        print(table(store, args.ladder))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
