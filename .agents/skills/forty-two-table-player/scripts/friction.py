#!/usr/bin/env python3
"""Read and append fair-table friction logs."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_LOG = Path(os.environ.get("FORTY_TWO_FRICTION_LOG", "scratch/42-table/friction.jsonl"))


def _load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"kind": "parse_error", "observed": line})
    return rows


def _write(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def cmd_add(args: argparse.Namespace) -> None:
    row = {
        "ts": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "seat": args.seat,
        "turn": args.turn,
        "kind": args.kind,
        "severity": args.severity,
        "tool": args.tool,
        "observed": args.observed,
        "desired": args.desired,
        "workaround": args.workaround,
        "proposal": args.proposal,
    }
    row = {k: v for k, v in row.items() if v not in (None, "")}
    _write(args.path, row)
    print(json.dumps(row, indent=2, sort_keys=True))


def cmd_recent(args: argparse.Namespace) -> None:
    rows = _load(args.path)[-args.limit :]
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return
    if not rows:
        print(f"No friction logged at {args.path}.")
        return
    for row in rows:
        prefix = f"{row.get('ts', '?')} {row.get('seat', '?')} {row.get('kind', '?')} s{row.get('severity', '?')}"
        observed = row.get("observed", "")
        desired = row.get("desired", "")
        proposal = row.get("proposal", "")
        print(f"- {prefix}: {observed}")
        if desired:
            print(f"  desired: {desired}")
        if proposal:
            print(f"  proposal: {proposal}")


def cmd_summary(args: argparse.Namespace) -> None:
    rows = _load(args.path)
    by_kind = collections.Counter(str(r.get("kind", "unknown")) for r in rows)
    by_tool = collections.Counter(str(r.get("tool", "none")) for r in rows if r.get("tool"))
    payload = {
        "path": str(args.path),
        "count": len(rows),
        "by_kind": dict(by_kind.most_common()),
        "by_tool": dict(by_tool.most_common()),
        "recent": rows[-min(args.limit, len(rows)) :],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DEFAULT_LOG)
    sub = parser.add_subparsers(dest="cmd", required=True)

    add = sub.add_parser("add", help="Append one friction entry")
    add.add_argument("--seat", required=True)
    add.add_argument("--turn")
    add.add_argument("--kind", choices=["view", "tool", "prompt", "script", "rules", "book", "flow"], required=True)
    add.add_argument("--severity", type=int, choices=[1, 2, 3], default=1)
    add.add_argument("--tool")
    add.add_argument("--observed", required=True)
    add.add_argument("--desired", default="")
    add.add_argument("--workaround", default="")
    add.add_argument("--proposal", default="")
    add.set_defaults(func=cmd_add)

    recent = sub.add_parser("recent", help="Print recent friction entries")
    recent.add_argument("--limit", type=int, default=8)
    recent.add_argument("--json", action="store_true")
    recent.set_defaults(func=cmd_recent)

    summary = sub.add_parser("summary", help="Summarize friction by kind and tool")
    summary.add_argument("--limit", type=int, default=5)
    summary.set_defaults(func=cmd_summary)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
