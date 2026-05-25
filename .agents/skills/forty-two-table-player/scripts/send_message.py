#!/usr/bin/env python3
"""Format fair-table messages for the referee or broker."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any


def _json_obj(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("value must be a JSON object")
    return value


def _emit(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    message = {
        "ts": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "to": args.to,
        "seat": args.seat,
        "turn": args.turn,
        **payload,
    }
    message = {k: v for k, v in message.items() if v not in (None, "")}
    text = json.dumps(message, indent=2, sort_keys=True)
    if args.outbox:
        args.outbox.parent.mkdir(parents=True, exist_ok=True)
        with args.outbox.open("a", encoding="utf-8") as f:
            f.write(json.dumps(message, sort_keys=True) + "\n")
    print(text)


def cmd_tool(args: argparse.Namespace) -> None:
    _emit(args, {"kind": "tool_request", "tool": args.name, "args": args.args})


def _ids(raw: str) -> list[int]:
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"not an integer domino id: {part}") from exc
    if not out:
        raise argparse.ArgumentTypeError("provide at least one domino id")
    return out


def cmd_labels(args: argparse.Namespace) -> None:
    _emit(args, {"kind": "tool_request", "tool": "domino_labels", "args": {"domino_ids": args.ids}})


def cmd_move(args: argparse.Namespace) -> None:
    _emit(
        args,
        {
            "kind": "commit_move",
            "domino_id": args.domino,
            "reason": args.reason,
            "confidence": args.confidence,
        },
    )


def cmd_propose_tool(args: argparse.Namespace) -> None:
    _emit(
        args,
        {
            "kind": "tool_proposal",
            "tool": args.name,
            "why": args.why,
            "inputs": args.inputs,
            "output": args.output,
        },
    )


def cmd_note(args: argparse.Namespace) -> None:
    _emit(args, {"kind": "seat_note", "scope": args.scope, "text": args.text})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seat", required=True, help="Seat label, e.g. P1")
    parser.add_argument("--turn", help="Optional hand/trick/turn label")
    parser.add_argument("--to", default="referee", help="Message destination")
    parser.add_argument("--outbox", type=Path, help="Optional JSONL outbox path")
    sub = parser.add_subparsers(dest="cmd", required=True)

    tool = sub.add_parser("tool", help="Format a broker tool request")
    tool.add_argument("--name", required=True)
    tool.add_argument("--args", type=_json_obj, default={})
    tool.set_defaults(func=cmd_tool)

    labels = sub.add_parser("labels", help="Request pip labels for domino ids")
    labels.add_argument("--ids", type=_ids, required=True, help="Comma-separated domino IDs, e.g. 14,19")
    labels.set_defaults(func=cmd_labels)

    move = sub.add_parser("move", help="Format a move commit")
    move.add_argument("--domino", type=int, required=True)
    move.add_argument("--reason", default="")
    move.add_argument("--confidence", choices=["low", "medium", "high"], default="medium")
    move.set_defaults(func=cmd_move)

    propose = sub.add_parser("propose-tool", help="Format a small tool or wrapper proposal")
    propose.add_argument("--name", required=True)
    propose.add_argument("--why", required=True)
    propose.add_argument("--inputs", default="")
    propose.add_argument("--output", default="")
    propose.set_defaults(func=cmd_propose_tool)

    note = sub.add_parser("note", help="Format a private/referee/table note")
    note.add_argument("--scope", choices=["private", "referee", "table"], default="referee")
    note.add_argument("--text", required=True)
    note.set_defaults(func=cmd_note)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
