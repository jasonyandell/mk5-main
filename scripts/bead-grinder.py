#!/usr/bin/env python3
"""
Simple bead grinder - processes beads one by one with live logging.
No worktrees, just runs in the current directory.

USAGE (run from project root):

    # Activate the virtual environment first:
    source .venv/bin/activate

    # Then run the script:
    python scripts/bead-grinder.py

    # When done, deactivate (optional):
    deactivate

ONE-LINER:
    source .venv/bin/activate && python scripts/bead-grinder.py
"""

from __future__ import annotations

import asyncio
import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Sequence

try:
    from claude_agent_sdk import (
        query,
        ClaudeAgentOptions,
        AssistantMessage,
        ResultMessage,
        TextBlock,
        ToolUseBlock,
    )
except ModuleNotFoundError as e:
    raise SystemExit(
        "Missing dependency: claude_agent_sdk\n"
        "Activate the repo venv first:\n"
        "  source .venv/bin/activate\n"
        "Then rerun:\n"
        "  python scripts/bead-grinder.py\n"
    ) from e

DEFAULT_EPIC_ID = "t42-21ze"


@dataclass(frozen=True)
class BdIssue:
    id: str
    title: str
    status: str
    priority: int | None
    issue_type: str
    created_at: str | None


def _run_bd_json(args: Sequence[str]) -> Any:
    result = subprocess.run(
        ["bd", *args, "--json"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if not result.stdout.strip():
        return None
    return json.loads(result.stdout)


def _parse_priority(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        s = value.strip().upper()
        if s.startswith("P"):
            s = s[1:]
        if s.isdigit():
            return int(s)
    return None


def list_epic_issues(epic_id: str, *, status: str = "open") -> list[BdIssue]:
    nodes = _run_bd_json(["dep", "tree", epic_id, "--direction=up", "--status", status])
    if not isinstance(nodes, list):
        raise RuntimeError(f"Unexpected bd output for epic {epic_id!r}: {type(nodes)}")

    issues: list[BdIssue] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        issue_id = node.get("id")
        if not isinstance(issue_id, str) or issue_id == epic_id:
            continue

        issues.append(
            BdIssue(
                id=issue_id,
                title=str(node.get("title") or ""),
                status=str(node.get("status") or ""),
                priority=_parse_priority(node.get("priority")),
                issue_type=str(node.get("issue_type") or ""),
                created_at=(str(node.get("created_at")) if node.get("created_at") else None),
            )
        )

    def sort_key(issue: BdIssue) -> tuple[int, str, str]:
        priority = issue.priority if issue.priority is not None else 999
        created_at = issue.created_at or ""
        return (priority, created_at, issue.id)

    return sorted(issues, key=sort_key)

def print_message(message):
    """Print messages in a readable format."""
    if isinstance(message, AssistantMessage):
        for block in message.content:
            if isinstance(block, TextBlock):
                print(block.text)
            elif isinstance(block, ToolUseBlock):
                print(f"  → {block.name}")
    elif isinstance(message, ResultMessage):
        status = "✅" if not message.is_error else "❌"
        print(f"\n{status} Done in {message.num_turns} turns (${message.total_cost_usd:.4f})")

async def grind_bead(bead_id: str):
    print(f"\n{'='*60}")
    print(f"🔮 Starting bead: {bead_id}")
    print(f"{'='*60}\n")

    options = ClaudeAgentOptions(
        allowed_tools=["Bash", "Read", "Edit", "Write"],
        permission_mode="acceptEdits",
    )

    async for message in query(
        prompt=f"Complete bead {bead_id}. Run 'bd show {bead_id}' first for context.",
        options=options
    ):
        print_message(message)

async def main():
    parser = argparse.ArgumentParser(description="Run Claude against a set of bd issues.")
    parser.add_argument(
        "--epic",
        default=DEFAULT_EPIC_ID,
        help=f"Epic issue ID to grind (default: {DEFAULT_EPIC_ID}).",
    )
    parser.add_argument(
        "--status",
        default="open",
        choices=["open", "in_progress", "blocked", "deferred", "closed"],
        help="Which issue status to include from the epic's dependent tree (default: open).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If set, only run the first N issues after sorting by priority/created_at.",
    )
    args = parser.parse_args()

    issues = list_epic_issues(args.epic, status=args.status)
    if args.limit and args.limit > 0:
        issues = issues[: args.limit]

    if not issues:
        print(f"No issues found under epic {args.epic} with status={args.status}.")
        return

    print(f"Epic: {args.epic}")
    print(f"Issues ({len(issues)}):")
    for issue in issues:
        p = f"P{issue.priority}" if issue.priority is not None else "P?"
        print(f"- {issue.id} [{p} {issue.status} {issue.issue_type}] {issue.title}")

    for issue in issues:
        await grind_bead(issue.id)

    print("\n🎉 All beads complete!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopped.")
